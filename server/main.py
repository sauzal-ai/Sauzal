"""
Sauzal · Control Plane (servidor central)
==========================================

Este es el "cerebro" de la red Sauzal. Es una API HTTP (FastAPI) que corre
en un solo lugar (por ejemplo, tu PC o un VPS) y coordina a los nodos
remotos (PCs con GPU corriendo agent.py) y a los clientes que piden
inferencias (texto vía Ollama o imagenes vía ComfyUI/FLUX).

Arquitectura general
---------------------

    cliente (client/infer.py)
         |
         |  POST /infer  (pide una inferencia)
         v
    servidor (este archivo)  <-- guarda todo en SQLite (sauzal.db)
         ^
         |  heartbeat + pull + result
         |
    agente remoto (agent/agent.py, corre en la PC/Pod con GPU)

El agente SIEMPRE inicia la conexión (heartbeat y pull de trabajos). El
servidor nunca se conecta hacia el agente. Esto permite que los nodos
remotos vivan detrás de NAT/firewalls sin necesidad de IP publica ni
puertos abiertos entrantes — solo necesitan salida a internet.

Ciclo de vida de un nodo
------------------------
1. El agente llama a POST /nodes/register una sola vez y recibe un
   (node_id, token) que guarda localmente (agent_config.json). Este par
   funciona como credencial: todas las llamadas siguientes del agente
   deben incluirlo para probar que son ese nodo.
2. Cada pocos segundos el agente llama a POST /nodes/heartbeat para avisar
   "sigo vivo" y actualizar sus capacidades (que modelos tiene disponibles).
   Un nodo se considera "online" en /nodes si tuvo heartbeat en los
   ultimos 30 segundos (ver el `now-30` repetido en varias queries).
3. En el mismo ciclo, el agente llama a POST /agent/pull para preguntar
   "¿tenes un trabajo para mi?". Si hay uno en cola asignado a ese nodo,
   se lo entrega y lo marca como "running".
4. Cuando el agente termina, llama a POST /jobs/{job_id}/result con el
   resultado (o el error), y el servidor libera al nodo (status=available
   de nuevo).

Ciclo de vida de un trabajo (job)
----------------------------------
POST /infer crea la fila en la tabla `jobs` con status="queued" y la
asigna a un nodo disponible (el mas recientemente visto, o uno especifico
si el cliente pidio `node=<nombre o id>`). El cliente despues hace polling
de GET /jobs/{job_id} hasta ver status="completed" o "failed".

Persistencia
------------
Se usa SQLite (`sauzal.db`, en la misma carpeta que este archivo) por
simplicidad: alcanza para un PoC de un solo proceso. No es apto para
multiples instancias del servidor corriendo en paralelo (SQLite no
soporta bien escrituras concurrentes desde varios procesos).

Limitaciones conocidas (ver README, seccion "No usar en produccion")
----------------------------------------------------------------------
- No hay autenticacion de clientes en /infer: cualquiera que llegue al
  servidor puede pedir inferencias. La autenticacion con token solo
  protege los endpoints que usan los AGENTES (heartbeat, pull, result).
- No hay HTTPS propio (si se necesita, se resuelve con un proxy/tunel
  delante, como hicimos con Cloudflare Tunnel).
- No hay reintentos: si un nodo se cae con un trabajo "running" asignado,
  ese trabajo queda huerfano para siempre (nadie lo reasigna).
"""

from __future__ import annotations
import html, ipaddress, json, os, sqlite3, time, urllib.parse, uuid
from contextlib import contextmanager
from pathlib import Path
import requests
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import memory

# Base de datos SQLite, vive al lado de este archivo (server/sauzal.db) por
# defecto. Configurable via SAUZAL_DB_PATH para que los tests (ver tests/)
# puedan apuntar a un archivo temporal sin tocar la base real -- en uso
# normal (sin la variable de entorno seteada) el comportamiento es identico
# al de siempre.
DB_PATH = Path(os.environ.get("SAUZAL_DB_PATH") or Path(__file__).with_name("sauzal.db"))
app = FastAPI(title="Sauzal Control Plane", version="0.2.0")


@contextmanager
def db():
    """
    Context manager que abre una conexion a SQLite, hace commit al salir
    del bloque `with` si no hubo excepciones, y siempre cierra la conexion.

    Uso: `with db() as conn: conn.execute(...)`.

    `row_factory = sqlite3.Row` hace que las filas se puedan tratar como
    diccionarios (`row["campo"]`) en vez de tuplas posicionales.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def client_ip(request: Request) -> str:
    """
    IP real de quien hizo la request, con un cuidado especial: si el
    servidor esta detras de un tunel/proxy (como el Cloudflare Tunnel que
    se uso para exponer este servidor a los Pods de RunPod), la conexion
    TCP que ve FastAPI es siempre la del proceso del tunel en localhost
    (127.0.0.1), no la del nodo/cliente real. Cloudflare agrega el header
    `CF-Connecting-IP` con la IP original; otros proxies suelen usar
    `X-Forwarded-For`. Se prueban esos headers primero y se cae a la IP
    de conexion directa si no estan presentes.

    Importante: estos headers los puede mandar CUALQUIERA si se le pega
    directo al servidor sin pasar por un proxy de confianza (no hay nada
    validando que vengan realmente de Cloudflare) -- es el mismo criterio
    de confianza que el resto de este servidor (sin autenticacion en
    /infer, nombres de nodo auto-declarados, etc.), no un dato a prueba
    de falsificacion.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "desconocida"


# Cache en memoria de IP -> "Ciudad, Pais" (o None si no se pudo resolver),
# para no golpear la API externa de geolocalizacion en cada heartbeat con
# la misma IP repetida. Se pierde si se reinicia el servidor -- no hace
# falta persistirla, es barato volver a calcularla.
_GEO_CACHE: dict[str, str | None] = {}

def geolocate(ip: str) -> str | None:
    """
    Resuelve una IP a "Ciudad, Pais" usando ip-api.com (servicio gratuito,
    sin necesidad de API key). Devuelve None para:
      - IPs privadas/de loopback (192.168.x.x, 10.x.x.x, 127.0.0.1, etc.):
        no tiene sentido geolocalizar una IP de LAN, y evita gastar
        cuota del servicio externo en vano.
      - Cualquier error de red o respuesta invalida del servicio.

    Los resultados se cachean en memoria por IP (`_GEO_CACHE`): la
    primera vez que se ve una IP nueva se hace la consulta real, las
    siguientes veces se devuelve el valor ya guardado.
    """
    if ip in _GEO_CACHE:
        return _GEO_CACHE[ip]
    try:
        if ipaddress.ip_address(ip).is_private:
            _GEO_CACHE[ip] = None
            return None
    except ValueError:
        _GEO_CACHE[ip] = None
        return None
    try:
        r = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,country,regionName,city"},
            timeout=5,
        )
        data = r.json()
        if data.get("status") == "success":
            parts = [p for p in (data.get("city"), data.get("country")) if p]
            location = ", ".join(parts) or None
        else:
            location = None
    except Exception:
        location = None
    _GEO_CACHE[ip] = location
    return location


def _pick_available_node(conn, node_hint: str | None = None):
    """
    Elige un nodo disponible (online, no pausado, no ocupado) para
    asignarle un job. Si `node_hint` viene (nombre o node_id), se
    restringe la busqueda a ESE nodo puntual. Devuelve la fila del nodo o
    None si no hay ninguno que cumpla.

    Extraido a una funcion propia porque lo necesitan dos casos: el
    /infer normal de un cliente, y el resumen automatico de sesiones
    (`_maybe_enqueue_summary()`), que tambien tiene que encolarle un job
    a *algun* nodo disponible.
    """
    now = time.time()
    if node_hint:
        return conn.execute("""
          SELECT node_id FROM nodes
          WHERE status='available' AND last_seen>? AND COALESCE(paused,0)=0
            AND (node_id=? OR name=?)
          ORDER BY last_seen DESC LIMIT 1
        """, (now-30, node_hint, node_hint)).fetchone()
    return conn.execute("""
      SELECT node_id FROM nodes
      WHERE status='available' AND last_seen>? AND COALESCE(paused,0)=0
      ORDER BY last_seen DESC LIMIT 1
    """, (now-30,)).fetchone()


def _is_image_model(model: str) -> bool:
    """
    Misma heuristica que `agent/comfy.py::is_image_model()` (por prefijo
    de nombre), duplicada aca a proposito: el servidor la necesita para
    decidir si tiene sentido inyectarle memoria de conversacion (texto) a
    un pedido, SIN tener que importar el paquete del agente (que ademas
    asume que ComfyUI esta corriendo localmente, cosa que no aplica en el
    servidor). Si algun dia se agrega un tercer prefijo de modelo de
    imagen, hay que actualizar los dos lugares.
    """
    m = (model or "").lower()
    return m.startswith("flux") or m.startswith("sdxl") or m.startswith("sauzal-image")


@app.on_event("startup")
def startup():
    """
    Se ejecuta una vez cuando arranca el servidor (uvicorn). Crea las
    tablas si todavia no existen (CREATE TABLE IF NOT EXISTS), asi que es
    seguro reiniciar el servidor sin perder datos ni romper nada.

    Tabla `nodes`: un registro por cada agente que se registro alguna vez.
        node_id       -- UUID, identifica al nodo (PK)
        name          -- nombre elegido con --name al arrancar el agente
        token         -- credencial secreta del nodo (par con node_id)
        status        -- "available" | "busy" (lo actualiza el propio nodo)
        last_seen     -- timestamp (epoch) del ultimo heartbeat
        capabilities  -- JSON en texto: hardware/software del nodo (VRAM, RAM,
                         driver, motor de computo, modelos, versiones,
                         benchmark, temperatura, consumo -- ver
                         agent.py::capabilities() para el detalle completo
                         de que trae cada campo y como se mide)
        paused        -- 0/1. Disponibilidad: si esta en 1, /infer nunca le
                         asigna trabajos aunque este online. Se puede fijar
                         como preferencia por defecto desde el agente
                         (--paused al registrarse) y despues sobreescribir
                         en cualquier momento desde /admin (el heartbeat NO
                         toca esta columna, para que la decision del panel
                         persista aunque el agente se reconecte)
        price_kwh     -- precio de la electricidad para ESTE nodo (moneda
                         libre, la que use el operador), cargado a mano en
                         /admin. NULL si no se configuro -- en ese caso no
                         se puede estimar el costo, solo el consumo en W
        energy_wh     -- energia acumulada estimada, en Watt-hora. El
                         agente manda un incremento en cada heartbeat
                         (consumo instantaneo x tiempo transcurrido desde
                         el heartbeat anterior) y el servidor lo va sumando.
                         Es una ESTIMACION (no un medidor real), y solo se
                         acumula mientras el agente esta corriendo y
                         reportando un consumo valido (GPUs no-NVIDIA no
                         reportan consumo, ver limitaciones)
        latency_ms    -- tiempo de ida y vuelta (ms) del ultimo heartbeat,
                         medido por el propio agente. Sirve como proxy de
                         "que tan lejos" esta ese nodo del servidor
        ip_address    -- IP real del nodo (ver client_ip()), actualizada
                         en cada register/heartbeat
        location      -- "Ciudad, Pais" resuelto de ip_address via
                         geolocate() (ver esa funcion), o NULL si la IP
                         es privada/LAN o no se pudo resolver

    Tabla `jobs`: un registro por cada pedido de inferencia.
        job_id             -- UUID (PK)
        model              -- nombre del modelo pedido (ej: "gemma3:4b", "flux...")
        prompt             -- el prompt de texto o imagen
        status             -- "queued" -> "running" -> "completed" | "failed"
        assigned_node      -- a que nodo se le asigno el trabajo
        client             -- quien lo pidio (dato libre que manda el cliente,
                              ej: su hostname; NO es una identidad autenticada,
                              ver limitaciones mas abajo)
        client_ip          -- IP real de quien mando el pedido (ver client_ip())
        client_location    -- "Ciudad, Pais" resuelta de client_ip, o NULL
        client_user_agent  -- header User-Agent de la request a /infer. Revela
                              si vino de un navegador (contiene "Mozilla/...")
                              o de un script (ej: "python-requests/2.32.4")
                              SIN que el cliente tenga que declarar nada
        client_os          -- sistema operativo del cliente (dato libre que
                              manda client/infer.py, ej: "Windows 10")
        client_processor   -- procesador/arquitectura del cliente (idem)
        result             -- JSON en texto con la respuesta (si completed)
        error              -- texto del error (si failed)
        prompt_tokens      -- tokens de entrada consumidos (solo jobs de texto;
                              NULL en jobs de imagen, que no tienen ese concepto)
        output_tokens      -- tokens de salida generados (idem)
        duration_ms        -- cuanto tardo la ejecucion real, en milisegundos.
                              Lo mide el AGENTE (cronometrando execute(job) de
                              punta a punta) y lo manda tanto si el job salio
                              bien como si fallo -- asi que tambien sirve para
                              ver cuanto tardo en fallar/timeoutear un job
        session_id         -- si el pedido pertenece a una conversacion con
                              memoria (ver mas abajo), a que sesion. NULL en
                              pedidos sueltos (stateless), que siguen
                              funcionando exactamente igual que siempre
        kind               -- 'chat' (un pedido normal) | 'summarize' (un job
                              INTERNO que el propio servidor encola para
                              comprimir mensajes viejos de una sesion, ver
                              memory.py). El agente no distingue nada especial:
                              para el, un job 'summarize' es un prompt de texto
                              mas, igual que cualquier otro
        context_tokens_estimate/context_messages_used/context_semantic_hits
                           -- metricas de cuanto contexto de la sesion se le
                              inyecto al prompt de este job en particular
                              (ver memory.py::build_context()). NULL en
                              pedidos sin sesion o de tipo 'summarize'
        summarize_message_ids -- (solo jobs kind='summarize') JSON con los
                              message_id que este resumen puntual va a cubrir;
                              al volver el resultado, esos mensajes se marcan
                              summarized=1 y el texto va a sessions.summary
        created_at / updated_at -- timestamps (epoch)

    Tabla `sessions` y `messages`: implementan la memoria de conversacion
    (ver server/memory.py para el detalle completo de la logica). Un
    "session_id" agrupa una conversacion que puede abarcar VARIOS jobs,
    cada uno potencialmente ejecutado por un nodo/GPU distinto -- el
    contexto para que la conversacion tenga continuidad lo arma y guarda
    el SERVIDOR, nunca el agente.

        sessions.session_id     -- UUID (PK), lo devuelve POST /sessions
        sessions.tenant_id      -- reservado para multi-tenant futuro (hoy
                                    siempre NULL, no se usa para filtrar nada)
        sessions.created_at / last_used_at -- timestamps (epoch)
        sessions.summary        -- resumen acumulado de los mensajes viejos
                                    de la sesion (memoria de largo plazo),
                                    generado automaticamente por un job
                                    kind='summarize' cuando hace falta
                                    (ver memory.py::SUMMARIZE_TRIGGER)

        messages.message_id     -- UUID (PK)
        messages.session_id     -- FK logica a sessions.session_id
        messages.role           -- 'user' | 'assistant'
        messages.content        -- texto del mensaje, tal cual
        messages.created_at     -- timestamp (epoch)
        messages.trivial        -- 0/1, si el mensaje es demasiado corto/generico
                                    para servir como memoria semantica (ver
                                    memory.py::is_trivial()) -- igual se guarda,
                                    solo se excluye de la busqueda semantica
        messages.token_estimate -- estimacion de tokens de ese mensaje
        messages.summarized     -- 0/1, si ya fue incorporado al resumen
                                    acumulado de la sesion

    Migracion liviana: si `sauzal.db` ya existia de una version anterior
    de este archivo (sin alguna de las columnas nuevas), se le agregan con
    ALTER TABLE la primera vez que arranca el servidor con este codigo. No
    hace falta borrar la base a mano.
    """
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
          node_id TEXT PRIMARY KEY, name TEXT, token TEXT, status TEXT,
          last_seen REAL, capabilities TEXT);
        CREATE TABLE IF NOT EXISTS jobs(
          job_id TEXT PRIMARY KEY, model TEXT, prompt TEXT, status TEXT,
          assigned_node TEXT, result TEXT, error TEXT,
          created_at REAL, updated_at REAL);
        CREATE TABLE IF NOT EXISTS sessions(
          session_id TEXT PRIMARY KEY, tenant_id TEXT,
          created_at REAL, last_used_at REAL, summary TEXT);
        CREATE TABLE IF NOT EXISTS messages(
          message_id TEXT PRIMARY KEY, session_id TEXT,
          role TEXT, content TEXT, created_at REAL,
          trivial INTEGER DEFAULT 0, token_estimate INTEGER,
          summarized INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
        """)
        existing_job_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, coltype in (
            ("client", "TEXT"),
            ("prompt_tokens", "INTEGER"),
            ("output_tokens", "INTEGER"),
            ("client_ip", "TEXT"),
            ("client_location", "TEXT"),
            ("client_user_agent", "TEXT"),
            ("client_os", "TEXT"),
            ("client_processor", "TEXT"),
            ("duration_ms", "REAL"),
            ("session_id", "TEXT"),
            ("kind", "TEXT DEFAULT 'chat'"),
            ("context_tokens_estimate", "INTEGER"),
            ("context_messages_used", "INTEGER"),
            ("context_semantic_hits", "INTEGER"),
            ("summarize_message_ids", "TEXT"),
        ):
            if column not in existing_job_cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {coltype}")

        existing_node_cols = {r["name"] for r in conn.execute("PRAGMA table_info(nodes)").fetchall()}
        for column, coltype in (
            ("paused", "INTEGER DEFAULT 0"),
            ("price_kwh", "REAL"),
            ("energy_wh", "REAL DEFAULT 0"),
            ("latency_ms", "REAL"),
            ("ip_address", "TEXT"),
            ("location", "TEXT"),
        ):
            if column not in existing_node_cols:
                conn.execute(f"ALTER TABLE nodes ADD COLUMN {column} {coltype}")


# ----------------------------------------------------------------------
# Modelos Pydantic: definen y validan el JSON que entra en cada endpoint.
# ----------------------------------------------------------------------

class Register(BaseModel):
    """Body de POST /nodes/register: como se presenta un agente nuevo."""
    name: str            # nombre legible elegido por el agente (--name)
    capabilities: str    # JSON en texto con lo que el nodo puede hacer
    paused: bool = False # preferencia de disponibilidad por defecto (--paused del agente)

class Auth(BaseModel):
    """
    Credenciales que debe mandar un agente en cada llamada autenticada
    (heartbeat, pull, result). Se heredan estos dos campos en los modelos
    de abajo que necesitan identificar de que nodo viene la request.
    """
    node_id: str
    token: str

class Heartbeat(Auth):
    """Body de POST /nodes/heartbeat: 'sigo vivo' + estado actual."""
    status: str = "available"
    capabilities: str | None = None  # si viene, actualiza las capacidades guardadas
    latency_ms: float | None = None  # ida y vuelta del heartbeat anterior, medido por el agente
    energy_wh_delta: float | None = None  # energia consumida (Wh) desde el heartbeat anterior, a sumar al acumulado

class Infer(BaseModel):
    """Body de POST /infer: lo que pide un cliente."""
    model: str = "gemma3:4b"      # modelo de texto (Ollama) o de imagen (ComfyUI)
    prompt: str = Field(min_length=1)
    node: str | None = None       # opcional: forzar un nodo puntual (nombre o node_id)
    client: str | None = None     # quien lo pide (dato libre, ej: hostname del cliente; no autenticado)
    client_os: str | None = None          # SO del cliente (dato libre, self-reportado)
    client_processor: str | None = None   # procesador/arquitectura del cliente (idem)
    session_id: str | None = None         # opcional: si viene, este pedido pertenece a una
                                           # conversacion con memoria (ver server/memory.py).
                                           # Sin esto, el comportamiento es 100% el de siempre.

class Result(Auth):
    """Body de POST /jobs/{job_id}/result: lo que devuelve el agente al terminar."""
    success: bool
    result: str | None = None   # JSON en texto con la respuesta, si success=True
    error: str | None = None    # texto del error, si success=False
    prompt_tokens: int | None = None   # tokens de entrada, si el backend los reporta (Ollama si, ComfyUI no)
    output_tokens: int | None = None   # tokens de salida, idem
    duration_ms: float | None = None   # tiempo real de ejecucion, medido por el agente (exito o fallo)


def authenticate(conn, node_id, token):
    """
    Verifica que (node_id, token) corresponda a un nodo registrado.
    Lanza HTTP 401 si no matchea. Se llama al principio de cada endpoint
    que un AGENTE usa (heartbeat, pull, result) para evitar que cualquiera
    con solo el node_id (que es publico, viaja en /nodes) pueda actuar en
    nombre de ese nodo sin conocer el token secreto.

    Devuelve la fila del nodo por si el caller la necesita.
    """
    node = conn.execute(
        "SELECT * FROM nodes WHERE node_id=? AND token=?",
        (node_id, token)
    ).fetchone()
    if not node:
        raise HTTPException(401, "Invalid node credentials")
    return node


@app.get("/health")
def health():
    """Chequeo simple de vida. Util para probar conectividad (ej: a traves
    de un tunel) antes de intentar nada mas complejo."""
    return {"status":"ok","service":"sauzal-control-plane","version":"0.2.0"}


# Bloque fijo de 2MB, generado una sola vez al importar el modulo (no en
# cada request). Datos aleatorios (no ceros) para que no lo comprima de
# arriba ningun proxy/middleware y la medicion de velocidad sea honesta.
_BENCH_PAYLOAD = os.urandom(2 * 1024 * 1024)

@app.get("/bench/payload")
def bench_payload():
    """
    Devuelve ese bloque de 2MB sin ningun otro proposito que servir de
    "peso conocido": el agente lo descarga una vez al registrarse y
    calcula su velocidad de red cronometrando la descarga (bytes / tiempo).
    Se mide contra ESTE servidor a proposito -- lo relevante para la red
    Sauzal es que tan rapido le llega el trabajo/resultado al control
    plane, no un benchmark generico de banda ancha contra un tercero.
    """
    return Response(content=_BENCH_PAYLOAD, media_type="application/octet-stream")


@app.post("/nodes/register")
def register(req: Register, request: Request):
    """
    Da de alta un nodo nuevo. Genera un node_id y un token aleatorios
    (UUID4) y los guarda junto con el nombre y las capacidades declaradas.
    El agente llama esto una sola vez (o cuando borra su agent_config.json)
    y persiste la respuesta para no tener que registrarse de nuevo en cada
    arranque.

    `paused` es la preferencia de disponibilidad que declara el AGENTE al
    registrarse (flag --paused). Es solo un valor inicial: una vez creado
    el nodo, el panel /admin puede pausarlo/reactivarlo en cualquier
    momento sin que los heartbeats posteriores lo pisen (ver heartbeat()).
    `price_kwh`, `energy_wh` y `latency_ms` quedan en sus valores por
    defecto (NULL, 0, NULL) hasta que se configuren/midan despues.

    `ip_address`/`location` se calculan aca mismo, del lado del servidor
    (ver `client_ip()` y `geolocate()`) -- el agente no manda ni puede
    falsificar estos dos campos.
    """
    node_id, token = str(uuid.uuid4()), str(uuid.uuid4())
    ip = client_ip(request)
    location = geolocate(ip)
    with db() as conn:
        conn.execute(
            "INSERT INTO nodes(node_id,name,token,status,last_seen,capabilities,paused,ip_address,location) VALUES(?,?,?,?,?,?,?,?,?)",
            (node_id, req.name, token, "available", time.time(), req.capabilities, int(req.paused), ip, location)
        )
    return {"node_id":node_id,"token":token}


@app.post("/nodes/heartbeat")
def heartbeat(req: Heartbeat, request: Request):
    """
    El agente llama esto en bucle (cada ~2s) para:
      1. Probar que sigue autenticado (si el token no matchea, 401).
      2. Actualizar last_seen (asi /nodes lo sigue mostrando "online").
      3. Opcionalmente refrescar sus capabilities (por si cambiaron los
         modelos disponibles en Ollama/ComfyUI desde el ultimo heartbeat).
      4. Opcionalmente reportar latencia (ida y vuelta del heartbeat
         anterior) y un incremento de energia consumida (Wh), que se van
         acumulando en `energy_wh`.
      5. Refrescar ip_address/location (por si el nodo cambio de red; en
         la practica `geolocate()` esta cacheada por IP asi que esto no
         implica golpear la API externa en cada heartbeat).

    El UPDATE se arma dinamicamente para no pisar con NULL los campos que
    el agente no mando en esta llamada puntual. A proposito NUNCA se toca
    `paused` aca: esa columna solo la cambian el registro inicial del
    agente o una accion manual en /admin, para que la pausa "pegue" sin
    importar cuantos heartbeats lleguen despues.
    """
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        ip = client_ip(request)
        sets = ["status=?", "last_seen=?", "ip_address=?", "location=?"]
        params = [req.status, time.time(), ip, geolocate(ip)]
        if req.capabilities is not None:
            sets.append("capabilities=?")
            params.append(req.capabilities)
        if req.latency_ms is not None:
            sets.append("latency_ms=?")
            params.append(req.latency_ms)
        if req.energy_wh_delta is not None:
            sets.append("energy_wh=COALESCE(energy_wh,0)+?")
            params.append(req.energy_wh_delta)
        params.append(req.node_id)
        conn.execute(f"UPDATE nodes SET {','.join(sets)} WHERE node_id=?", params)
    return {"ok":True}


@app.get("/nodes")
def list_nodes():
    """
    Lista todos los nodos que se registraron alguna vez (incluidos los
    que ya no estan online). El campo calculado `online` es `True` si
    tuvieron un heartbeat en los ultimos 30 segundos — ese es el mismo
    umbral que usa /infer para decidir si un nodo esta disponible.

    No hay limpieza automatica: los nodos viejos quedan en la tabla para
    siempre (posible mejora futura: borrar los que llevan mucho offline).
    """
    now=time.time()
    with db() as conn:
        rows=conn.execute(
            "SELECT node_id,name,status,last_seen,capabilities,ip_address,location FROM nodes ORDER BY last_seen DESC"
        ).fetchall()
    return [{**dict(r),"online":now-r["last_seen"]<30} for r in rows]


@app.post("/infer")
def infer(req: Infer, request: Request):
    """
    Punto de entrada para pedir una inferencia. No requiere autenticacion
    (ver limitacion en el docstring del modulo).

    Datos del cliente que se guardan junto con el job (ver tabla `jobs`
    en el docstring de startup() para el detalle de cada columna):
      - `client_ip`/`client_location`: calculados por el SERVIDOR (no los
        puede falsificar el cliente, salvo spoofeando su propia IP de
        origen o los headers de proxy que confia client_ip()).
      - `client_user_agent`: header User-Agent tal cual lo manda el
        cliente HTTP -- revela sin que nadie lo declare si vino de un
        navegador o de un script (ej: "python-requests/2.32.4").
      - `client_os`/`client_processor`: SI son self-reportados por el
        cliente (`client/infer.py` los arma con el modulo `platform`) --
        no autenticados, igual criterio que el campo `client`.

    Eleccion de nodo:
      - Se excluyen siempre los nodos con `paused=1` (pausados desde el
        agente al registrarse, o manualmente desde /admin), aunque esten
        online y con status='available'.
      - Si el cliente mando `node` (nombre o node_id), se busca ESE nodo
        puntual, y debe estar disponible, no pausado, y con heartbeat
        reciente (<30s), sino 503.
      - Si no mando `node`, se toma el nodo disponible (no pausado) visto
        mas recientemente (heuristica simple, no es un balanceador real:
        no pesa carga ni afinidad de modelo/GPU).

    Al asignar el trabajo:
      1. Se inserta la fila en `jobs` con status="queued".
      2. Se marca el nodo como "busy" para que /infer no lo vuelva a
         asignar mientras trabaja (igual el agente hace overwrite de su
         propio status a "available" en cada heartbeat, asi que esto es
         mas una señal de "en este preciso instante tiene un pendiente"
         que un lock estricto).

    El nodo recien se entera del trabajo en su proximo POST /agent/pull.

    Memoria de conversacion (solo si viene `session_id`):
      1. Se valida que la sesion exista (404 si no).
      2. Si el modelo NO es de imagen, `memory.build_context()` arma el
         prompt final combinando resumen + memoria semantica + mensajes
         recientes de esa sesion + el mensaje nuevo -- ESE prompt
         combinado es el que se guarda en `jobs.prompt` y el que
         efectivamente recibe el nodo. Para modelos de imagen se omite
         (no tendria sentido inyectar una transcripcion de texto en un
         prompt de imagen); el mensaje se guarda igual, para que el
         historial de la sesion quede completo.
      3. Recien DESPUES de armar el contexto se guarda el mensaje del
         usuario (ver el aviso de orden en memory.build_context()).
      4. Las metricas de contexto usadas quedan en la fila del job.

    El nodo que ejecuta esto NUNCA se entera de que existe una sesion:
    solo ve un prompt de texto, como cualquier otro job.
    """
    now=time.time()
    with db() as conn:
        node = _pick_available_node(conn, req.node)
        if not node:
            raise HTTPException(503,"No available Sauzal nodes")

        final_prompt = req.prompt
        ctx_tokens=ctx_messages=ctx_hits=None
        if req.session_id:
            session = memory.get_session(conn, req.session_id)
            if not session:
                raise HTTPException(404, "Session not found")
            if not _is_image_model(req.model):
                final_prompt, metrics = memory.build_context(conn, req.session_id, req.prompt)
                ctx_tokens = metrics["context_tokens_estimate"]
                ctx_messages = metrics["context_messages_used"]
                ctx_hits = metrics["context_semantic_hits"]
            memory.add_message(conn, req.session_id, "user", req.prompt)

        job_id=str(uuid.uuid4())
        ip = client_ip(request)
        conn.execute("""
          INSERT INTO jobs(
            job_id,model,prompt,status,assigned_node,client,
            client_ip,client_location,client_user_agent,client_os,client_processor,
            session_id,kind,context_tokens_estimate,context_messages_used,context_semantic_hits,
            created_at,updated_at
          ) VALUES(?,?,?,'queued',?,?,?,?,?,?,?,?,'chat',?,?,?,?,?)
        """,(
            job_id,req.model,final_prompt,node["node_id"],req.client,
            ip,geolocate(ip),request.headers.get("user-agent"),req.client_os,req.client_processor,
            req.session_id,ctx_tokens,ctx_messages,ctx_hits,
            now,now
        ))
        conn.execute("UPDATE nodes SET status='busy' WHERE node_id=?",(node["node_id"],))
    return {"job_id":job_id,"assigned_node":node["node_id"],"status":"queued"}


@app.post("/agent/pull")
def pull(req: Auth):
    """
    El agente llama esto en cada ciclo de heartbeat para preguntar si
    tiene trabajo asignado. Busca el job "queued" mas antiguo asignado a
    ESE node_id (podria haber mas de uno en cola si se le mandaron varios
    seguidos), lo marca "running" y se lo devuelve.

    Si no hay nada, devuelve {"job": None} y el agente simplemente espera
    al proximo ciclo (ver el `time.sleep(2)` en agent.py).
    """
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        job=conn.execute("""
          SELECT job_id,model,prompt FROM jobs
          WHERE assigned_node=? AND status='queued'
          ORDER BY created_at LIMIT 1
        """,(req.node_id,)).fetchone()
        if not job:
            return {"job":None}
        conn.execute(
            "UPDATE jobs SET status='running',updated_at=? WHERE job_id=?",
            (time.time(),job["job_id"])
        )
    return {"job":dict(job)}


def _maybe_enqueue_summary(conn, session_id: str, model: str) -> None:
    """
    Si la sesion acumulo demasiados mensajes viejos sin resumir (ver
    `memory.SUMMARIZE_TRIGGER`), encola un job INTERNO `kind='summarize'`
    para comprimirlos, usando el mismo modelo que se viene usando en la
    conversacion y el mismo mecanismo de seleccion de nodo que un pedido
    normal (`_pick_available_node()`).

    Si no hay ningun nodo disponible en este momento, no hace nada -- se
    vuelve a evaluar la proxima vez que se guarde una respuesta de esa
    sesion (no es critico que el resumen se genere de inmediato).

    Tambien se abstiene si YA hay un resumen en curso (queued o running)
    para esta sesion: sin este chequeo, una conversacion muy activa
    podria disparar un job de resumen nuevo en cada turno mientras el
    anterior todavia no termino, porque `pending` sigue creciendo hasta
    que el resumen en curso efectivamente se complete y marque esos
    mensajes como `summarized=1`.
    """
    pending = memory.messages_needing_summary(conn, session_id)
    if len(pending) < memory.SUMMARIZE_TRIGGER:
        return
    already_running = conn.execute("""
      SELECT 1 FROM jobs WHERE session_id=? AND kind='summarize'
        AND status IN ('queued','running') LIMIT 1
    """, (session_id,)).fetchone()
    if already_running:
        return
    node = _pick_available_node(conn)
    if not node:
        return
    session = memory.get_session(conn, session_id)
    summary_prompt = memory.build_summary_prompt(session, pending)
    now = time.time()
    job_id = str(uuid.uuid4())
    conn.execute("""
      INSERT INTO jobs(job_id,model,prompt,status,assigned_node,session_id,kind,summarize_message_ids,created_at,updated_at)
      VALUES(?,?,?,'queued',?,?,'summarize',?,?,?)
    """, (
        job_id, model, summary_prompt, node["node_id"], session_id,
        json.dumps([m["message_id"] for m in pending]), now, now,
    ))
    conn.execute("UPDATE nodes SET status='busy' WHERE node_id=?", (node["node_id"],))


@app.post("/jobs/{job_id}/result")
def result(job_id: str, req: Result):
    """
    El agente llama esto cuando termina de ejecutar (bien o mal) el
    trabajo que le tocaba. Guarda el resultado/error y status final, y
    libera al nodo (status='available') para que /infer pueda volver a
    asignarle trabajos.

    Si el job exitoso pertenecia a una sesion:
      - `kind='chat'`: la respuesta del asistente se guarda como mensaje
        nuevo de esa sesion (memoria de corto plazo para el proximo
        turno), y se evalua si hace falta disparar un resumen automatico
        (`_maybe_enqueue_summary()`).
      - `kind='summarize'`: el resultado ES el resumen -- se guarda en
        `sessions.summary` (reemplazando el anterior, que ya estaba
        incorporado en el prompt que genero este resumen nuevo) y se
        marcan `summarized=1` los mensajes que cubria
        (`jobs.summarize_message_ids`).

    No valida que el job_id realmente estuviera asignado a este node_id
    (posible mejora: chequear `assigned_node=?` en el UPDATE).
    """
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        job = conn.execute(
            "SELECT model,session_id,kind,summarize_message_ids FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        status="completed" if req.success else "failed"
        conn.execute("""
          UPDATE jobs SET status=?,result=?,error=?,prompt_tokens=?,output_tokens=?,duration_ms=?,updated_at=? WHERE job_id=?
        """,(status,req.result,req.error,req.prompt_tokens,req.output_tokens,req.duration_ms,time.time(),job_id))
        conn.execute(
            "UPDATE nodes SET status='available',last_seen=? WHERE node_id=?",
            (time.time(),req.node_id)
        )

        if req.success and job and job["session_id"]:
            try:
                response_text = json.loads(req.result).get("response") if req.result else None
            except json.JSONDecodeError:
                response_text = None
            if response_text:
                if job["kind"] == "summarize":
                    conn.execute(
                        "UPDATE sessions SET summary=? WHERE session_id=?",
                        (response_text, job["session_id"]),
                    )
                    ids = json.loads(job["summarize_message_ids"] or "[]")
                    if ids:
                        placeholders = ",".join("?" for _ in ids)
                        conn.execute(
                            f"UPDATE messages SET summarized=1 WHERE message_id IN ({placeholders})", ids
                        )
                else:
                    memory.add_message(conn, job["session_id"], "assistant", response_text)
                    _maybe_enqueue_summary(conn, job["session_id"], job["model"])
    return {"ok":True}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """
    El cliente hace polling a este endpoint hasta que status sea
    "completed" o "failed". Se incluye `node_name` (via LEFT JOIN con
    `nodes`) ademas de `assigned_node` (el UUID), para que el cliente
    pueda mostrar un nombre legible de que maquina ejecuto el trabajo en
    vez de solo el ID.
    """
    with db() as conn:
        job=conn.execute("""
          SELECT jobs.*, nodes.name AS node_name FROM jobs
          LEFT JOIN nodes ON nodes.node_id = jobs.assigned_node
          WHERE jobs.job_id=?
        """,(job_id,)).fetchone()
    if not job:
        raise HTTPException(404,"Job not found")
    return dict(job)


# ============================================================
# Sesiones de conversacion (memoria)
# ============================================================
#
# Estos cuatro endpoints son la interfaz publica de la memoria de
# conversacion (ver server/memory.py para la logica de recuperacion de
# contexto). El uso tipico es:
#
#   1. POST /sessions                      -> {"session_id": "..."}
#   2. POST /infer  {"session_id": "...", "prompt": "..."}   (repetido,
#      potencialmente contra nodos distintos cada vez -- ver la logica de
#      contexto en el propio /infer, mas abajo)
#   3. DELETE /sessions/{id}               (cuando termina la conversacion)
#
# Una request a /infer SIN session_id sigue funcionando exactamente igual
# que antes de que existiera este modulo (inferencia suelta, sin memoria).

class SessionCreate(BaseModel):
    """Body de POST /sessions. Todo opcional: alcanza con POST /sessions
    con body vacio para el caso comun."""
    tenant_id: str | None = None  # reservado para multi-tenant futuro, no se usa todavia


@app.post("/sessions")
def create_session(req: SessionCreate = SessionCreate()):
    """Crea una sesion de conversacion nueva y devuelve su session_id."""
    with db() as conn:
        session_id = memory.create_session(conn, req.tenant_id)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    """
    Info resumida de una sesion: cuando se creo, cuando se uso por
    ultima vez, cuantos mensajes tiene, y su resumen acumulado (si ya se
    genero uno). Util para debugging/monitoreo, no hace falta para el
    flujo normal de un cliente.
    """
    with db() as conn:
        session = memory.get_session(conn, session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()["n"]
    return {**dict(session), "message_count": count}


@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Historial completo de mensajes de una sesion, en orden
    cronologico. Incluye los mensajes triviales y ya resumidos -- esto
    devuelve el registro crudo completo, no lo que efectivamente se le
    manda al modelo en cada inferencia (eso lo arma memory.build_context())."""
    with db() as conn:
        if not memory.get_session(conn, session_id):
            raise HTTPException(404, "Session not found")
        messages = [dict(r) for r in memory.list_messages(conn, session_id)]
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Borra la sesion Y todos sus mensajes -- aislamiento y borrado
    completo, sin dejar rastro de memoria asociada a esa conversacion."""
    with db() as conn:
        if not memory.get_session(conn, session_id):
            raise HTTPException(404, "Session not found")
        memory.delete_session(conn, session_id)
    return {"ok": True}


# ============================================================
# Panel de administracion
# ============================================================
#
# Pagina web (HTML servido directamente por FastAPI, sin frontend ni
# libreria de templates aparte) para ver de un vistazo el estado
# completo de la base de datos -- todos los nodos y todos los trabajos
# -- y hacer los cambios manuales mas comunes sin necesitar un visor de
# SQLite externo: eliminar un nodo, forzarlo de vuelta a "available" si
# quedo colgado en "busy", o eliminar un trabajo.
#
# Sin autenticacion, con el mismo criterio que el resto del servidor
# (ver limitaciones en el docstring del modulo). Pensado para uso en LAN
# o en tu propia PC durante desarrollo -- si el servidor se llega a
# exponer a internet, esta pagina permite borrar datos a cualquiera que
# la encuentre, asi que convendria ponerle autenticacion antes.
#
# La pagina se autorefresca cada 5 segundos (via <meta http-equiv=
# "refresh">, sin JavaScript) para funcionar como un mini-dashboard de
# monitoreo en vivo de la red.

def _ago(ts: float, now: float) -> str:
    """
    Convierte un timestamp epoch en un texto relativo corto ("5s", "3m",
    "2h", "4d") para mostrar en el panel sin depender de ninguna libreria
    de fechas externa.
    """
    delta = max(0, now - ts)
    if delta < 60:
        return f"{int(delta)}s"
    if delta < 3600:
        return f"{int(delta/60)}m"
    if delta < 86400:
        return f"{int(delta/3600)}h"
    return f"{int(delta/86400)}d"


def _fmt(value, suffix: str = "") -> str:
    """Formatea un valor que puede ser None (dato no disponible en este
    nodo, ej: temperatura en una GPU no-NVIDIA) como "N/A" en vez de
    mostrar "None" crudo en el panel."""
    return "N/A" if value is None else f"{value}{suffix}"


def _client_origin(user_agent: str | None) -> str:
    """
    Heuristica simple para distinguir si un pedido a /infer vino de un
    navegador o de un script/API, a partir del header User-Agent (que
    cualquier cliente HTTP manda automaticamente, sin que nadie lo
    declare a mano). Los navegadores incluyen "Mozilla/5.0" en su User-
    Agent por convencion historica; `requests` (lo que usa client/infer.py)
    manda algo como "python-requests/2.32.4", `curl` manda "curl/8.x", etc.
    No es infalible (un script puede mandar un User-Agent de navegador a
    proposito), es solo una senal informativa mas.
    """
    if not user_agent:
        return "desconocido"
    ua = user_agent.lower()
    if any(x in ua for x in ("mozilla", "chrome", "safari", "firefox", "edge/", "webkit")):
        return "🌐 navegador"
    return "🖥️ script/API"


def _node_row(row: dict, now: float) -> str:
    """
    Arma la fila <tr> compacta de un nodo para la tabla principal del
    panel (nombre, estado, disponibilidad, backends). Todas las metricas
    extendidas (VRAM, RAM, driver, benchmark, temperatura, consumo,
    energia/costo, historial de fallas) viven en la pagina de detalle
    (`/admin/nodes/{id}`, ver admin_node_detail) para no volver ilegible
    esta tabla con 15 columnas.

    Todo texto que viene de datos guardados por agentes/clientes (nombre)
    pasa por `html.escape` antes de insertarse en el HTML, porque esos
    valores no son de confianza (el nombre de un nodo lo elige quien
    arranca el agente con --name, y /infer no tiene autenticacion) -- sin
    este escape, alguien podria registrar un nodo con un nombre tipo
    "<script>...</script>" y ejecutar JavaScript en el navegador de quien
    abra este panel.
    """
    try:
        caps = json.loads(row["capabilities"] or "{}")
    except json.JSONDecodeError:
        caps = {}
    online = now - row["last_seen"] < 30
    services = caps.get("services", {}) if isinstance(caps, dict) else {}
    backends = ", ".join(k for k, v in services.items() if v) or "-"
    badge = "🟢 online" if online else "⚪ offline"
    name = html.escape(row["name"] or "")
    paused = bool(row.get("paused"))
    disponibilidad = "⏸️ Pausado" if paused else "✅ Activo"
    pause_action, pause_label = ("resume", "Reactivar") if paused else ("pause", "Pausar")
    ip_label = html.escape(row.get("ip_address") or "N/A")
    location = row.get("location")
    ip_display = f"{ip_label} ({html.escape(location)})" if location else ip_label
    return f"""
    <tr>
      <td><a href="/admin/nodes/{row['node_id']}">{name}</a></td>
      <td>{badge}</td>
      <td>{disponibilidad}</td>
      <td>{html.escape(row["status"] or "")}</td>
      <td>{ip_display}</td>
      <td>{_ago(row["last_seen"], now)}</td>
      <td>{html.escape(backends)}</td>
      <td class="actions">
        <form method="post" action="/admin/nodes/{row['node_id']}/{pause_action}">
          <button>{pause_label}</button>
        </form>
        <form method="post" action="/admin/nodes/{row['node_id']}/unstick">
          <button title="Forzar status='available' (util si quedo colgado en busy)">Forzar disponible</button>
        </form>
        <form method="post" action="/admin/nodes/{row['node_id']}/delete" onsubmit="return confirm('¿Eliminar este nodo?')">
          <button class="danger">Eliminar</button>
        </form>
      </td>
    </tr>"""


def _job_row(row: dict, now: float) -> str:
    """
    Arma la fila <tr> de un trabajo para la tabla del panel. El prompt se
    trunca a 60 caracteres para no romper el layout, pero el texto
    completo queda disponible al pasar el mouse (atributo `title`). El
    link del Job ID abre GET /jobs/{id} (el endpoint ya existente para
    clientes) en una pestaña nueva, para ver el resultado/error completo
    en crudo sin duplicar esa logica aca.
    """
    prompt = row["prompt"] or ""
    prompt_short = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    node_label = row["node_name"] or row["assigned_node"] or "-"
    client_name = row.get("client")
    client = (
        f'<a href="/admin/clients/{urllib.parse.quote(client_name)}">{html.escape(client_name)}</a>'
        if client_name else "-"
    )
    # Los tokens solo existen para jobs de texto (Ollama los reporta);
    # los de imagen (ComfyUI) quedan en NULL, se muestra "-" en ese caso.
    pt, ot = row.get("prompt_tokens"), row.get("output_tokens")
    tokens = f"{pt if pt is not None else '?'}/{ot if ot is not None else '?'}" if (pt is not None or ot is not None) else "-"
    status_icon = {
        "queued": "⏳",
        "running": "⚙️",
        "completed": "✅",
        "failed": "❌",
    }.get(row["status"], "")
    # duration_ms lo mide el agente (exito o fallo); "-" mientras el job
    # sigue queued/running y todavia no hay un resultado que reportar.
    duration_ms = row.get("duration_ms")
    duration = f"{duration_ms/1000:.1f}s" if duration_ms and duration_ms >= 1000 else (
        f"{duration_ms:.0f}ms" if duration_ms is not None else "-"
    )
    return f"""
    <tr>
      <td><a href="/jobs/{row['job_id']}" target="_blank">{row['job_id'][:8]}…</a></td>
      <td>{html.escape(row["model"] or "")}</td>
      <td title="{html.escape(prompt)}">{html.escape(prompt_short)}</td>
      <td>{client}</td>
      <td>{status_icon} {html.escape(row["status"] or "")}</td>
      <td>{html.escape(node_label)}</td>
      <td title="tokens de entrada/salida">{tokens}</td>
      <td>{duration}</td>
      <td>{_ago(row["created_at"], now)}</td>
      <td class="actions">
        <form method="post" action="/admin/jobs/{row['job_id']}/delete" onsubmit="return confirm('¿Eliminar este trabajo?')">
          <button class="danger">Eliminar</button>
        </form>
      </td>
    </tr>"""


def _client_row(c: dict, now: float) -> str:
    """
    Arma la fila <tr> de un cliente para la tabla principal del panel.

    A diferencia de los nodos, un "cliente" no se registra ni manda
    heartbeat: esta fila se arma agregando la tabla `jobs` por el campo
    `client` (ver admin_panel()). Los datos de IP/ubicacion/SO/user-agent
    que se muestran son los del job MAS RECIENTE de ese nombre de
    cliente -- una "foto" del ultimo pedido, no un estado "en vivo" como
    el online/offline de los nodos.
    """
    name = c["client"]
    ip = html.escape(c.get("client_ip") or "N/A")
    location = c.get("client_location")
    ip_display = f"{ip} ({html.escape(location)})" if location else ip
    return f"""
    <tr>
      <td><a href="/admin/clients/{urllib.parse.quote(name)}">{html.escape(name)}</a></td>
      <td>{ip_display}</td>
      <td>{_client_origin(c.get("client_user_agent"))}</td>
      <td>{html.escape(c.get("client_os") or "N/A")}</td>
      <td>{c["job_count"]}</td>
      <td>{_ago(c["created_at"], now)}</td>
    </tr>"""


def _session_row(s: dict, now: float) -> str:
    """
    Fila <tr> de una sesion de conversacion para la tabla del panel.
    `message_count` viene de un COUNT(*) separado (ver admin_panel()),
    no de `sessions` -- esa tabla no guarda un contador propio.
    """
    summary = s.get("summary")
    summary_preview = (summary[:60] + "…") if summary and len(summary) > 60 else (summary or "-")
    return f"""
    <tr>
      <td><a href="/admin/sessions/{s['session_id']}">{s['session_id'][:8]}…</a></td>
      <td>{s["message_count"]}</td>
      <td title="{html.escape(summary or '')}">{html.escape(summary_preview)}</td>
      <td>{_ago(s["created_at"], now)}</td>
      <td>{_ago(s["last_used_at"], now)}</td>
      <td class="actions">
        <form method="post" action="/admin/sessions/{s['session_id']}/delete" onsubmit="return confirm('¿Borrar esta sesion y toda su memoria?')">
          <button class="danger">Eliminar</button>
        </form>
      </td>
    </tr>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    """
    Pagina principal del panel: trae TODOS los nodos, todos los clientes
    distintos (agregados de `jobs`), y los ultimos 200 trabajos (mas
    nuevos primero), y arma una unica pagina HTML con tres tablas. No
    pagina nodos/clientes (en la practica nunca son tantos como para
    necesitarlo); los jobs si se limitan a 200 para que la tabla no
    crezca sin limite en una red con mucho trafico.
    """
    now = time.time()
    with db() as conn:
        nodes = [dict(r) for r in conn.execute(
            "SELECT * FROM nodes ORDER BY last_seen DESC"
        ).fetchall()]
        jobs = [dict(r) for r in conn.execute("""
          SELECT jobs.*, nodes.name AS node_name FROM jobs
          LEFT JOIN nodes ON nodes.node_id = jobs.assigned_node
          ORDER BY jobs.created_at DESC LIMIT 200
        """).fetchall()]
        client_job_rows = [dict(r) for r in conn.execute("""
          SELECT client, client_ip, client_location, client_os, client_processor,
                 client_user_agent, created_at
          FROM jobs WHERE client IS NOT NULL ORDER BY created_at DESC
        """).fetchall()]
        sessions = [dict(r) for r in conn.execute("""
          SELECT sessions.*, COUNT(messages.message_id) AS message_count
          FROM sessions LEFT JOIN messages ON messages.session_id = sessions.session_id
          GROUP BY sessions.session_id ORDER BY sessions.last_used_at DESC
        """).fetchall()]

    # Agrupa por nombre de cliente en Python (no en SQL) para quedarse con
    # los datos del job MAS RECIENTE de cada uno (la lista ya viene
    # ordenada DESC, asi que el primero que aparece es el mas nuevo) mas
    # un conteo total de trabajos.
    clients_by_name: dict[str, dict] = {}
    for r in client_job_rows:
        entry = clients_by_name.setdefault(r["client"], {**r, "job_count": 0})
        entry["job_count"] += 1
    clients = list(clients_by_name.values())

    nodes_html = "".join(_node_row(n, now) for n in nodes) or \
        "<tr><td colspan='8'>No hay nodos registrados todavia.</td></tr>"
    clients_html = "".join(_client_row(c, now) for c in clients) or \
        "<tr><td colspan='6'>Todavia no hay pedidos de ningun cliente.</td></tr>"
    sessions_html = "".join(_session_row(s, now) for s in sessions) or \
        "<tr><td colspan='6'>Todavia no hay sesiones de conversacion.</td></tr>"
    jobs_html = "".join(_job_row(j, now) for j in jobs) or \
        "<tr><td colspan='10'>No hay trabajos todavia.</td></tr>"
    online_count = sum(1 for n in nodes if now - n["last_seen"] < 30)

    return HTMLResponse(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Sauzal · Admin</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e6e6e6; margin:2rem; }}
  h1 {{ font-size:1.4rem; margin-bottom:0.2rem; }}
  h2 {{ font-size:1.1rem; margin-top:2rem; color:#9aa0a6; }}
  table {{ width:100%; border-collapse: collapse; margin-top:0.5rem; }}
  th, td {{ text-align:left; padding:0.5rem 0.7rem; border-bottom:1px solid #2a2d34; font-size:0.9rem; vertical-align:top; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  tr:hover {{ background:#171a20; }}
  .actions {{ display:flex; gap:0.4rem; white-space:nowrap; }}
  .actions form {{ margin:0; }}
  button {{ background:#2a2d34; color:#e6e6e6; border:1px solid #3a3d44; border-radius:4px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.8rem; }}
  button:hover {{ background:#3a3d44; }}
  button.danger:hover {{ background:#5c2020; border-color:#7a2a2a; }}
  a {{ color:#7fb0ff; }}
  .summary {{ color:#9aa0a6; font-size:0.9rem; }}
</style>
</head>
<body>
  <h1>Sauzal · Panel de administracion</h1>
  <p class="summary">{len(nodes)} nodos registrados ({online_count} online) · {len(clients)} clientes distintos · {len(sessions)} sesiones de conversacion · {len(jobs)} trabajos mostrados (ultimos 200) · se actualiza sola cada 5s · click en un nombre para ver mas detalle</p>

  <h2>Nodos</h2>
  <table>
    <tr><th>Nombre</th><th>Online</th><th>Disponibilidad</th><th>Status</th><th>IP (ubicacion)</th><th>Ultimo heartbeat</th><th>Backends</th><th>Acciones</th></tr>
    {nodes_html}
  </table>

  <h2>Clientes</h2>
  <table>
    <tr><th>Nombre</th><th>IP (ubicacion)</th><th>Origen</th><th>SO</th><th>Trabajos</th><th>Ultimo pedido hace</th></tr>
    {clients_html}
  </table>

  <h2>Sesiones de conversacion</h2>
  <table>
    <tr><th>Session</th><th>Mensajes</th><th>Resumen</th><th>Creada hace</th><th>Ultimo uso hace</th><th>Acciones</th></tr>
    {sessions_html}
  </table>

  <h2>Trabajos</h2>
  <table>
    <tr><th>Job</th><th>Modelo</th><th>Prompt</th><th>Cliente</th><th>Estado</th><th>Nodo</th><th>Tokens (in/out)</th><th>Duracion</th><th>Creado hace</th><th>Acciones</th></tr>
    {jobs_html}
  </table>
</body>
</html>""")


@app.get("/admin/nodes/{node_id}", response_class=HTMLResponse)
def admin_node_detail(node_id: str):
    """
    Pagina de detalle de UN nodo puntual: todas las metricas extendidas
    que no entran comodas en la tabla principal de /admin -- hardware
    (VRAM, RAM, driver, motor de computo), versiones de software,
    benchmark medido al registrarse, latencia/velocidad de red,
    temperatura/consumo en vivo (si el nodo es NVIDIA), energia
    acumulada + costo estimado (configurando un precio de $/kWh), el
    toggle de disponibilidad, y el historial de trabajos fallidos de
    este nodo puntual.

    Todos los campos de hardware/software que un nodo no pueda reportar
    (por ejemplo, temperatura en una GPU no-NVIDIA) se muestran como
    "N/A" en vez de inventar un valor o romper la pagina (ver `_fmt`).
    """
    now = time.time()
    with db() as conn:
        node = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        if not node:
            raise HTTPException(404, "Nodo no encontrado")
        node = dict(node)
        failed_jobs = [dict(r) for r in conn.execute("""
          SELECT job_id, model, error, created_at FROM jobs
          WHERE assigned_node=? AND status='failed'
          ORDER BY created_at DESC LIMIT 20
        """, (node_id,)).fetchall()]
        fail_count = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE assigned_node=? AND status='failed'", (node_id,)
        ).fetchone()["n"]
        total_jobs = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE assigned_node=?", (node_id,)
        ).fetchone()["n"]

    try:
        caps = json.loads(node["capabilities"] or "{}")
        if not isinstance(caps, dict):
            caps = {}
    except json.JSONDecodeError:
        caps = {}

    online = now - node["last_seen"] < 30
    paused = bool(node["paused"])
    services = caps.get("services", {}) if isinstance(caps.get("services"), dict) else {}
    backends = ", ".join(k for k, v in services.items() if v) or "-"

    energy_wh = node["energy_wh"] or 0
    price_kwh = node["price_kwh"]
    cost = (energy_wh/1000)*price_kwh if price_kwh is not None else None

    benchmark = caps.get("benchmark") if isinstance(caps.get("benchmark"), dict) else {}
    benchmark_html = "".join(
        f"<div><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</div>"
        for k, v in benchmark.items()
    ) or "<div>Sin datos de benchmark.</div>"

    failed_rows = "".join(f"""
      <tr>
        <td><a href="/jobs/{j['job_id']}" target="_blank">{j['job_id'][:8]}…</a></td>
        <td>{html.escape(j['model'] or '')}</td>
        <td title="{html.escape(j['error'] or '')}">{html.escape((j['error'] or '')[:80])}</td>
        <td>{_ago(j['created_at'], now)}</td>
      </tr>""" for j in failed_jobs) or "<tr><td colspan='4'>Sin fallas registradas.</td></tr>"

    return HTMLResponse(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>Sauzal · {html.escape(node["name"] or node_id)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e6e6e6; margin:2rem; }}
  h1 {{ font-size:1.4rem; margin-bottom:0.2rem; }}
  h2 {{ font-size:1.1rem; margin-top:2rem; color:#9aa0a6; }}
  table {{ width:100%; border-collapse: collapse; margin-top:0.5rem; }}
  th, td {{ text-align:left; padding:0.5rem 0.7rem; border-bottom:1px solid #2a2d34; font-size:0.9rem; vertical-align:top; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  a {{ color:#7fb0ff; }}
  .summary {{ color:#9aa0a6; font-size:0.9rem; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:1rem; margin-top:1rem; }}
  .card {{ background:#171a20; border:1px solid #2a2d34; border-radius:8px; padding:1rem; }}
  .card h3 {{ margin:0 0 0.6rem 0; font-size:0.78rem; color:#9aa0a6; text-transform:uppercase; letter-spacing:0.05em; }}
  .card div {{ font-size:0.95rem; margin-bottom:0.25rem; }}
  .card form {{ margin-top:0.6rem; display:flex; gap:0.4rem; }}
  button {{ background:#2a2d34; color:#e6e6e6; border:1px solid #3a3d44; border-radius:4px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.8rem; }}
  button:hover {{ background:#3a3d44; }}
  button.danger:hover {{ background:#5c2020; border-color:#7a2a2a; }}
  input[type=number] {{ background:#0f1115; color:#e6e6e6; border:1px solid #3a3d44; border-radius:4px; padding:0.3rem 0.5rem; width:110px; }}
</style>
</head>
<body>
  <p><a href="/admin">← Volver al panel</a></p>
  <h1>{html.escape(node["name"] or "")} {"🟢" if online else "⚪"}</h1>
  <p class="summary">node_id: {node_id} · {"⏸️ Pausado" if paused else "✅ Activo"} · visto hace {_ago(node["last_seen"], now)} · {total_jobs} trabajos totales, {fail_count} fallidos</p>

  <div class="grid">
    <div class="card">
      <h3>Disponibilidad</h3>
      <div>{"Pausado manualmente: no recibe trabajos nuevos." if paused else "Disponible: puede recibir trabajos."}</div>
      <form method="post" action="/admin/nodes/{node_id}/{'resume' if paused else 'pause'}">
        <button>{"Reactivar" if paused else "Pausar"}</button>
      </form>
    </div>

    <div class="card">
      <h3>Hardware</h3>
      <div>GPU: {html.escape(", ".join(caps.get("gpu") or []) or "N/A")}</div>
      <div>VRAM: {_fmt(caps.get("vram_used_mb"))} / {_fmt(caps.get("vram_total_mb"))} MB</div>
      <div>RAM: {_fmt(caps.get("ram_used_mb"))} / {_fmt(caps.get("ram_total_mb"))} MB</div>
      <div>Driver: {html.escape(str(caps.get("driver_version") or "N/A"))}</div>
      <div>Motor de computo: {html.escape(str(caps.get("compute_backend") or "N/A"))}</div>
    </div>

    <div class="card">
      <h3>Software</h3>
      <div>SO: {html.escape(caps.get("os") or "N/A")}</div>
      <div>Ollama: {html.escape(str(caps.get("ollama_version") or "N/A"))}</div>
      <div>ComfyUI (Python): {html.escape(str(caps.get("comfy_python_version") or "N/A"))}</div>
      <div>Motores activos: {html.escape(backends)}</div>
      <div>Modelos texto: {html.escape(", ".join(caps.get("ollama_models") or []) or "-")}</div>
      <div>Modelos imagen: {html.escape(", ".join(caps.get("image_models") or []) or "-")}</div>
    </div>

    <div class="card">
      <h3>Benchmark (medido al registrarse)</h3>
      {benchmark_html}
    </div>

    <div class="card">
      <h3>Red</h3>
      <div>IP: {html.escape(node.get("ip_address") or "N/A")}</div>
      <div>Ubicacion: {html.escape(node.get("location") or "N/A (IP privada o sin resolver)")}</div>
      <div>Latencia heartbeat: {_fmt(node["latency_ms"], " ms")}</div>
      <div>Velocidad hacia el servidor: {_fmt(caps.get("network_mbps"), " Mbps")}</div>
    </div>

    <div class="card">
      <h3>Temperatura / Consumo (en vivo)</h3>
      <div>Temperatura: {_fmt(caps.get("gpu_temp_c"), " °C")}</div>
      <div>Consumo: {_fmt(caps.get("gpu_power_w"), " W")}</div>
      <div style="color:#9aa0a6; font-size:0.8rem; margin-top:0.4rem;">Solo disponible en GPUs NVIDIA (nvidia-smi).</div>
    </div>

    <div class="card">
      <h3>Energia y costo electrico</h3>
      <div>Energia acumulada: {round(energy_wh, 2)} Wh</div>
      <div>Precio configurado: {_fmt(price_kwh, "/kWh")}</div>
      <div>Costo estimado: {_fmt(round(cost, 4) if cost is not None else None)}</div>
      <form method="post" action="/admin/nodes/{node_id}/price">
        <input type="number" step="0.0001" name="price_kwh" placeholder="precio $/kWh"
          value="{price_kwh if price_kwh is not None else ''}">
        <button>Guardar precio</button>
      </form>
      <form method="post" action="/admin/nodes/{node_id}/reset-energy" onsubmit="return confirm('¿Reiniciar el contador de energia a 0?')">
        <button class="danger">Reiniciar contador</button>
      </form>
    </div>
  </div>

  <h2>Historial de fallas (ultimas 20)</h2>
  <table>
    <tr><th>Job</th><th>Modelo</th><th>Error</th><th>Hace</th></tr>
    {failed_rows}
  </table>
</body>
</html>""")


@app.get("/admin/clients/{client_name}", response_class=HTMLResponse)
def admin_client_detail(client_name: str):
    """
    Pagina de detalle de UN cliente puntual: IP, ubicacion, sistema
    operativo, procesador, y si el origen parece un navegador o un
    script -- todo tomado del job MAS RECIENTE de ese cliente -- mas su
    historial completo de trabajos.

    A diferencia de los nodos, los clientes NO se registran ni mandan
    heartbeat: no existe una tabla `clients` separada. Esta pagina entera
    se arma agregando la tabla `jobs` por el campo `client` (ver tambien
    `admin_panel()`, que arma la tabla resumen con el mismo criterio).
    Por eso un 404 aca significa "ese nombre nunca aparecio como client
    en ningun job", no "no existe tal registro".
    """
    now = time.time()
    with db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE client=?", (client_name,)
        ).fetchone()["n"]
        if total == 0:
            raise HTTPException(404, "Cliente no encontrado (sin trabajos registrados)")
        latest = dict(conn.execute(
            "SELECT * FROM jobs WHERE client=? ORDER BY created_at DESC LIMIT 1", (client_name,)
        ).fetchone())
        jobs = [dict(r) for r in conn.execute("""
          SELECT jobs.*, nodes.name AS node_name FROM jobs
          LEFT JOIN nodes ON nodes.node_id = jobs.assigned_node
          WHERE jobs.client=? ORDER BY jobs.created_at DESC LIMIT 100
        """, (client_name,)).fetchall()]

    jobs_html = "".join(_job_row(j, now) for j in jobs) or \
        "<tr><td colspan='10'>Sin trabajos.</td></tr>"
    user_agent = latest.get("client_user_agent") or ""

    return HTMLResponse(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>Sauzal · {html.escape(client_name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e6e6e6; margin:2rem; }}
  h1 {{ font-size:1.4rem; margin-bottom:0.2rem; }}
  h2 {{ font-size:1.1rem; margin-top:2rem; color:#9aa0a6; }}
  table {{ width:100%; border-collapse: collapse; margin-top:0.5rem; }}
  th, td {{ text-align:left; padding:0.5rem 0.7rem; border-bottom:1px solid #2a2d34; font-size:0.9rem; vertical-align:top; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  a {{ color:#7fb0ff; }}
  .summary {{ color:#9aa0a6; font-size:0.9rem; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:1rem; margin-top:1rem; }}
  .card {{ background:#171a20; border:1px solid #2a2d34; border-radius:8px; padding:1rem; }}
  .card h3 {{ margin:0 0 0.6rem 0; font-size:0.78rem; color:#9aa0a6; text-transform:uppercase; letter-spacing:0.05em; }}
  .card div {{ font-size:0.95rem; margin-bottom:0.25rem; }}
  .actions {{ display:flex; gap:0.4rem; white-space:nowrap; }}
  .actions form {{ margin:0; }}
  button {{ background:#2a2d34; color:#e6e6e6; border:1px solid #3a3d44; border-radius:4px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.8rem; }}
  button.danger:hover {{ background:#5c2020; border-color:#7a2a2a; }}
</style>
</head>
<body>
  <p><a href="/admin">← Volver al panel</a></p>
  <h1>{html.escape(client_name)}</h1>
  <p class="summary">{total} trabajos totales (mostrando los ultimos 100) · ultimo pedido hace {_ago(latest["created_at"], now)}</p>

  <div class="grid">
    <div class="card">
      <h3>Origen</h3>
      <div>IP: {html.escape(latest.get("client_ip") or "N/A")}</div>
      <div>Ubicacion: {html.escape(latest.get("client_location") or "N/A (IP privada o sin resolver)")}</div>
      <div>Tipo: {_client_origin(user_agent)}</div>
      <div title="{html.escape(user_agent)}">User-Agent: {html.escape(user_agent[:60] or "N/A")}</div>
    </div>
    <div class="card">
      <h3>Sistema (self-reportado)</h3>
      <div>SO: {html.escape(latest.get("client_os") or "N/A")}</div>
      <div>Procesador: {html.escape(latest.get("client_processor") or "N/A")}</div>
    </div>
  </div>

  <h2>Historial de trabajos (ultimos 100)</h2>
  <table>
    <tr><th>Job</th><th>Modelo</th><th>Prompt</th><th>Cliente</th><th>Estado</th><th>Nodo</th><th>Tokens (in/out)</th><th>Duracion</th><th>Creado hace</th><th>Acciones</th></tr>
    {jobs_html}
  </table>
</body>
</html>""")


@app.get("/admin/sessions/{session_id}", response_class=HTMLResponse)
def admin_session_detail(session_id: str):
    """
    Pagina de detalle de UNA sesion de conversacion: su resumen completo
    (si ya se genero uno), el historial completo de mensajes (marcando
    cuales son triviales o ya fueron incorporados al resumen), y los
    jobs que la fueron atendiendo -- potencialmente varios nodos
    distintos a lo largo de la conversacion, que es justamente lo que
    este modulo de memoria esta pensado para permitir.
    """
    now = time.time()
    with db() as conn:
        session = memory.get_session(conn, session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        session = dict(session)
        messages = [dict(r) for r in memory.list_messages(conn, session_id)]
        jobs = [dict(r) for r in conn.execute("""
          SELECT jobs.*, nodes.name AS node_name FROM jobs
          LEFT JOIN nodes ON nodes.node_id = jobs.assigned_node
          WHERE jobs.session_id=? ORDER BY jobs.created_at DESC
        """, (session_id,)).fetchall()]

    def _message_row(m: dict) -> str:
        flags = []
        if m["trivial"]:
            flags.append("trivial")
        if m["summarized"]:
            flags.append("resumido")
        flags_html = f' <span style="color:#9aa0a6">({", ".join(flags)})</span>' if flags else ""
        role_icon = "🧑" if m["role"] == "user" else "🤖"
        return f"""
        <tr>
          <td>{role_icon} {html.escape(m["role"])}</td>
          <td>{html.escape(m["content"])}{flags_html}</td>
          <td>{m["token_estimate"]}</td>
          <td>{_ago(m["created_at"], now)}</td>
        </tr>"""

    messages_html = "".join(_message_row(m) for m in messages) or \
        "<tr><td colspan='4'>Sin mensajes todavia.</td></tr>"
    jobs_html = "".join(_job_row(j, now) for j in jobs) or \
        "<tr><td colspan='10'>Sin jobs asociados.</td></tr>"

    return HTMLResponse(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>Sauzal · Sesion {session_id[:8]}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e6e6e6; margin:2rem; }}
  h1 {{ font-size:1.4rem; margin-bottom:0.2rem; }}
  h2 {{ font-size:1.1rem; margin-top:2rem; color:#9aa0a6; }}
  table {{ width:100%; border-collapse: collapse; margin-top:0.5rem; }}
  th, td {{ text-align:left; padding:0.5rem 0.7rem; border-bottom:1px solid #2a2d34; font-size:0.9rem; vertical-align:top; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  a {{ color:#7fb0ff; }}
  .summary {{ color:#9aa0a6; font-size:0.9rem; }}
  .card {{ background:#171a20; border:1px solid #2a2d34; border-radius:8px; padding:1rem; margin-top:1rem; }}
  button {{ background:#2a2d34; color:#e6e6e6; border:1px solid #3a3d44; border-radius:4px; padding:0.3rem 0.6rem; cursor:pointer; font-size:0.8rem; }}
  button.danger:hover {{ background:#5c2020; border-color:#7a2a2a; }}
</style>
</head>
<body>
  <p><a href="/admin">← Volver al panel</a></p>
  <h1>Sesion {session_id}</h1>
  <p class="summary">{len(messages)} mensajes · creada hace {_ago(session["created_at"], now)} · ultimo uso hace {_ago(session["last_used_at"], now)}</p>

  <div class="card">
    <b>Resumen acumulado (memoria de largo plazo):</b>
    <p>{html.escape(session["summary"]) if session["summary"] else "Todavia no se genero ningun resumen (hace falta que se acumulen varios mensajes viejos, ver memory.SUMMARIZE_TRIGGER)."}</p>
    <form method="post" action="/admin/sessions/{session_id}/delete" onsubmit="return confirm('¿Borrar esta sesion y toda su memoria? No se puede deshacer.')">
      <button class="danger">Eliminar sesion completa</button>
    </form>
  </div>

  <h2>Mensajes</h2>
  <table>
    <tr><th>Rol</th><th>Contenido</th><th>Tokens (aprox)</th><th>Hace</th></tr>
    {messages_html}
  </table>

  <h2>Jobs de esta sesion</h2>
  <table>
    <tr><th>Job</th><th>Modelo</th><th>Prompt</th><th>Cliente</th><th>Estado</th><th>Nodo</th><th>Tokens (in/out)</th><th>Duracion</th><th>Creado hace</th><th>Acciones</th></tr>
    {jobs_html}
  </table>
</body>
</html>""")


@app.post("/admin/sessions/{session_id}/delete")
def admin_delete_session(session_id: str):
    """Borra la sesion y todos sus mensajes desde el panel (misma logica
    que DELETE /sessions/{{id}}, expuesta como boton en vez de API)."""
    with db() as conn:
        memory.delete_session(conn, session_id)
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/nodes/{node_id}/delete")
def admin_delete_node(node_id: str):
    """
    Borra la fila del nodo en `nodes`. No toca los trabajos que ese nodo
    tenia asignados: quedan en la tabla `jobs` con `assigned_node`
    apuntando a un node_id que ya no existe (el panel los sigue
    mostrando, con el node_id crudo en vez del nombre, via el LEFT JOIN).
    """
    with db() as conn:
        conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/nodes/{node_id}/unstick")
def admin_unstick_node(node_id: str):
    """
    Fuerza `status='available'` en un nodo puntual. Sirve para el caso en
    que un agente se cayo (se cerro la ventana, se apago la PC, se borro
    el Pod) justo mientras tenia un trabajo "running": el servidor nunca
    se entera de que el nodo desaparecio (no hay timeout automatico) y lo
    deja marcado "busy" para siempre, por lo que /infer deja de
    considerarlo disponible aunque el agente vuelva a conectarse y mande
    heartbeats con status="available" (el heartbeat SI lo destraba solo
    en ese caso; este boton es para cuando el nodo no va a volver y hay
    que liberarlo a mano).
    """
    with db() as conn:
        conn.execute("UPDATE nodes SET status='available' WHERE node_id=?", (node_id,))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/nodes/{node_id}/pause")
def admin_pause_node(node_id: str):
    """
    Pausa manualmente un nodo desde el panel: /infer deja de asignarle
    trabajos aunque siga online (ver el filtro `paused=0` en /infer). A
    diferencia de `status='busy'`, esto NO se resetea con los heartbeats
    normales -- se queda pausado hasta un admin_resume_node() explicito.
    """
    with db() as conn:
        conn.execute("UPDATE nodes SET paused=1 WHERE node_id=?", (node_id,))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/nodes/{node_id}/resume")
def admin_resume_node(node_id: str):
    """Reactiva un nodo pausado (ver admin_pause_node)."""
    with db() as conn:
        conn.execute("UPDATE nodes SET paused=0 WHERE node_id=?", (node_id,))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/nodes/{node_id}/price")
def admin_set_price(node_id: str, price_kwh: float = Form(...)):
    """
    Fija el precio de electricidad ($/kWh, en la moneda que use quien
    opera este nodo puntual -- cada nodo puede tener su propia tarifa) que
    se usa junto con `energy_wh` para estimar el costo electrico en la
    pagina de detalle del nodo.
    """
    with db() as conn:
        conn.execute("UPDATE nodes SET price_kwh=? WHERE node_id=?", (price_kwh, node_id))
    return RedirectResponse(f"/admin/nodes/{node_id}", status_code=303)


@app.post("/admin/nodes/{node_id}/reset-energy")
def admin_reset_energy(node_id: str):
    """Reinicia a cero el contador de energia acumulada (`energy_wh`) de
    un nodo, por ejemplo al arrancar un periodo de facturacion nuevo."""
    with db() as conn:
        conn.execute("UPDATE nodes SET energy_wh=0 WHERE node_id=?", (node_id,))
    return RedirectResponse(f"/admin/nodes/{node_id}", status_code=303)


@app.post("/admin/jobs/{job_id}/delete")
def admin_delete_job(job_id: str):
    """Borra la fila del trabajo en `jobs`. Accion irreversible: si el
    job tenia un resultado o error guardado, se pierde."""
    with db() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
    return RedirectResponse("/admin", status_code=303)
