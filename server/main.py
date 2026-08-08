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
import html, json, sqlite3, time, uuid
from contextlib import contextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

# Base de datos SQLite, vive al lado de este archivo (server/sauzal.db).
DB_PATH = Path(__file__).with_name("sauzal.db")
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
        capabilities  -- JSON en texto: que backends/modelos tiene el nodo

    Tabla `jobs`: un registro por cada pedido de inferencia.
        job_id         -- UUID (PK)
        model          -- nombre del modelo pedido (ej: "gemma3:4b", "flux...")
        prompt         -- el prompt de texto o imagen
        status         -- "queued" -> "running" -> "completed" | "failed"
        assigned_node  -- a que nodo se le asigno el trabajo
        client         -- quien lo pidio (dato libre que manda el cliente,
                          ej: su hostname; NO es una identidad autenticada,
                          ver limitaciones mas abajo)
        result         -- JSON en texto con la respuesta (si completed)
        error          -- texto del error (si failed)
        prompt_tokens  -- tokens de entrada consumidos (solo jobs de texto;
                          NULL en jobs de imagen, que no tienen ese concepto)
        output_tokens  -- tokens de salida generados (idem)
        created_at / updated_at -- timestamps (epoch)

    Migracion liviana: si `sauzal.db` ya existia de una version anterior
    de este archivo (sin las columnas client/prompt_tokens/output_tokens),
    se le agregan con ALTER TABLE la primera vez que arranca el servidor
    con este codigo. No hace falta borrar la base a mano.
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
        """)
        existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, coltype in (
            ("client", "TEXT"),
            ("prompt_tokens", "INTEGER"),
            ("output_tokens", "INTEGER"),
        ):
            if column not in existing_cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {coltype}")


# ----------------------------------------------------------------------
# Modelos Pydantic: definen y validan el JSON que entra en cada endpoint.
# ----------------------------------------------------------------------

class Register(BaseModel):
    """Body de POST /nodes/register: como se presenta un agente nuevo."""
    name: str            # nombre legible elegido por el agente (--name)
    capabilities: str    # JSON en texto con lo que el nodo puede hacer

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

class Infer(BaseModel):
    """Body de POST /infer: lo que pide un cliente."""
    model: str = "gemma3:4b"      # modelo de texto (Ollama) o de imagen (ComfyUI)
    prompt: str = Field(min_length=1)
    node: str | None = None       # opcional: forzar un nodo puntual (nombre o node_id)
    client: str | None = None     # quien lo pide (dato libre, ej: hostname del cliente; no autenticado)

class Result(Auth):
    """Body de POST /jobs/{job_id}/result: lo que devuelve el agente al terminar."""
    success: bool
    result: str | None = None   # JSON en texto con la respuesta, si success=True
    error: str | None = None    # texto del error, si success=False
    prompt_tokens: int | None = None   # tokens de entrada, si el backend los reporta (Ollama si, ComfyUI no)
    output_tokens: int | None = None   # tokens de salida, idem


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


@app.post("/nodes/register")
def register(req: Register):
    """
    Da de alta un nodo nuevo. Genera un node_id y un token aleatorios
    (UUID4) y los guarda junto con el nombre y las capacidades declaradas.
    El agente llama esto una sola vez (o cuando borra su agent_config.json)
    y persiste la respuesta para no tener que registrarse de nuevo en cada
    arranque.
    """
    node_id, token = str(uuid.uuid4()), str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            "INSERT INTO nodes VALUES(?,?,?,?,?,?)",
            (node_id, req.name, token, "available", time.time(), req.capabilities)
        )
    return {"node_id":node_id,"token":token}


@app.post("/nodes/heartbeat")
def heartbeat(req: Heartbeat):
    """
    El agente llama esto en bucle (cada ~2s) para:
      1. Probar que sigue autenticado (si el token no matchea, 401).
      2. Actualizar last_seen (asi /nodes lo sigue mostrando "online").
      3. Opcionalmente refrescar sus capabilities (por si cambiaron los
         modelos disponibles en Ollama/ComfyUI desde el ultimo heartbeat).
    """
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        if req.capabilities is None:
            conn.execute(
                "UPDATE nodes SET status=?,last_seen=? WHERE node_id=?",
                (req.status,time.time(),req.node_id)
            )
        else:
            conn.execute(
                "UPDATE nodes SET status=?,last_seen=?,capabilities=? WHERE node_id=?",
                (req.status,time.time(),req.capabilities,req.node_id)
            )
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
            "SELECT node_id,name,status,last_seen,capabilities FROM nodes ORDER BY last_seen DESC"
        ).fetchall()
    return [{**dict(r),"online":now-r["last_seen"]<30} for r in rows]


@app.post("/infer")
def infer(req: Infer):
    """
    Punto de entrada para pedir una inferencia. No requiere autenticacion
    (ver limitacion en el docstring del modulo).

    Eleccion de nodo:
      - Si el cliente mando `node` (nombre o node_id), se busca ESE nodo
        puntual, y debe estar disponible y con heartbeat reciente (<30s),
        sino 503.
      - Si no mando `node`, se toma el nodo disponible visto mas
        recientemente (heuristica simple, no es un balanceador real: no
        pesa carga ni afinidad de modelo/GPU).

    Al asignar el trabajo:
      1. Se inserta la fila en `jobs` con status="queued".
      2. Se marca el nodo como "busy" para que /infer no lo vuelva a
         asignar mientras trabaja (igual el agente hace overwrite de su
         propio status a "available" en cada heartbeat, asi que esto es
         mas una señal de "en este preciso instante tiene un pendiente"
         que un lock estricto).

    El nodo recien se entera del trabajo en su proximo POST /agent/pull.
    """
    now=time.time()
    with db() as conn:
        if req.node:
            node=conn.execute("""
              SELECT node_id FROM nodes
              WHERE status='available' AND last_seen>? AND (node_id=? OR name=?)
              ORDER BY last_seen DESC LIMIT 1
            """,(now-30,req.node,req.node)).fetchone()
        else:
            node=conn.execute("""
              SELECT node_id FROM nodes
              WHERE status='available' AND last_seen>?
              ORDER BY last_seen DESC LIMIT 1
            """,(now-30,)).fetchone()
        if not node:
            raise HTTPException(503,"No available Sauzal nodes")
        job_id=str(uuid.uuid4())
        conn.execute("""
          INSERT INTO jobs(job_id,model,prompt,status,assigned_node,client,created_at,updated_at)
          VALUES(?,?,?,'queued',?,?,?,?)
        """,(job_id,req.model,req.prompt,node["node_id"],req.client,now,now))
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


@app.post("/jobs/{job_id}/result")
def result(job_id: str, req: Result):
    """
    El agente llama esto cuando termina de ejecutar (bien o mal) el
    trabajo que le tocaba. Guarda el resultado/error y status final, y
    libera al nodo (status='available') para que /infer pueda volver a
    asignarle trabajos.

    No valida que el job_id realmente estuviera asignado a este node_id
    (posible mejora: chequear `assigned_node=?` en el UPDATE).
    """
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        status="completed" if req.success else "failed"
        conn.execute("""
          UPDATE jobs SET status=?,result=?,error=?,prompt_tokens=?,output_tokens=?,updated_at=? WHERE job_id=?
        """,(status,req.result,req.error,req.prompt_tokens,req.output_tokens,time.time(),job_id))
        conn.execute(
            "UPDATE nodes SET status='available',last_seen=? WHERE node_id=?",
            (time.time(),req.node_id)
        )
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


def _node_row(row: dict, now: float) -> str:
    """
    Arma la fila <tr> de un nodo para la tabla del panel. Parsea el JSON
    de `capabilities` para mostrar backends y GPU en columnas legibles
    (si el JSON esta corrupto o vacio, se muestran vacios en vez de
    romper toda la pagina).

    Todo texto que viene de datos guardados por agentes/clientes (nombre,
    GPU) pasa por `html.escape` antes de insertarse en el HTML, porque
    esos valores no son de confianza (el nombre de un nodo, por ejemplo,
    lo elige quien arranca el agente con --name, y /infer no tiene
    autenticacion) -- sin este escape, alguien podria registrar un nodo
    con un nombre tipo "<script>...</script>" y ejecutar JavaScript en el
    navegador de quien abra este panel.
    """
    try:
        caps = json.loads(row["capabilities"] or "{}")
    except json.JSONDecodeError:
        caps = {}
    online = now - row["last_seen"] < 30
    services = caps.get("services", {}) if isinstance(caps, dict) else {}
    backends = ", ".join(k for k, v in services.items() if v) or "-"
    gpu = ", ".join(caps.get("gpu", [])) if isinstance(caps, dict) else ""
    badge = "🟢 online" if online else "⚪ offline"
    name = html.escape(row["name"] or "")
    return f"""
    <tr>
      <td>{name}</td>
      <td>{badge}</td>
      <td>{html.escape(row["status"] or "")}</td>
      <td>{_ago(row["last_seen"], now)}</td>
      <td>{html.escape(backends)}</td>
      <td title="{html.escape(gpu)}">{html.escape(gpu[:40])}</td>
      <td class="actions">
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
    client = html.escape(row.get("client") or "-")
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
    return f"""
    <tr>
      <td><a href="/jobs/{row['job_id']}" target="_blank">{row['job_id'][:8]}…</a></td>
      <td>{html.escape(row["model"] or "")}</td>
      <td title="{html.escape(prompt)}">{html.escape(prompt_short)}</td>
      <td>{client}</td>
      <td>{status_icon} {html.escape(row["status"] or "")}</td>
      <td>{html.escape(node_label)}</td>
      <td title="tokens de entrada/salida">{tokens}</td>
      <td>{_ago(row["created_at"], now)}</td>
      <td class="actions">
        <form method="post" action="/admin/jobs/{row['job_id']}/delete" onsubmit="return confirm('¿Eliminar este trabajo?')">
          <button class="danger">Eliminar</button>
        </form>
      </td>
    </tr>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    """
    Pagina principal del panel: trae TODOS los nodos y los ultimos 200
    trabajos (mas nuevos primero) y arma una unica pagina HTML con dos
    tablas. No pagina los nodos (en la practica nunca son tantos como
    para necesitarlo); los jobs si se limitan a 200 para que la tabla no
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

    nodes_html = "".join(_node_row(n, now) for n in nodes) or \
        "<tr><td colspan='7'>No hay nodos registrados todavia.</td></tr>"
    jobs_html = "".join(_job_row(j, now) for j in jobs) or \
        "<tr><td colspan='9'>No hay trabajos todavia.</td></tr>"
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
  <p class="summary">{len(nodes)} nodos registrados ({online_count} online) · {len(jobs)} trabajos mostrados (ultimos 200) · se actualiza sola cada 5s</p>

  <h2>Nodos</h2>
  <table>
    <tr><th>Nombre</th><th>Online</th><th>Status</th><th>Ultimo heartbeat</th><th>Backends</th><th>GPU</th><th>Acciones</th></tr>
    {nodes_html}
  </table>

  <h2>Trabajos</h2>
  <table>
    <tr><th>Job</th><th>Modelo</th><th>Prompt</th><th>Cliente</th><th>Estado</th><th>Nodo</th><th>Tokens (in/out)</th><th>Creado hace</th><th>Acciones</th></tr>
    {jobs_html}
  </table>
</body>
</html>""")


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


@app.post("/admin/jobs/{job_id}/delete")
def admin_delete_job(job_id: str):
    """Borra la fila del trabajo en `jobs`. Accion irreversible: si el
    job tenia un resultado o error guardado, se pierde."""
    with db() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
    return RedirectResponse("/admin", status_code=303)
