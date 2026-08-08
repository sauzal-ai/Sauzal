# Sauzal — Arquitectura general

Este documento explica el proyecto completo desde arriba: qué es cada
programa, cómo se relacionan entre sí, qué hay que levantar y dónde, y
cómo está armada la base de datos. Para instrucciones paso a paso del
primer PoC en red local, ver [`README.md`](../README.md). Para el
detalle linea por linea de cada archivo, los propios archivos ya tienen
comentarios y docstrings explicando cada función.

## 1. El concepto, en una frase

Sauzal es una red de inferencia distribuida: un **servidor central**
(control plane) recibe pedidos de texto o imagen de cualquier
**cliente**, y se los reparte a **nodos remotos** (PCs o Pods en la nube
con GPU) que hacen el trabajo pesado y devuelven el resultado. Ningún
nodo remoto necesita IP pública ni puertos abiertos: siempre es el nodo
el que se conecta hacia el servidor, nunca al revés.

```
   CLIENTE                    SERVIDOR                    NODO REMOTO
 (client/infer.py)      (server/main.py, FastAPI)      (agent/agent.py)
       |                          |                            |
       |--- POST /infer --------->|                            |
       |<-- job_id, nodo elegido -|                            |
       |                          |<--- heartbeat (cada 2s) ---|
       |                          |<--- "¿tengo trabajo?" -----|
       |                          |---- si, este job ---------->|
       |                          |                       ejecuta con
       |                          |                    Ollama o ComfyUI
       |                          |<--- resultado --------------|
       |--- GET /jobs/{id} ------>|                            |
       |<-- (polling hasta        |                            |
       |     "completed") --------|                            |
```

El servidor nunca ejecuta nada de IA él mismo — solo coordina. Todo el
cómputo pesado pasa en la máquina del nodo remoto.

## 2. Los programas del repo

| Archivo | Rol | Dónde corre |
|---|---|---|
| `server/main.py` | Control plane (API FastAPI + base SQLite) | Un solo lugar central (tu PC, o idealmente un VPS) |
| `server/memory.py` | Memoria de conversación: sesiones, mensajes, recuperación semántica (ver sección 6) | Se importa dentro de `main.py`, no se corre solo |
| `client/infer.py` | Pide una inferencia y espera el resultado | Cualquier PC que quiera *usar* la red |
| `agent/agent.py` | Se conecta al servidor, recibe trabajos y los ejecuta | Cada nodo que *aporta* GPU |
| `agent/comfy.py` | Módulo del agente: backend de generación de imagen (ComfyUI) | Se importa dentro de `agent.py`, no se corre solo |
| `Dockerfile` + `entrypoint.sh` | Arman la imagen de contenedor para nodos en la nube (RunPod) | Se construye en CI, corre dentro del Pod |
| `.github/workflows/build-image.yml` | Compila y publica esa imagen a Docker Hub automáticamente | GitHub Actions (nube), no en tu PC |

### 2.1 `server/main.py` — el control plane

Es la única pieza que sabe de todo: qué nodos existen, cuáles están
vivos, qué trabajos hay pendientes y sus resultados. Guarda todo en un
archivo SQLite (`server/sauzal.db`, se crea solo al arrancar). Expone
una API HTTP con estos endpoints:

- `GET /health` — chequeo de vida.
- `POST /nodes/register` — un nodo nuevo se da de alta, recibe credenciales.
- `POST /nodes/heartbeat` — un nodo avisa "sigo vivo" + qué puede hacer.
- `GET /nodes` — lista todos los nodos conocidos y si están online.
- `POST /infer` — un cliente pide una inferencia; se encola y se asigna a un nodo.
- `POST /agent/pull` — un nodo pregunta si tiene trabajo asignado.
- `POST /jobs/{id}/result` — un nodo entrega el resultado (o el error).
- `GET /jobs/{id}` — un cliente consulta el estado/resultado de un pedido.
- `POST /sessions` — crea una sesión de conversación, devuelve `session_id`.
- `GET /sessions/{id}` — info resumida de una sesión (mensajes, resumen).
- `GET /sessions/{id}/messages` — historial completo de mensajes.
- `DELETE /sessions/{id}` — borra la sesión y toda su memoria asociada.
- `GET /bench/payload` — devuelve un bloque fijo de 2MB; un nodo lo
  descarga una sola vez al registrarse para medir su velocidad de red
  hacia el servidor.
- `GET /admin` — panel web (HTML) para ver todos los nodos y trabajos en
  vivo, y hacer cambios manuales: eliminar un nodo, forzarlo de vuelta a
  "available" si quedó colgado en "busy", pausarlo/reactivarlo, o
  eliminar un trabajo.
- `GET /admin/nodes/{id}` — página de detalle de un nodo puntual: VRAM,
  RAM, driver, motor de cómputo, versiones, benchmark, latencia,
  velocidad de red, temperatura/consumo en vivo, energía acumulada +
  costo estimado, IP/ubicación, y su historial de trabajos fallidos.
- `POST /admin/nodes/{id}/pause` y `/resume` — pausar/reactivar
  manualmente un nodo (deja de recibir trabajos aunque esté online).
- `POST /admin/nodes/{id}/price` — fija el precio de electricidad
  ($/kWh) de ese nodo, usado para estimar el costo eléctrico.
- `POST /admin/nodes/{id}/reset-energy` — reinicia a cero el contador de
  energía acumulada de un nodo.
- `GET /admin/clients/{nombre}` — página de detalle de un cliente
  puntual: IP, ubicación, sistema operativo, procesador, si el origen
  parece un navegador o un script, y su historial completo de trabajos.
  No existe una tabla `clients`: esta página agrega la tabla `jobs` por
  el campo `client` (ver sección 5).
- `GET /admin/sessions/{id}` — detalle de una sesión de conversación:
  resumen acumulado, historial completo de mensajes, y los jobs que la
  fueron atendiendo (ver sección 6).
- `POST /admin/sessions/{id}/delete` — borra una sesión desde el panel.

Todo lo de `/admin*` es sin autenticación (igual que el resto del
servidor) — pensado para uso en LAN/desarrollo, no para exponer a
internet sin agregarle protección.

### 2.2 `client/infer.py` — el consumidor

Un script chico (solo depende de `requests`) que cualquiera puede copiar
a su PC para mandar preguntas a la red, sin instalar nada del resto del
proyecto. Manda el pedido, y hace *polling* del resultado hasta que
termina.

### 2.3 `agent/agent.py` + `agent/comfy.py` — el que aporta GPU

Es el programa que convierte una PC (o un Pod) en un nodo útil para la
red. Al arrancar:

1. Revisa qué tiene disponible localmente: Ollama corriendo (texto) y/o
   ComfyUI corriendo con el modelo FLUX (imagen).
2. Se registra contra el servidor (una sola vez; guarda las credenciales
   en `agent/agent_config.json` para no repetirlo).
3. Entra en bucle: heartbeat + "¿hay trabajo para mí?" cada 2 segundos.
   Si le llega un trabajo, `agent.py` decide con qué backend correrlo
   (según el nombre del modelo pedido) y `comfy.py` es el que sabe hablar
   con ComfyUI si el trabajo es de imagen.

Un mismo nodo puede tener los dos backends a la vez (como probamos:
Ollama con `gemma3:4b` + ComfyUI con FLUX en el mismo Pod), o solo uno
(como las PCs de este proyecto, que solo corren Ollama).

### 2.4 `Dockerfile` / `entrypoint.sh` / workflow de CI — nodos en la nube

Estos tres archivos no son "un programa" que se corre a mano: arman la
**imagen** que va a un Pod de RunPod (o cualquier host con GPU NVIDIA y
Docker). El `Dockerfile` instala ComfyUI, FLUX, Ollama y el propio
`agent.py` dentro de la imagen. `entrypoint.sh` es lo primero que corre
el contenedor: arranca Ollama, arranca ComfyUI, espera a que ambos
respondan, y recién ahí lanza `agent.py`. El workflow de GitHub Actions
es el que compila esa imagen y la sube a Docker Hub cada vez que se
modifica algo relevante, para no tener que hacerlo a mano con Docker
Desktop local.

## 3. Topología: qué corre dónde

Estos son los tres roles que puede tener cualquier máquina en esta red.
Una misma PC puede cumplir varios roles a la vez (por ejemplo, en las
pruebas de este proyecto, una PC fue servidor + nodo al mismo tiempo).

| Rol | Programa que corre | Requisitos |
|---|---|---|
| **Servidor** (uno solo, central) | `uvicorn server.main:app` | Python + `pip install -r requirements.txt`. Debe ser alcanzable por los otros dos roles (misma LAN, o expuesto con un túnel/VPS si los nodos están afuera). |
| **Nodo con GPU** (tantos como se quiera) | `python agent/agent.py --server URL --name NOMBRE --register` | Ollama instalado (para texto) y/o ComfyUI+FLUX corriendo (para imagen), más `pip install -r agent/requirements.txt`. O bien, correr directamente la imagen Docker en un Pod con GPU. |
| **Cliente** (tantos como se quiera) | `python client/infer.py --server URL --prompt "..."` | Solo `pip install requests`. |

### Cómo se probó en este proyecto, como referencia concreta

- **Servidor**: corriendo en la PC de escritorio (`192.168.1.78`, puerto
  8000), con `uvicorn`. Para que un Pod de RunPod (en internet) pudiera
  alcanzarlo, se expuso con un túnel de Cloudflare (`cloudflared tunnel
  --url http://localhost:8000`), que da una URL pública tipo
  `https://xxxx.trycloudflare.com`. Para nodos en la misma red local
  (LAN), no hace falta túnel: alcanza con la IP local y abrir el puerto
  en el Firewall de Windows.
- **Nodo 1** (`rx6600-test`): la misma PC de escritorio, con Ollama
  nativo de Windows usando la GPU AMD Radeon RX 6600.
- **Nodo 2** (`pc-elmar`): otra PC de la misma red, también con Ollama
  nativo, agente corriendo con `--server http://192.168.1.78:8000`.
- **Nodo 3** (Pod de RunPod): la imagen Docker de este repo, con
  `SAUZAL_SERVER` apuntando a la URL del túnel de Cloudflare, exponiendo
  el puerto 8188 (ComfyUI).

## 4. Cómo levantar todo, paso a paso

**A) Arrancar el servidor** (en la máquina que va a ser el control plane):
```powershell
cd Sauzal
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```
Confirmar que responde: `curl http://localhost:8000/health`.

**B) Sumar un nodo con GPU** (en cada PC que va a aportar cómputo):
```powershell
# Si va a hacer texto, necesita Ollama instalado y un modelo bajado:
winget install --id Ollama.Ollama -e
ollama pull gemma3:4b

cd Sauzal
pip install -r agent/requirements.txt
python agent\agent.py --server http://IP_DEL_SERVIDOR:8000 --name mi-nodo --register
# Para que arranque pausado (no recibe trabajos hasta reactivarlo en /admin):
python agent\agent.py --server http://IP_DEL_SERVIDOR:8000 --name mi-nodo --register --paused
```
Al registrarse, el agente corre una vez un benchmark fijo (Ollama y/o
ComfyUI, según lo que tenga disponible) y mide su velocidad de red hacia
el servidor — por eso el primer registro puede tardar unos segundos más
que los siguientes arranques.

Si en cambio es un Pod de RunPod con la imagen Docker de este repo, no
hay que instalar nada a mano: solo correr el Pod con la imagen
`usuario/sauzal-node:latest` y la variable de entorno `SAUZAL_SERVER`
apuntando a la URL pública del servidor.

**C) Mandar un pedido** (desde cualquier PC, incluso el mismo servidor):
```powershell
pip install requests
python client\infer.py --server http://IP_DEL_SERVIDOR:8000 --model gemma3:4b --prompt "Hola"
# Para elegir un nodo puntual en vez de dejar que el servidor elija:
python client\infer.py --server http://IP_DEL_SERVIDOR:8000 --model gemma3:4b --node mi-nodo --prompt "Hola"
```

**Verificar el estado de la red en cualquier momento:**
```
GET http://IP_DEL_SERVIDOR:8000/nodes
```
Muestra todos los nodos conocidos, si están `online` (heartbeat en los
últimos 30 segundos), y sus capacidades (`ollama_models`,
`image_models`, etc.).

## 5. La base de datos (`server/sauzal.db`, SQLite)

Se crea sola al arrancar el servidor (`startup()` en `server/main.py`,
con `CREATE TABLE IF NOT EXISTS`). Son solo dos tablas.

### Tabla `nodes` — un registro por cada agente que se registró alguna vez

| Columna | Tipo | Quién la escribe | Cuándo |
|---|---|---|---|
| `node_id` | TEXT (PK) | Servidor | Al registrar (`POST /nodes/register`), genera un `uuid4()` nuevo. Nunca cambia después. |
| `name` | TEXT | Servidor, con el dato que manda el agente | Al registrar. El agente lo elige con `--name` (o usa el hostname si no se especifica). No se actualiza después — si un agente se re-registra, se crea una fila nueva, no se pisa la vieja. |
| `token` | TEXT | Servidor | Al registrar, genera un `uuid4()` nuevo. Es la "contraseña" del nodo: el agente la guarda en `agent/agent_config.json` y la manda en cada llamada autenticada (heartbeat, pull, result) para probar que es quien dice ser. |
| `status` | TEXT (`available` / `busy`) | Servidor | Se pone en `available` al registrar; el propio servidor lo pasa a `busy` cuando le asigna un job (`POST /infer`) y de vuelta a `available` cuando el agente entrega el resultado (`POST /jobs/{id}/result`). También se sobreescribe con lo que mande el agente en cada `POST /nodes/heartbeat` (normalmente siempre manda `"available"`). |
| `last_seen` | REAL (timestamp epoch) | Servidor | Se actualiza al registrar, en cada `POST /nodes/heartbeat` (cada ~2s, mientras el agente esté corriendo), y también al entregar un resultado. Se usa para calcular si un nodo está `online` (diferencia menor a 30s respecto a "ahora") tanto en `GET /nodes` como al elegir nodo en `POST /infer`. |
| `capabilities` | TEXT (JSON serializado) | Servidor, con el dato que manda el agente | Al registrar, y refrescado en cada heartbeat. Ver el detalle completo de campos más abajo. |
| `paused` | INTEGER (0/1) | Servidor, con el valor inicial que manda el agente (`--paused`) o una acción manual en `/admin` | Al registrar (valor inicial) y en `POST /admin/nodes/{id}/pause`\|`/resume`. Los heartbeats normales **nunca** tocan esta columna a propósito, para que una pausa hecha desde el panel no se pierda mientras el agente sigue mandando heartbeats. `/infer` excluye siempre los nodos con `paused=1`, aunque estén online. |
| `price_kwh` | REAL | Servidor, cargado a mano en `/admin` | En `POST /admin/nodes/{id}/price`. `NULL` hasta que se configure — sin esto no se puede estimar costo, solo se ve el consumo en W. |
| `energy_wh` | REAL | Servidor, acumulando lo que manda el agente en cada heartbeat | En cada `POST /nodes/heartbeat` se suma `energy_wh_delta` (consumo instantáneo × tiempo transcurrido desde el heartbeat anterior). Es una **estimación**, no un medidor real, y solo se acumula en nodos donde se puede leer el consumo (NVIDIA vía `nvidia-smi`; en otras GPUs queda en 0 para siempre). Se puede reiniciar a mano con `POST /admin/nodes/{id}/reset-energy`. |
| `latency_ms` | REAL | Servidor, con el dato que manda el agente | En cada `POST /nodes/heartbeat`. Es el tiempo de ida y vuelta del heartbeat ANTERIOR (medido por el propio agente) — sirve como proxy de qué tan lejos/lenta está la conexión de ese nodo con el servidor. |
| `ip_address` | TEXT | Servidor, calculada de la conexión HTTP | Al registrar y en cada heartbeat. Ver `client_ip()`: usa el header `CF-Connecting-IP` (Cloudflare) o `X-Forwarded-For` si existen, si no la IP directa de la conexión TCP. **El agente no manda ni puede falsificar este campo.** |
| `location` | TEXT | Servidor, resuelta a partir de `ip_address` | Al registrar y en cada heartbeat (cacheada por IP, ver `geolocate()`), consultando el servicio externo gratuito `ip-api.com`. `NULL` si la IP es privada/LAN (127.0.0.1, 192.168.x.x, etc.) o si la consulta externa falla. |

**Sobre el envío de IPs a un servicio externo:** `geolocate()` manda la IP pública (nunca las privadas) a `ip-api.com` para resolver ciudad/país. Es una decisión consciente de diseño, confirmada explícitamente al implementar esta función — si en algún momento se quiere sacar esa dependencia externa, alcanza con hacer que `geolocate()` devuelva siempre `None`.

**Los campos dentro de `capabilities` (JSON armado por `agent.py::capabilities()`):**

| Campo | Qué es | Cómo se obtiene | Limitación |
|---|---|---|---|
| `hostname`, `os` | Identifican la máquina | `socket`/`platform` de Python | — |
| `gpu` | Nombre(s) de GPU | `nvidia-smi` si es NVIDIA; si no, WMI en Windows (`Win32_VideoController`) | En Linux sin NVIDIA queda vacío (no se agregó una dependencia extra para eso) |
| `ollama_models`, `image_models`, `services` | Qué modelos/backends tiene activos | `GET /api/tags` de Ollama y `comfy.py::available()` | — |
| `vram_total_mb`, `vram_used_mb` | VRAM de la GPU | `nvidia-smi`; si no hay NVIDIA, se usa `/system_stats` de ComfyUI (VRAM vista por PyTorch) como alternativa | `NULL` si no hay NVIDIA y ComfyUI tampoco está corriendo |
| `ram_total_mb`, `ram_used_mb` | RAM del sistema (no de la GPU) | WMI en Windows / `/proc/meminfo` en Linux | Es una **foto tomada al arrancar el agente**, no se refresca en cada heartbeat (para no pagar el costo de esa consulta cada 2s) |
| `driver_version` | Versión del driver de la GPU | `nvidia-smi`; si no, WMI en Windows | `NULL` en Linux sin NVIDIA |
| `compute_backend` | Motor de cómputo estimado (`CUDA`, `ROCm/Vulkan`, `Vulkan/oneAPI`, `CPU`) | Heurística según qué GPU se detectó | Es una estimación, no 100% autoritativa |
| `ollama_version`, `comfy_python_version` | Versiones de software | `GET /api/version` de Ollama; `/system_stats` de ComfyUI | `NULL` si el backend correspondiente no está activo |
| `benchmark` | Desempeño real medido (`tokens_per_sec` y/o `image_seconds`) | Se corre **una sola vez**, al registrarse: un prompt fijo a Ollama y/o una generación mínima en ComfyUI | Agrega demora al primer registro; no se vuelve a medir salvo que el agente se reinicie |
| `network_mbps` | Velocidad de red hacia el servidor | Descarga única de `GET /bench/payload` (2MB) al registrarse | Mide contra el servidor Sauzal, no es un speedtest genérico |
| `gpu_temp_c`, `gpu_power_w` | Temperatura y consumo en vivo | `nvidia-smi`, recalculado en cada heartbeat | `NULL` en GPUs no-NVIDIA (sin herramienta estándar disponible sin dependencias extra) |

### Tabla `jobs` — un registro por cada pedido de inferencia

| Columna | Tipo | Quién la escribe | Cuándo |
|---|---|---|---|
| `job_id` | TEXT (PK) | Servidor | Al crear el pedido (`POST /infer`), genera un `uuid4()` nuevo. |
| `model` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido. Es el nombre de modelo que decide, del lado del agente, si el trabajo va a Ollama (texto) o a ComfyUI (imagen prefijo `flux`/`sdxl`/`sauzal-image`). |
| `prompt` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido. Texto plano, o un JSON con parámetros si es un pedido de imagen (ver `agent/comfy.py::parse_request`). |
| `status` | TEXT (`queued`→`running`→`completed`\|`failed`) | Servidor | `queued` al crear (`POST /infer`); `running` cuando el nodo asignado hace *pull* exitoso del trabajo (`POST /agent/pull`); `completed` o `failed` cuando el nodo entrega el resultado (`POST /jobs/{id}/result`, según el campo `success` que mande). |
| `assigned_node` | TEXT (FK lógica a `nodes.node_id`) | Servidor | Al crear el pedido: es el `node_id` que el servidor eligió (el nodo disponible visto más recientemente, o el pedido explícitamente por el cliente vía el campo `node`). No cambia después — si ese nodo se cae a mitad de camino, el job queda huérfano (no hay reintento ni reasignación automática). |
| `client` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido (`POST /infer`, campo `client`). Identifica "quién lo pidió" — por defecto, `client/infer.py` manda su propio hostname. **No es una identidad autenticada**: cualquiera puede mandar el valor que quiera, es solo informativo (mismo criterio de confianza que el `name` de los nodos). |
| `client_ip` | TEXT | Servidor, calculada de la conexión HTTP | Al crear el pedido, con `client_ip()` (mismo mecanismo que usan los nodos). No lo puede falsificar el cliente. |
| `client_location` | TEXT | Servidor, resuelta a partir de `client_ip` | Al crear el pedido, vía `geolocate()` (cacheada por IP). `NULL` en LAN o si falla la consulta externa. |
| `client_user_agent` | TEXT | Servidor, del header `User-Agent` | Al crear el pedido. Lo manda automáticamente CUALQUIER cliente HTTP sin que nadie lo declare a mano — sirve para distinguir un navegador (`Mozilla/...`) de un script (`python-requests/...`, `curl/...`) vía la heurística `_client_origin()`. |
| `client_os`, `client_processor` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido. `client/infer.py` los arma con el módulo `platform` de Python. **No autenticados**, mismo criterio que `client`. |
| `result` | TEXT (JSON serializado) | El AGENTE, vía el servidor | Al entregar el resultado (`POST /jobs/{id}/result`), solo si `success=true`. El JSON lo arma el agente: `agent.py::infer()` para texto (con `response`, tokens, duraciones) o `agent/comfy.py::generate()` para imagen (con las imágenes en base64, dimensiones, seed, etc). |
| `error` | TEXT | El AGENTE, vía el servidor | Al entregar el resultado, solo si `success=false`. Es el `repr()` de la excepción que capturó `agent.py` al ejecutar el trabajo. |
| `prompt_tokens` | INTEGER | El AGENTE, vía el servidor | Al entregar el resultado. Solo en jobs de **texto**: `agent.py` extrae `prompt_tokens` del JSON que ya arma `infer()` (viene de Ollama, campo `prompt_eval_count`). En jobs de **imagen** queda `NULL` (ComfyUI no tiene concepto de tokens). |
| `output_tokens` | INTEGER | El AGENTE, vía el servidor | Ídem `prompt_tokens`, pero de salida (`eval_count` de Ollama). `NULL` en jobs de imagen. |
| `duration_ms` | REAL | El AGENTE, vía el servidor | Al entregar el resultado, tanto si `success=true` como `success=false`. Es el tiempo real que tardó `execute(job)` de punta a punta, cronometrado por el propio agente — a diferencia de `prompt_tokens`/`output_tokens`, funciona igual para jobs de texto **e imagen**, y también queda registrado cuánto tardó un job que terminó fallando. |
| `created_at` | REAL (timestamp epoch) | Servidor | Al crear el pedido. |
| `updated_at` | REAL (timestamp epoch) | Servidor | Se actualiza cuando el job pasa a `running` (pull) y de nuevo cuando pasa a `completed`/`failed` (result). |

> `client`/`prompt_tokens`/`output_tokens`/`duration_ms`/`client_ip`/
> `client_location`/`client_user_agent`/`client_os`/`client_processor`
> (en `jobs`) y
> `paused`/`price_kwh`/`energy_wh`/`latency_ms`/`ip_address`/`location`
> (en `nodes`) se agregaron después de la versión inicial del esquema.
> El servidor migra sola cualquier `sauzal.db` vieja al arrancar
> (`ALTER TABLE` si faltan columnas), así que no hace falta borrar la
> base para actualizar.

### Ni el "historial de fallas" ni "clientes" son tablas aparte

La página de detalle de un nodo (`/admin/nodes/{id}`) muestra sus
trabajos fallidos, pero eso NO se guarda en una columna nueva: se
calcula al vuelo con `SELECT ... FROM jobs WHERE assigned_node=? AND
status='failed'`.

De la misma forma, **no existe una tabla `clients`**. Un "cliente" no se
registra ni manda heartbeat (a diferencia de un nodo) — es solo un
nombre libre que viaja en cada `POST /infer`. La tabla de Clientes en
`/admin` y la página `/admin/clients/{nombre}` se arman agregando
`jobs` por la columna `client` (agrupando en Python, ver
`admin_panel()`): el "último visto"/IP/SO que se muestran son los del
job MÁS RECIENTE con ese nombre, no un estado en vivo. Como `jobs` ya
guarda todo lo necesario, alcanza con consultarla — no hace falta una
tabla nueva ni un ciclo de registro/heartbeat para los clientes.

### Dato que NO vive en la base: `agent/agent_config.json`

Cada nodo guarda localmente (no en el servidor) un archivo con
`{"server": "...", "node_id": "...", "token": "..."}`. Es lo que le
permite al agente reconectarse tras un reinicio sin volver a registrarse
como un nodo nuevo. Si se borra este archivo (o se corre con
`--register`), el agente se registra de cero y aparece como una fila
NUEVA en la tabla `nodes` — por eso en las pruebas de este proyecto a
veces se ven nodos viejos duplicados y `offline` en `GET /nodes`: son
registros de sesiones anteriores que nunca se limpian solos.

## 6. Memoria de conversación (sesiones)

Toda la lógica vive en **`server/memory.py`** — un módulo nuevo, sin
dependencias externas, que le da a Sauzal la capacidad de mantener una
conversación con contexto **aunque cada mensaje lo procese un nodo/GPU
distinto**. El agente no cambió ni una línea para soportar esto: sigue
recibiendo un prompt de texto plano como siempre.

### La idea central

El servidor arma, antes de encolar el job, un prompt "aplanado" que
combina: un resumen de la conversación (si ya se generó uno), los
mensajes viejos semánticamente relevantes a la pregunta actual, y los
últimos mensajes recientes tal cual. **Ese prompt combinado es el que
recibe el nodo** — para el nodo es un prompt de texto suelto, igual que
cualquier otro `/infer`. El nodo A nunca se entera de que existe una
sesión ni necesita recordar nada para que el nodo B pueda continuar la
misma conversación después.

```
Turno 1 (nodo A):
  Cliente -> POST /infer {session_id, prompt:"Mi auto favorito es un BMW Isetta"}
  Servidor guarda el mensaje, arma el job (sin contexto previo: es el primero)
  Nodo A ejecuta, responde -> servidor guarda la respuesta como mensaje

  ... pasan varios turnos, el mensaje sale de la ventana reciente ...

Turno N (nodo B, un nodo DISTINTO):
  Cliente -> POST /infer {session_id, prompt:"Cual es mi auto favorito?"}
  Servidor busca en los mensajes viejos de la sesion los mas relevantes
  a "Cual es mi auto favorito?" -> encuentra el mensaje del BMW Isetta
  Servidor arma el prompt final:
      "Contexto relevante de mensajes anteriores:
       - Mi auto favorito es un BMW Isetta

       Conversacion reciente:
       ...
       Usuario: Cual es mi auto favorito?
       Asistente:"
  ESE prompt (no el original) es lo que recibe el nodo B.
  El nodo B responde "BMW Isetta" sin haber visto nunca el turno 1.
```

Esto está probado en vivo (ver "Cómo probarlo" más abajo) y con un test
automático (`tests/test_sessions.py::test_conversation_survives_across_different_nodes`)
que simula exactamente este escenario con dos nodos falsos.

### Motor de similitud: TF-IDF, no un modelo de embeddings

Se evaluó agregar un modelo de embeddings neuronal (por ejemplo, via
`/api/embeddings` de Ollama), pero se descartó para la primera versión:
obligaría al SERVIDOR (que puede correr en una máquina sin GPU) a
depender de que haya un nodo disponible solo para vectorizar texto, lo
cual complica la arquitectura para un beneficio marginal en conversaciones
del tamaño típico de un chat (decenas o cientos de mensajes, no miles).

En cambio, `memory.py` implementa **TF-IDF + similitud coseno en Python
puro**, sin ninguna librería nueva: es el modelo vectorial clásico de
recuperación de información (siempre fue "semántico/vectorial" en el
sentido de la IR clásica, aunque no sea una red neuronal), y se recalcula
al vuelo en cada búsqueda porque el corpus es chico (los mensajes de UNA
sola sesión). Si en el futuro hace falta más precisión, alcanza con
reemplazar la función `_vectorize()` de `memory.py` por embeddings reales
sin tocar el resto del diseño — toda la recuperación pasa por
`top_similar()`.

### Las tres capas de memoria

| Capa | Qué es | Cómo se arma |
|---|---|---|
| **Corto plazo** | Los últimos `SHORT_TERM_WINDOW` (6) mensajes, tal cual, sin resumir | Se incluyen siempre en el prompt |
| **Semántica/vectorial** | Hasta `SEMANTIC_TOP_K` (3) mensajes viejos, elegidos por similitud TF-IDF con la pregunta actual | `memory.top_similar()`, con un piso de similitud (`SEMANTIC_MIN_SCORE`) para no traer ruido |
| **Largo plazo (resumen)** | Un resumen acumulado de TODOS los mensajes viejos de la sesión | Generado automáticamente por un job interno cuando hay demasiados mensajes sin resumir (ver abajo) |

Los mensajes triviales ("hola", "gracias", "ok" — ver `memory.is_trivial()`)
se guardan igual (hacen falta para la memoria de corto plazo), pero se
excluyen del banco de candidatos para la búsqueda semántica.

### Resumen automático de conversaciones largas

Cuando una sesión acumula `SUMMARIZE_TRIGGER` (12) o más mensajes viejos
sin resumir, el servidor encola **un job interno `kind='summarize'`**
usando exactamente el mismo mecanismo que cualquier otro job: se le
asigna a un nodo disponible (`_pick_available_node()`), el nodo lo
procesa como un prompt de texto más (le pide "resumí esta conversación
preservando lo importante"), y cuando el resultado vuelve por el mismo
`POST /jobs/{id}/result` de siempre, el servidor detecta que es un job de
resumen y guarda el resultado en `sessions.summary` en vez de crear un
mensaje nuevo, marcando los mensajes cubiertos como `summarized=1`. Hay
una guarda (`_maybe_enqueue_summary()`) para no encolar un resumen nuevo
si ya hay uno en curso para esa sesión.

### Métricas de contexto por job

Cada job de tipo `chat` que pertenece a una sesión guarda cuánto contexto
se le inyectó: `context_tokens_estimate` (aproximado, ~4 caracteres por
token — no hay tokenizador real del lado del servidor), `context_messages_used`
(cuántos mensajes de la ventana reciente entraron) y `context_semantic_hits`
(cuántos mensajes viejos se recuperaron por similitud). Se pueden ver en
`/admin` y en `GET /jobs/{id}`.

### Aislamiento y borrado

Todo se filtra por `session_id` — no hay forma de que una sesión vea
mensajes de otra (probado en `test_sessions_are_isolated_from_each_other`).
`DELETE /sessions/{id}` (o el botón "Eliminar" en `/admin/sessions/{id}`)
borra la sesión y TODOS sus mensajes sin dejar rastro. La columna
`sessions.tenant_id` queda reservada para el día que haga falta aislar
sesiones por cliente/organización — hoy siempre es `NULL` y no filtra
nada, pero el esquema ya la tiene para no requerir una migración futura.

### Compatibilidad con lo existente

Un `POST /infer` **sin** `session_id` funciona exactamente igual que
antes de que existiera este módulo: el prompt no se toca, no se crea
ningún mensaje, `jobs.session_id` queda `NULL`. Esto está cubierto por
`test_stateless_infer_is_unaffected_by_sessions`. Los modelos de imagen
tampoco reciben contexto conversacional inyectado (no tendría sentido
mezclar una transcripción de texto con un prompt de imagen), aunque el
mensaje se sigue guardando para que el historial de la sesión quede
completo.

### Multi-sesión y concurrencia real

Un mismo cliente puede tener **N sesiones activas en paralelo**, cada
una con su propio historial, resumen y memoria semántica — no hay
ningún límite artificial ni relación entre "cliente" (el campo `client`,
solo informativo) y "sesión": el aislamiento es pura y exclusivamente
por `session_id`. Dos sesiones nunca comparten candidatos de búsqueda
semántica entre sí (`memory.build_context()` solo consulta los mensajes
de ESA sesión), así que no hay forma de que se "contaminen".

Como FastAPI corre cada request de forma síncrona en un thread pool,
esto además funciona con **concurrencia real** (no solo intercalado):
dos inferencias de sesiones distintas literalmente se ejecutan al mismo
tiempo, en threads distintos. Dos cosas hacen que esto sea seguro:

1. **SQLite en modo WAL** (`PRAGMA journal_mode=WAL`, activado una vez
   en `startup()`): sin esto, bajo carga concurrente el modo por
   defecto de SQLite puede devolver errores de `"database is locked"`.
2. **Un lock en memoria por `session_id`** (`_lock_for_session()` en
   `server/main.py`): si dos requests le pegan a la MISMA sesión al
   mismo tiempo (por ejemplo, el cliente dispara dos mensajes de la
   misma conversación sin esperar el primero), se serializan entre sí
   para no pisarse leyendo/escribiendo el historial a medias. Sesiones
   *distintas* usan locks *distintos* y jamás se esperan entre sí.

Esto está probado con threads reales (no simulado) en
`test_multiple_sessions_run_concurrently_without_cross_contamination`
(N sesiones y N nodos en paralelo, sin mezclarse) y en
`test_same_session_concurrent_requests_do_not_corrupt_history` (dos
pedidos simultáneos de la misma sesión, ninguno se pierde).

### Sauzal NO cachea respuestas — pero sí ignora reintentos

Cada `POST /infer` ejecuta el modelo de cero, tenga o no `session_id`:
no existe ningún mecanismo que devuelva una respuesta guardada en vez de
generar una nueva. Esto es intencional — un LLM normalmente debe poder
regenerar (pedir "un número al azar" dos veces no debería dar lo mismo),
y cachear por texto literal de prompt podría servir una respuesta vieja
que ya no coincide con el resumen/contexto actualizado de la sesión.

Lo que sí existe es una protección más chica y específica: si un pedido
con `session_id` es **textualmente igual al último mensaje de esa
sesión** y llegó hace menos de `DEDUPE_WINDOW_SECONDS` (10s por
defecto), se asume que es un **reintento** (timeout de red, doble click
del cliente) y se devuelve el job que ya se había creado para el
original —sin crear un job ni un mensaje nuevos— marcado con
`"deduplicated": true` en la respuesta. Pasada esa ventana, o con un
texto distinto, se ejecuta de cero con total normalidad. Ver
`_find_recent_duplicate()` en `server/main.py` y los tests
`test_immediate_retry_with_same_prompt_returns_same_job`,
`test_different_prompt_in_same_session_is_not_deduplicated` y
`test_same_prompt_outside_dedupe_window_runs_again`.

### Cómo probarlo

**A) Automático** (no necesita GPU ni Ollama — simula nodos falsos):
```powershell
cd Sauzal
pip install -r requirements-dev.txt
pytest tests/ -v
```

**B) En vivo, con un nodo real** (repite el caso del BMW Isetta):
```powershell
# 1. Crear una sesión y contar el dato
python client\infer.py --server http://IP_DEL_SERVIDOR:8000 --model gemma3:4b \
    --new-session --prompt "Mi auto favorito es un BMW Isetta"
# (anotar el session_id que imprime)

# 2. Un par de turnos de relleno, para sacar el dato de la ventana reciente
python client\infer.py --server http://IP_DEL_SERVIDOR:8000 --model gemma3:4b \
    --session SESSION_ID --prompt "Decime un numero al azar"
# (repetir 2-3 veces mas)

# 3. Preguntar por el dato -- puede ejecutarlo cualquier nodo disponible
python client\infer.py --server http://IP_DEL_SERVIDOR:8000 --model gemma3:4b \
    --session SESSION_ID --prompt "Cual es mi auto favorito?"
```
El modelo debería responder "BMW Isetta" (o similar) sin que el prompt
de este último pedido lo mencione explícitamente. Para confirmar que
realmente vino de la memoria semántica (y no de casualidad), revisar
`GET /jobs/{job_id}` de ese último pedido: el campo `prompt` va a tener
una sección "Contexto relevante de mensajes anteriores" con la frase del
BMW Isetta, y `context_semantic_hits` va a ser mayor a 0. También se
puede ver todo desde `/admin/sessions/{session_id}` en el navegador.
