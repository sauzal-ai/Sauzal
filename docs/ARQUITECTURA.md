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
- `GET /admin` — panel web (HTML) para ver todos los nodos y trabajos en
  vivo, y hacer cambios manuales: eliminar un nodo, forzarlo de vuelta a
  "available" si quedó colgado en "busy", o eliminar un trabajo. Sin
  autenticación (igual que el resto del servidor) — pensado para uso en
  LAN/desarrollo, no para exponer a internet sin agregarle protección.

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
```
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
| `capabilities` | TEXT (JSON serializado) | Servidor, con el dato que manda el agente | Al registrar, y refrescado en cada heartbeat si el agente manda un valor nuevo. Contiene: `hostname`, `os`, `gpu` (salida de `nvidia-smi`, vacío si no es NVIDIA), `ollama_models` (lista de modelos de texto disponibles), `image_models` (lista de modelos de imagen disponibles) y `services` (booleanos `ollama`/`comfyui`, resumen de si hay algo en las dos listas anteriores). |

### Tabla `jobs` — un registro por cada pedido de inferencia

| Columna | Tipo | Quién la escribe | Cuándo |
|---|---|---|---|
| `job_id` | TEXT (PK) | Servidor | Al crear el pedido (`POST /infer`), genera un `uuid4()` nuevo. |
| `model` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido. Es el nombre de modelo que decide, del lado del agente, si el trabajo va a Ollama (texto) o a ComfyUI (imagen prefijo `flux`/`sdxl`/`sauzal-image`). |
| `prompt` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido. Texto plano, o un JSON con parámetros si es un pedido de imagen (ver `agent/comfy.py::parse_request`). |
| `status` | TEXT (`queued`→`running`→`completed`\|`failed`) | Servidor | `queued` al crear (`POST /infer`); `running` cuando el nodo asignado hace *pull* exitoso del trabajo (`POST /agent/pull`); `completed` o `failed` cuando el nodo entrega el resultado (`POST /jobs/{id}/result`, según el campo `success` que mande). |
| `assigned_node` | TEXT (FK lógica a `nodes.node_id`) | Servidor | Al crear el pedido: es el `node_id` que el servidor eligió (el nodo disponible visto más recientemente, o el pedido explícitamente por el cliente vía el campo `node`). No cambia después — si ese nodo se cae a mitad de camino, el job queda huérfano (no hay reintento ni reasignación automática). |
| `client` | TEXT | Servidor, con el dato que manda el cliente | Al crear el pedido (`POST /infer`, campo `client`). Identifica "quién lo pidió" — por defecto, `client/infer.py` manda su propio hostname. **No es una identidad autenticada**: cualquiera puede mandar el valor que quiera, es solo informativo (mismo criterio de confianza que el `name` de los nodos). |
| `result` | TEXT (JSON serializado) | El AGENTE, vía el servidor | Al entregar el resultado (`POST /jobs/{id}/result`), solo si `success=true`. El JSON lo arma el agente: `agent.py::infer()` para texto (con `response`, tokens, duraciones) o `agent/comfy.py::generate()` para imagen (con las imágenes en base64, dimensiones, seed, etc). |
| `error` | TEXT | El AGENTE, vía el servidor | Al entregar el resultado, solo si `success=false`. Es el `repr()` de la excepción que capturó `agent.py` al ejecutar el trabajo. |
| `prompt_tokens` | INTEGER | El AGENTE, vía el servidor | Al entregar el resultado. Solo en jobs de **texto**: `agent.py` extrae `prompt_tokens` del JSON que ya arma `infer()` (viene de Ollama, campo `prompt_eval_count`). En jobs de **imagen** queda `NULL` (ComfyUI no tiene concepto de tokens). |
| `output_tokens` | INTEGER | El AGENTE, vía el servidor | Ídem `prompt_tokens`, pero de salida (`eval_count` de Ollama). `NULL` en jobs de imagen. |
| `created_at` | REAL (timestamp epoch) | Servidor | Al crear el pedido. |
| `updated_at` | REAL (timestamp epoch) | Servidor | Se actualiza cuando el job pasa a `running` (pull) y de nuevo cuando pasa a `completed`/`failed` (result). |

> Estas tres columnas (`client`, `prompt_tokens`, `output_tokens`) se
> agregaron después de la versión inicial del esquema. El servidor migra
> solo cualquier `sauzal.db` vieja al arrancar (`ALTER TABLE` si faltan),
> así que no hace falta borrar la base para actualizar.

### Dato que NO vive en la base: `agent/agent_config.json`

Cada nodo guarda localmente (no en el servidor) un archivo con
`{"server": "...", "node_id": "...", "token": "..."}`. Es lo que le
permite al agente reconectarse tras un reinicio sin volver a registrarse
como un nodo nuevo. Si se borra este archivo (o se corre con
`--register`), el agente se registra de cero y aparece como una fila
NUEVA en la tabla `nodes` — por eso en las pruebas de este proyecto a
veces se ven nodos viejos duplicados y `offline` en `GET /nodes`: son
registros de sesiones anteriores que nunca se limpian solos.
