"""
Sauzal · Cliente de inferencia
================================

Script de linea de comandos para pedirle un trabajo (texto o imagen) a la
red Sauzal y esperar el resultado. Es el rol de "usuario final": no
registra nada, no tiene credenciales, solo habla con el servidor
(control plane) via HTTP.

Flujo:
  1. POST /infer al servidor, con el modelo, el prompt, y opcionalmente
     el nodo destino. Tambien se manda un identificador de "quien pide"
     (--client, por defecto el hostname de esta PC) mas el sistema
     operativo y procesador de esta maquina (via el modulo `platform`) --
     son datos informativos, no una identidad autenticada (cualquiera
     puede correr este script con datos falsos). La IP real y si el
     pedido vino de un navegador o de un script, en cambio, los calcula
     el SERVIDOR solo (de la conexion HTTP y el header User-Agent), sin
     que este script tenga que declarar nada. El servidor elige (o
     valida) que nodo remoto va a procesar el pedido y devuelve un job_id.
  2. Polling a GET /jobs/{job_id} cada 1 segundo hasta que el status sea
     "completed" (se imprime la respuesta y las metricas) o "failed" (se
     lanza una excepcion con el mensaje de error del agente remoto).

Ejemplos de uso:
    python infer.py --server http://192.168.1.78:8000 \\
        --model gemma3:4b --prompt "Hola, como estas?"

    # Forzando que lo procese un nodo puntual (por nombre):
    python infer.py --server http://192.168.1.78:8000 \\
        --model gemma3:4b --node pc-elmar --prompt "Hola"

    # Conversacion con memoria (ver server/memory.py): primero se crea la
    # sesion una vez, y ese session_id se reutiliza en cada mensaje
    # siguiente -- puede ir a un nodo distinto cada vez, la continuidad
    # la mantiene el servidor, no hace falta que el cliente reenvie nada
    # del historial:
    python infer.py --server http://192.168.1.78:8000 --new-session \\
        --model gemma3:4b --prompt "Mi auto favorito es un BMW Isetta"
    # (imprime el session_id -- se reusa en la siguiente llamada)
    python infer.py --server http://192.168.1.78:8000 --session SESSION_ID \\
        --model gemma3:4b --prompt "Cual es mi auto favorito?"

Requiere solo la libreria `requests` (pip install requests) — no necesita
el resto del repo Sauzal para funcionar, por eso es facil de copiar a
cualquier PC que solo quiera consumir la red, sin ser un nodo.
"""

import argparse, json, platform, socket, sys, time, requests

# La consola de Windows por defecto usa cp1252, que no puede imprimir
# emojis ni varios acentos que los modelos devuelven con total libertad.
# Se fuerza UTF-8 en la salida estandar para no crashear al mostrar la
# respuesta (esto no afecta a Linux/Mac, que ya usan UTF-8 de por si).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p=argparse.ArgumentParser()
p.add_argument("--server",required=True,help="URL del control plane, ej: http://192.168.1.78:8000")
p.add_argument("--model",default="gemma3:4b",help="Modelo de texto (Ollama) o de imagen (ej: flux1-schnell-fp8)")
p.add_argument("--prompt",required=True,help="Texto del pedido")
p.add_argument("--node",default=None,help="Nombre o node_id del nodo destino (opcional; si se omite, el servidor elige uno disponible)")
p.add_argument("--client",default=socket.gethostname(),help="Identificador de quien pide el trabajo (default: hostname de esta PC). Dato libre, no autenticado.")
p.add_argument("--session",default=None,help="session_id de una conversacion existente (ver --new-session). Sin esto, el pedido es stateless (sin memoria), igual que siempre.")
p.add_argument("--new-session",action="store_true",help="Crea una sesion de conversacion nueva antes de mandar este pedido, y la usa. Imprime el session_id para reusarlo en el proximo mensaje.")
args=p.parse_args()
server=args.server.rstrip("/")

session_id=args.session
if args.new_session:
    r=requests.post(f"{server}/sessions",json={},timeout=15)
    r.raise_for_status()
    session_id=r.json()["session_id"]
    print("Sesion nueva:",session_id)

# SO y procesador de esta PC, para que el panel /admin del servidor los
# pueda mostrar en el detalle del cliente. Igual que --client, son datos
# que este script declara por su cuenta, no verificados por el servidor.
client_os=f"{platform.system()} {platform.release()}".strip()
client_processor=platform.processor() or platform.machine() or None

# 1. Pedimos el trabajo. El servidor responde apenas lo encola, sin
#    esperar a que termine (por eso hace falta el polling de abajo).
r=requests.post(f"{server}/infer",
    json={
        "model":args.model,"prompt":args.prompt,"node":args.node,"client":args.client,
        "client_os":client_os,"client_processor":client_processor,"session_id":session_id,
    },timeout=20)
r.raise_for_status()
job=r.json()
print("Job:",job["job_id"])
print("Nodo remoto:",job["assigned_node"])

# 2. Polling: preguntamos el estado una vez por segundo hasta terminar.
while True:
    status=requests.get(f"{server}/jobs/{job['job_id']}",timeout=15).json()
    print("Estado:",status["status"])
    if status["status"]=="completed":
        # El campo "result" viaja como JSON serializado en un string
        # (columna TEXT en SQLite), por eso hay que parsearlo de nuevo.
        data=json.loads(status["result"])
        print(f"\n--- RESPUESTA REMOTA (ejecutado en: {status.get('node_name') or status['assigned_node']}) ---")
        print(data["response"])
        print("\n--- MÉTRICAS ---")
        # Se imprime todo menos "response" (que ya se mostro arriba):
        # tokens, duraciones, etc. Varian segun sea un job de texto o imagen.
        print(json.dumps({k:v for k,v in data.items() if k!="response"},indent=2))
        break
    if status["status"]=="failed":
        raise RuntimeError(status["error"])
    time.sleep(1)
