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
     (--client, por defecto el hostname de esta PC) que el servidor
     guarda junto con el job -- es un dato informativo, no una identidad
     autenticada (cualquiera puede mandar el --client que quiera). El
     servidor elige (o valida) que nodo remoto va a procesar el pedido y
     devuelve un job_id.
  2. Polling a GET /jobs/{job_id} cada 1 segundo hasta que el status sea
     "completed" (se imprime la respuesta y las metricas) o "failed" (se
     lanza una excepcion con el mensaje de error del agente remoto).

Ejemplos de uso:
    python infer.py --server http://192.168.1.78:8000 \\
        --model gemma3:4b --prompt "Hola, como estas?"

    # Forzando que lo procese un nodo puntual (por nombre):
    python infer.py --server http://192.168.1.78:8000 \\
        --model gemma3:4b --node pc-elmar --prompt "Hola"

Requiere solo la libreria `requests` (pip install requests) — no necesita
el resto del repo Sauzal para funcionar, por eso es facil de copiar a
cualquier PC que solo quiera consumir la red, sin ser un nodo.
"""

import argparse, json, socket, time, requests

p=argparse.ArgumentParser()
p.add_argument("--server",required=True,help="URL del control plane, ej: http://192.168.1.78:8000")
p.add_argument("--model",default="gemma3:4b",help="Modelo de texto (Ollama) o de imagen (ej: flux1-schnell-fp8)")
p.add_argument("--prompt",required=True,help="Texto del pedido")
p.add_argument("--node",default=None,help="Nombre o node_id del nodo destino (opcional; si se omite, el servidor elige uno disponible)")
p.add_argument("--client",default=socket.gethostname(),help="Identificador de quien pide el trabajo (default: hostname de esta PC). Dato libre, no autenticado.")
args=p.parse_args()
server=args.server.rstrip("/")

# 1. Pedimos el trabajo. El servidor responde apenas lo encola, sin
#    esperar a que termine (por eso hace falta el polling de abajo).
r=requests.post(f"{server}/infer",
    json={"model":args.model,"prompt":args.prompt,"node":args.node,"client":args.client},timeout=20)
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
