"""
Sauzal · Cliente de tareas compuestas (Task Decomposition + Aggregation)
===========================================================================

Script de linea de comandos para probar `POST /tasks`: arma una tarea
compuesta por varias subtareas, la manda al servidor, y hace polling de
`GET /tasks/{id}` hasta que termine, mostrando al final el resultado
agregado y (via `GET /tasks/{id}/subtasks`) que nodo ejecuto cada parte.

No hace falta GPU ni Ollama para PROBAR que el mecanismo de reparto y
agregacion funciona (ver tests/test_tasks.py, que lo hace con nodos
falsos) -- este script es para verlo correr con nodos reales.

Dos modos de uso, cubriendo los dos casos minimos del diseño:

  1. Batch (subtareas independientes, se junta TODO):
     python task.py --server http://192.168.1.78:8000 --mode batch \\
         --model gemma3:4b \\
         --prompt "Parte 1: escribi un titulo" \\
         --prompt "Parte 2: escribi un resumen" \\
         --prompt "Parte 3: escribi una conclusion"

  2. Fan-out / primera respuesta valida (se manda LA MISMA pregunta a
     varios nodos, se usa la primera que responda y se cancelan las
     demas):
     python task.py --server http://192.168.1.78:8000 --mode fan_out \\
         --model gemma3:4b --prompt "Decime un chiste corto" --replicas 3

Requiere solo `requests` -- igual criterio que client/infer.py.
"""

import argparse, json, sys, time, requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--server", required=True, help="URL del control plane, ej: http://192.168.1.78:8000")
p.add_argument("--mode", required=True,
    choices=["batch", "fan_out", "pipeline", "map_reduce", "first_success", "consensus"],
    help="Como se ejecutan y agregan las subtareas (ver docs/ARQUITECTURA.md seccion 7)")
p.add_argument("--model", default="gemma3:4b")
p.add_argument("--prompt", action="append", required=True,
    help="Una subtarea por cada --prompt (en batch/pipeline); en fan_out/consensus se usa el PRIMERO junto con --replicas")
p.add_argument("--replicas", type=int, default=1,
    help="Para fan_out/first_success/consensus: cuantas subtareas identicas mandar (default 1)")
p.add_argument("--aggregation", default=None, help="Pisa la estrategia de agregacion por defecto del modo")
p.add_argument("--session", default=None, help="session_id de una conversacion existente (opcional)")
p.add_argument("--max-parallel", type=int, default=None, help="Politica: maximo de subtareas corriendo a la vez")
p.add_argument("--max-retries", type=int, default=None, help="Politica: reintentos por subtarea fallida")
args = p.parse_args()
server = args.server.rstrip("/")

if args.mode in ("fan_out", "first_success", "consensus"):
    subtasks = [
        {"key": f"try{i}", "model": args.model, "prompt": args.prompt[0]}
        for i in range(max(1, args.replicas))
    ]
else:
    subtasks = [
        {"key": f"part{i}", "model": args.model, "prompt": prompt}
        for i, prompt in enumerate(args.prompt)
    ]

policy = {}
if args.max_parallel is not None:
    policy["max_parallel"] = args.max_parallel
if args.max_retries is not None:
    policy["max_retries"] = args.max_retries

# 1. Creamos la tarea. El servidor arma todas las subtareas de una y
#    responde de inmediato (no espera a que ninguna termine).
r = requests.post(f"{server}/tasks", json={
    "mode": args.mode, "subtasks": subtasks, "session_id": args.session,
    "aggregation": args.aggregation, "policy": policy or None,
}, timeout=20)
r.raise_for_status()
created = r.json()
task_id = created["task_id"]
print("Task:", task_id)
print("Subtareas:", json.dumps(created["subtasks"], indent=2))

# 2. Polling: igual que client/infer.py, pero de /tasks en vez de /jobs.
while True:
    status = requests.get(f"{server}/tasks/{task_id}", timeout=15).json()
    print("Estado:", status["status"])
    if status["status"] in ("completed", "failed", "cancelled"):
        subtasks_status = requests.get(f"{server}/tasks/{task_id}/subtasks", timeout=15).json()["subtasks"]
        print("\n--- TRAZABILIDAD (que nodo ejecuto cada subtarea) ---")
        for s in subtasks_status:
            node_label = s.get("node_name") or s["assigned_node"] or "-"
            print(f"  {s['subtask_key'] or s['job_id'][:8]}: {s['status']} (nodo: {node_label})")

        print("\n--- METRICAS ---")
        print(json.dumps(status["metrics"], indent=2))

        if status["status"] == "completed":
            print("\n--- RESULTADO AGREGADO ---")
            print(json.dumps(json.loads(status["result"]), indent=2, ensure_ascii=False))
        else:
            print("\n--- ERROR ---")
            print(status.get("error"))
        break
    time.sleep(1)
