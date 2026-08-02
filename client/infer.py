import argparse, json, time, requests

p=argparse.ArgumentParser()
p.add_argument("--server",required=True)
p.add_argument("--model",default="gemma3:4b")
p.add_argument("--prompt",required=True)
args=p.parse_args()
server=args.server.rstrip("/")

r=requests.post(f"{server}/infer",
    json={"model":args.model,"prompt":args.prompt},timeout=20)
r.raise_for_status()
job=r.json()
print("Job:",job["job_id"])
print("Nodo remoto:",job["assigned_node"])

while True:
    status=requests.get(f"{server}/jobs/{job['job_id']}",timeout=15).json()
    print("Estado:",status["status"])
    if status["status"]=="completed":
        data=json.loads(status["result"])
        print("\n--- RESPUESTA REMOTA ---")
        print(data["response"])
        print("\n--- MÉTRICAS ---")
        print(json.dumps({k:v for k,v in data.items() if k!="response"},indent=2))
        break
    if status["status"]=="failed":
        raise RuntimeError(status["error"])
    time.sleep(1)
