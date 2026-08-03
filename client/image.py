import argparse, base64, json, time, requests
from pathlib import Path

p=argparse.ArgumentParser(description="Pide una imagen a la red Sauzal")
p.add_argument("--server",required=True)
p.add_argument("--prompt",required=True)
p.add_argument("--model",default="flux1-schnell-fp8")
p.add_argument("--width",type=int,default=1024)
p.add_argument("--height",type=int,default=1024)
p.add_argument("--steps",type=int,default=4)
p.add_argument("--seed",type=int)
p.add_argument("--out",default="sauzal.png")
args=p.parse_args()
server=args.server.rstrip("/")

payload={"prompt":args.prompt,"width":args.width,"height":args.height,"steps":args.steps}
if args.seed is not None:
    payload["seed"]=args.seed

r=requests.post(f"{server}/infer",
    json={"model":args.model,"prompt":json.dumps(payload,ensure_ascii=False)},timeout=30)
r.raise_for_status()
job=r.json()
print("Job:",job["job_id"])
print("Nodo remoto:",job["assigned_node"])

while True:
    status=requests.get(f"{server}/jobs/{job['job_id']}",timeout=15).json()
    print("Estado:",status["status"])
    if status["status"]=="completed":
        data=json.loads(status["result"])
        out=Path(args.out)
        images=data["images"]
        for i,b64 in enumerate(images):
            path=out if len(images)==1 else out.with_name(f"{out.stem}_{i+1}{out.suffix}")
            path.write_bytes(base64.b64decode(b64))
            print("Guardada:",path)
        print("\n--- MÉTRICAS ---")
        print(json.dumps({k:v for k,v in data.items() if k!="images"},indent=2,ensure_ascii=False))
        break
    if status["status"]=="failed":
        raise RuntimeError(status["error"])
    time.sleep(1)
