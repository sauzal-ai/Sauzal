from __future__ import annotations
import argparse, json, platform, socket, subprocess, time
from pathlib import Path
import requests

CONFIG=Path(__file__).with_name("agent_config.json")
OLLAMA="http://127.0.0.1:11434"

def cmd(args):
    try:
        return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=10).strip()
    except Exception:
        return ""

def capabilities():
    gpu=cmd(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"])
    try:
        models=requests.get(f"{OLLAMA}/api/tags",timeout=5).json().get("models",[])
        names=[m.get("name") for m in models]
    except Exception:
        names=[]
    return json.dumps({
        "hostname":socket.gethostname(),
        "os":platform.system(),
        "gpu":gpu.splitlines() if gpu else [],
        "ollama_models":names
    },ensure_ascii=False)

def register(server,name):
    r=requests.post(f"{server}/nodes/register",
        json={"name":name,"capabilities":capabilities()},timeout=20)
    r.raise_for_status()
    data={"server":server.rstrip("/"),**r.json()}
    CONFIG.write_text(json.dumps(data,indent=2),encoding="utf-8")
    return data

def ensure_model(model):
    tags=requests.get(f"{OLLAMA}/api/tags",timeout=10)
    tags.raise_for_status()
    names={m["name"] for m in tags.json().get("models",[])}
    if model not in names:
        print(f"Descargando {model}...")
        r=requests.post(f"{OLLAMA}/api/pull",
            json={"model":model,"stream":False},timeout=3600)
        r.raise_for_status()

def infer(model,prompt):
    ensure_model(model)
    r=requests.post(f"{OLLAMA}/api/generate",
        json={"model":model,"prompt":prompt,"stream":False},timeout=600)
    r.raise_for_status()
    b=r.json()
    return json.dumps({
        "response":b.get("response",""),
        "prompt_tokens":b.get("prompt_eval_count"),
        "output_tokens":b.get("eval_count"),
        "total_duration_ns":b.get("total_duration"),
        "load_duration_ns":b.get("load_duration"),
        "eval_duration_ns":b.get("eval_duration")
    },ensure_ascii=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--server",required=True)
    p.add_argument("--name",default=socket.gethostname())
    p.add_argument("--register",action="store_true")
    args=p.parse_args()

    requests.get(f"{OLLAMA}/api/tags",timeout=10).raise_for_status()
    config=register(args.server,args.name) if args.register or not CONFIG.exists() else json.loads(CONFIG.read_text())
    config["server"]=args.server.rstrip("/")
    auth={"node_id":config["node_id"],"token":config["token"]}
    print(f"Agente Sauzal conectado: {config['node_id']}")

    while True:
        try:
            requests.post(f"{config['server']}/nodes/heartbeat",
              json={**auth,"status":"available","capabilities":capabilities()},timeout=15).raise_for_status()
            r=requests.post(f"{config['server']}/agent/pull",json=auth,timeout=20)
            r.raise_for_status()
            job=r.json()["job"]
            if job:
                print(f"Ejecutando {job['job_id']} con {job['model']}")
                try:
                    output=infer(job["model"],job["prompt"])
                    payload={**auth,"success":True,"result":output}
                except Exception as exc:
                    payload={**auth,"success":False,"error":repr(exc)}
                requests.post(
                    f"{config['server']}/jobs/{job['job_id']}/result",
                    json=payload,timeout=30
                ).raise_for_status()
        except requests.RequestException as exc:
            print("Error:",exc)
        time.sleep(2)

if __name__=="__main__":
    main()
