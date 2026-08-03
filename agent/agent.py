from __future__ import annotations
import argparse, json, platform, socket, subprocess, time
from pathlib import Path
import requests

import comfy

CONFIG=Path(__file__).with_name("agent_config.json")
OLLAMA="http://127.0.0.1:11434"

def cmd(args):
    try:
        return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=10).strip()
    except Exception:
        return ""

def ollama_models():
    try:
        models=requests.get(f"{OLLAMA}/api/tags",timeout=5).json().get("models",[])
        return [m.get("name") for m in models]
    except Exception:
        return []

def capabilities():
    gpu=cmd(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"])
    text_models=ollama_models()
    image_models=comfy.models()
    return json.dumps({
        "hostname":socket.gethostname(),
        "os":platform.system(),
        "gpu":gpu.splitlines() if gpu else [],
        "ollama_models":text_models,
        "image_models":image_models,
        "services":{"ollama":bool(text_models),"comfyui":bool(image_models)}
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
        "type":"text",
        "response":b.get("response",""),
        "prompt_tokens":b.get("prompt_eval_count"),
        "output_tokens":b.get("eval_count"),
        "total_duration_ns":b.get("total_duration"),
        "load_duration_ns":b.get("load_duration"),
        "eval_duration_ns":b.get("eval_duration")
    },ensure_ascii=False)

def execute(job):
    """Elige el backend segun el modelo pedido."""
    model=job["model"]
    if comfy.is_image_model(model):
        return comfy.generate(model,job["prompt"])
    return infer(model,job["prompt"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--server",required=True)
    p.add_argument("--name",default=socket.gethostname())
    p.add_argument("--register",action="store_true")
    args=p.parse_args()

    has_ollama=bool(ollama_models())
    has_comfy=comfy.available()
    if not has_ollama and not has_comfy:
        raise SystemExit(
            "Este nodo no tiene ningun backend disponible.\n"
            f"  Ollama en {OLLAMA}: no responde o sin modelos\n"
            f"  ComfyUI en {comfy.COMFY}: no responde o falta {comfy.WORKFLOW.name}"
        )

    config=register(args.server,args.name) if args.register or not CONFIG.exists() else json.loads(CONFIG.read_text())
    config["server"]=args.server.rstrip("/")
    auth={"node_id":config["node_id"],"token":config["token"]}

    backends=[n for n,ok in (("ollama",has_ollama),("comfyui",has_comfy)) if ok]
    print(f"Agente Sauzal conectado: {config['node_id']}")
    print(f"Backends activos: {', '.join(backends)}")

    while True:
        try:
            requests.post(f"{config['server']}/nodes/heartbeat",
              json={**auth,"status":"available","capabilities":capabilities()},timeout=15).raise_for_status()
            r=requests.post(f"{config['server']}/agent/pull",json=auth,timeout=20)
            r.raise_for_status()
            job=r.json()["job"]
            if job:
                kind="imagen" if comfy.is_image_model(job["model"]) else "texto"
                print(f"Ejecutando {job['job_id']} [{kind}] con {job['model']}")
                try:
                    output=execute(job)
                    payload={**auth,"success":True,"result":output}
                except Exception as exc:
                    print("  fallo:",repr(exc))
                    payload={**auth,"success":False,"error":repr(exc)}
                requests.post(
                    f"{config['server']}/jobs/{job['job_id']}/result",
                    json=payload,timeout=120
                ).raise_for_status()
        except requests.RequestException as exc:
            print("Error:",exc)
        time.sleep(2)

if __name__=="__main__":
    main()
