"""
Sauzal · Agente (corre en cada nodo con GPU)
===============================================

Este script se ejecuta en cada maquina que aporta computo a la red: puede
ser tu PC con una placa de video, o un Pod de RunPod con el Dockerfile de
este repo (ComfyUI + FLUX + Ollama ya instalados).

Que hace, en resumen:
  1. Detecta que backends de inferencia tiene disponibles localmente:
     - Ollama (texto), escuchando en http://127.0.0.1:11434
     - ComfyUI (imagenes, modulo `comfy.py`), escuchando en :8188
  2. Se registra contra el servidor (control plane) la primera vez, y
     guarda las credenciales (node_id + token) en agent_config.json para
     no tener que volver a registrarse en cada arranque.
  3. Entra en un bucle infinito: cada 2 segundos manda un heartbeat
     ("sigo vivo, esto es lo que puedo hacer") y pregunta si tiene un
     trabajo asignado. Si lo tiene, lo ejecuta con el backend que
     corresponda segun el modelo pedido, y devuelve el resultado.

Importante: el agente SIEMPRE inicia la conexion hacia el servidor
(nunca al reves). Esto permite correrlo detras de NAT/firewall sin abrir
ningun puerto de entrada — solo necesita salida a internet (o a la red
local, si el servidor esta en la misma LAN).

Uso:
    python agent.py --server http://192.168.1.78:8000 --name mi-pc --register

    --server    URL del control plane (obligatorio)
    --name      nombre legible para identificar el nodo (default: hostname)
    --register  fuerza un registro nuevo aunque ya exista agent_config.json
                (util si se borro el nodo del lado del servidor, o si se
                quiere renombrar)
"""

from __future__ import annotations
import argparse, json, platform, socket, subprocess, time
from pathlib import Path
import requests

import comfy

# Archivo donde se persisten las credenciales despues del primer registro.
CONFIG=Path(__file__).with_name("agent_config.json")
# Ollama corre localmente en la misma maquina que este agente.
OLLAMA="http://127.0.0.1:11434"


def cmd(args):
    """
    Ejecuta un comando de sistema y devuelve su salida como texto, o
    string vacio si falla por cualquier motivo (comando no encontrado,
    timeout, etc). Se usa para `nvidia-smi` sin que la falta de GPU
    NVIDIA (ej: en una maquina con AMD) rompa el resto del agente.
    """
    try:
        return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=10).strip()
    except Exception:
        return ""


def ollama_models():
    """
    Devuelve la lista de nombres de modelos que Ollama tiene descargados
    localmente (ej: ["gemma3:4b"]). Lista vacia si Ollama no esta
    corriendo o no responde — eso es lo que usa el resto del codigo para
    decidir si el backend "ollama" esta disponible en este nodo.
    """
    try:
        models=requests.get(f"{OLLAMA}/api/tags",timeout=5).json().get("models",[])
        return [m.get("name") for m in models]
    except Exception:
        return []


def capabilities():
    """
    Arma el JSON de "capabilities" que el nodo le manda al servidor en el
    registro y en cada heartbeat. Sirve para que el servidor (y quien
    mire /nodes) sepa que puede hacer esta maquina: que GPU tiene, que
    modelos de texto (Ollama) y de imagen (ComfyUI) tiene disponibles.

    El campo "services" es un resumen booleano (bool(lista) es True si la
    lista no esta vacia) que usa el propio agente para decidir que
    backends anunciar como activos.
    """
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
    """
    Da de alta este nodo contra el servidor (POST /nodes/register) y
    guarda la respuesta (node_id, token) mas la URL del servidor en
    agent_config.json, para poder reusarla en el proximo arranque sin
    tener que registrarse de nuevo (cada registro crea un nodo NUEVO del
    lado del servidor, asi que no conviene llamarlo en cada inicio).
    """
    r=requests.post(f"{server}/nodes/register",
        json={"name":name,"capabilities":capabilities()},timeout=20)
    r.raise_for_status()
    data={"server":server.rstrip("/"),**r.json()}
    CONFIG.write_text(json.dumps(data,indent=2),encoding="utf-8")
    return data


def ensure_model(model):
    """
    Antes de correr una inferencia de texto, verifica que Ollama ya tenga
    el modelo pedido descargado. Si no lo tiene, lo baja con
    /api/pull (puede tardar varios minutos la primera vez, de ahi el
    timeout largo de una hora).
    """
    tags=requests.get(f"{OLLAMA}/api/tags",timeout=10)
    tags.raise_for_status()
    names={m["name"] for m in tags.json().get("models",[])}
    if model not in names:
        print(f"Descargando {model}...")
        r=requests.post(f"{OLLAMA}/api/pull",
            json={"model":model,"stream":False},timeout=3600)
        r.raise_for_status()

def infer(model,prompt):
    """
    Ejecuta una inferencia de TEXTO contra Ollama (backend local) y
    devuelve un JSON (en texto) con la respuesta y metricas de
    rendimiento (tokens de entrada/salida, duraciones en nanosegundos)
    que Ollama ya calcula y devuelve el mismo.

    Este string es lo que termina guardado tal cual en la columna
    `result` de la tabla `jobs` del servidor.
    """
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
    """
    Punto central de despacho: mira el modelo pedido en el job y decide
    si el trabajo es de imagen (delegado a comfy.generate) o de texto
    (delegado a infer, que habla con Ollama). `comfy.is_image_model`
    reconoce el modelo por prefijo (flux/sdxl/sauzal-image).
    """
    model=job["model"]
    if comfy.is_image_model(model):
        return comfy.generate(model,job["prompt"])
    return infer(model,job["prompt"])


def main():
    """
    Punto de entrada del script. Hace, en orden:

    1. Parsea los argumentos de linea de comandos.
    2. Chequea que backends estan disponibles en esta maquina (Ollama
       con al menos un modelo, y/o ComfyUI con el workflow de FLUX). Si
       no hay NINGUNO, corta la ejecucion — no tiene sentido ser un nodo
       sin nada que ofrecer.
    3. Se registra contra el servidor (si es la primera vez o si se paso
       --register), o reusa las credenciales guardadas.
    4. Entra en el bucle principal: cada 2 segundos, manda un heartbeat
       y pregunta si tiene trabajo (POST /agent/pull). Si le llega un
       job, lo ejecuta con `execute()` y reporta el resultado (o el
       error, capturando cualquier excepcion para que un fallo puntual
       no tire abajo el agente entero) con POST /jobs/{id}/result.

    Los errores de red (servidor caido, timeout) se atrapan y solo se
    imprimen — el bucle sigue reintentando indefinidamente cada 2s, asi
    que el agente se recupera solo si el servidor vuelve a estar arriba.
    """
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
            # Heartbeat: avisa "sigo vivo" y refresca las capabilities
            # (por si en el medio se bajo/borro un modelo de Ollama).
            requests.post(f"{config['server']}/nodes/heartbeat",
              json={**auth,"status":"available","capabilities":capabilities()},timeout=15).raise_for_status()
            # Pregunta si hay un trabajo en cola para este nodo puntual.
            r=requests.post(f"{config['server']}/agent/pull",json=auth,timeout=20)
            r.raise_for_status()
            job=r.json()["job"]
            if job:
                kind="imagen" if comfy.is_image_model(job["model"]) else "texto"
                print(f"Ejecutando {job['job_id']} [{kind}] con {job['model']}")
                try:
                    output=execute(job)
                    # `output` es siempre un JSON serializado (tanto infer()
                    # como comfy.generate() devuelven json.dumps(...)). Los
                    # jobs de texto traen prompt_tokens/output_tokens; los de
                    # imagen no tienen esas claves, .get() devuelve None y
                    # el servidor los guarda como NULL.
                    parsed=json.loads(output)
                    payload={
                        **auth,"success":True,"result":output,
                        "prompt_tokens":parsed.get("prompt_tokens"),
                        "output_tokens":parsed.get("output_tokens"),
                    }
                except Exception as exc:
                    # Cualquier fallo durante la ejecucion (ComfyUI caido,
                    # modelo invalido, timeout, etc.) se reporta como
                    # resultado fallido en vez de tirar abajo el agente.
                    print("  fallo:",repr(exc))
                    payload={**auth,"success":False,"error":repr(exc)}
                requests.post(
                    f"{config['server']}/jobs/{job['job_id']}/result",
                    json=payload,timeout=120
                ).raise_for_status()
        except requests.RequestException as exc:
            # Problema de red hablando con el servidor (caido, timeout,
            # DNS, etc.) — se loguea y se reintenta en el proximo ciclo.
            print("Error:",exc)
        time.sleep(2)


if __name__=="__main__":
    main()
