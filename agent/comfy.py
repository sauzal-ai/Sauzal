"""
Sauzal · backend de imagen (ComfyUI)

Modulo del agente. No expone puertos propios: el agente (agent.py) lo
importa y lo llama cuando el servidor le asigna un job cuyo modelo es de
imagen (ver `is_image_model`). Habla con ComfyUI, que corre localmente en
esta misma maquina en el puerto 8188 (arrancado por entrypoint.sh en el
Pod, o manualmente si se prueba en una PC).

Los jobs de imagen viajan por la misma tabla que los de texto. El campo
prompt admite dos formas:

    texto plano                -> se usa tal cual, con los valores por defecto
    {"prompt":"...", "width":1024, "height":1024, "steps":4, "seed":42}

El resultado es un JSON con el PNG en base64, para que entre en la
columna result sin cambiar el esquema de la base.

Como funciona la generacion, paso a paso (funcion `generate`):
    1. `parse_request`  -> normaliza el prompt (texto o JSON) a un dict
                           con width/height/steps/seed/batch_size validos.
    2. `build_workflow` -> toma la plantilla de workflow exportada de
                           ComfyUI (workflow_flux_schnell.json) y le
                           inyecta esos valores en los nodos correctos.
    3. `_queue`         -> manda el workflow armado a la API de ComfyUI
                           (POST /prompt), que lo encola y devuelve un
                           prompt_id para hacerle seguimiento.
    4. `_wait`          -> hace polling a GET /history/{prompt_id} hasta
                           que ComfyUI marque la generacion como
                           terminada (o falle, o se pase el timeout).
    5. `_fetch`         -> descarga el/los PNG generados con GET /view.
    6. Se devuelve todo como JSON con las imagenes en base64.
"""

from __future__ import annotations

import base64
import json
import random
import time
import uuid
from pathlib import Path

import requests

# ComfyUI corre local, en el mismo host/contenedor que este agente.
COMFY = "http://127.0.0.1:8188"
# Plantilla de workflow exportada desde la UI de ComfyUI (Workflow -> Export API).
WORKFLOW = Path(__file__).with_name("workflow_flux_schnell.json")

# IDs de nodo del workflow exportado con Workflow -> Export (API).
# Si reexportas despues de agregar o borrar nodos, revisa que sigan siendo estos.
NODE_POSITIVE = "6"   # nodo CLIPTextEncode: donde va el prompt positivo
NODE_LATENT = "27"    # nodo EmptyLatentImage: tamaño/batch de la imagen
NODE_SAMPLER = "31"   # nodo KSampler (o similar): seed y steps
NODE_SAVE = "9"       # nodo SaveImage: prefijo del nombre de archivo

MODEL_NAME = "flux1-schnell-fp8"
TIMEOUT = 600   # segundos maximos a esperar que termine una generacion
POLL = 0.5      # segundos entre cada chequeo de /history mientras se espera


def available() -> bool:
    """
    True si este nodo puede generar imagenes: necesita que exista el
    archivo de workflow Y que ComfyUI este corriendo y responda. Se usa
    en agent.py para decidir si anunciar el backend "comfyui" como activo
    -- se llama en CADA heartbeat (cada ~2s), asi que el timeout se
    mantiene corto a proposito: en un nodo sin ComfyUI corriendo, conectar
    a un puerto local sin nada escuchando tarda ~2s en fallar en Windows
    con un timeout de 5s (se agota antes por otro motivo del lado del OS,
    no por el timeout en si), lo que duplicaba el intervalo real del
    heartbeat. Con 1s alcanza de sobra para una llamada local cuando
    ComfyUI SI esta corriendo (responde en milisegundos, medido en la
    practica en <10ms via loopback).
    """
    if not WORKFLOW.exists():
        return False
    try:
        requests.get(f"{COMFY}/system_stats", timeout=0.5).raise_for_status()
        return True
    except requests.RequestException:
        return False


def models() -> list[str]:
    """Lista de modelos de imagen disponibles (hoy, siempre uno solo:
    MODEL_NAME) si `available()` es True, o vacia si no."""
    return [MODEL_NAME] if available() else []


def is_image_model(model: str) -> bool:
    """
    Heuristica para decidir, a partir del nombre de modelo pedido en el
    job, si hay que enrutarlo a ComfyUI (imagen) en vez de a Ollama
    (texto). Reconoce por prefijo: flux*, sdxl*, sauzal-image*.
    """
    m = (model or "").lower()
    return m.startswith("flux") or m.startswith("sdxl") or m.startswith("sauzal-image")


def busy() -> bool:
    """
    True si ComfyUI ya tiene algo corriendo o encolado. ComfyUI procesa
    de a un workflow por vez, asi que esto sirve para evitar mandar un
    segundo pedido "a ciegas" mientras el anterior todavia esta en curso
    (no se usa actualmente en el flujo principal de agent.py, queda
    disponible como utilidad).
    """
    try:
        q = requests.get(f"{COMFY}/queue", timeout=5).json()
        return bool(q.get("queue_running")) or bool(q.get("queue_pending"))
    except requests.RequestException:
        return False


def parse_request(prompt: str) -> dict:
    """
    El prompt puede ser texto plano o un JSON con parametros. Devuelve
    siempre un dict con valores validos y acotados a rangos seguros:

    - width/height: redondeados a multiplos de 16 (FLUX trabaja en
      latentes de 16px; un valor no alineado da errores opacos de
      ComfyUI) y limitados a [256, 2048].
    - steps: limitado a [1, 16] (FLUX-schnell esta pensado para pocos
      pasos; mas de eso no mejora la calidad y solo hace mas lento).
    - batch_size: limitado a [1, 4].

    Si el JSON es invalido o no trae "prompt", se cae de nuevo a
    interpretar todo el string original como prompt de texto plano.
    """
    req = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "seed": None,
        "batch_size": 1,
    }

    raw = (prompt or "").strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return req
        if isinstance(data, dict) and "prompt" in data:
            req["prompt"] = str(data["prompt"])
            for key in ("width", "height", "steps", "seed", "batch_size"):
                if data.get(key) is not None:
                    req[key] = int(data[key])

    # FLUX trabaja en latentes de 16px; redondear evita errores opacos.
    req["width"] = max(256, min(2048, (req["width"] // 16) * 16))
    req["height"] = max(256, min(2048, (req["height"] // 16) * 16))
    req["steps"] = max(1, min(16, req["steps"]))
    req["batch_size"] = max(1, min(4, req["batch_size"]))

    if not req["prompt"].strip():
        raise ValueError("El prompt de imagen esta vacio")

    return req


def build_workflow(req: dict, seed: int, job_id: str) -> dict:
    """
    Copia profunda de la plantilla (WORKFLOW) con los valores del pedido
    inyectados en los nodos correspondientes (ver las constantes NODE_*
    arriba). Cada llamada relee el JSON de disco, asi que no hay estado
    compartido entre generaciones concurrentes.

    El nombre de archivo de salida incluye el job_id para poder
    encontrarlo despues entre los outputs de ComfyUI sin colisiones.
    """
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    for node in (NODE_POSITIVE, NODE_LATENT, NODE_SAMPLER, NODE_SAVE):
        if node not in wf:
            raise RuntimeError(
                f"El workflow no tiene el nodo {node}. "
                "Reexportalo y actualiza las constantes NODE_* de comfy.py"
            )

    wf[NODE_POSITIVE]["inputs"]["text"] = req["prompt"]
    wf[NODE_LATENT]["inputs"]["width"] = req["width"]
    wf[NODE_LATENT]["inputs"]["height"] = req["height"]
    wf[NODE_LATENT]["inputs"]["batch_size"] = req["batch_size"]
    wf[NODE_SAMPLER]["inputs"]["seed"] = seed
    wf[NODE_SAMPLER]["inputs"]["steps"] = req["steps"]
    wf[NODE_SAVE]["inputs"]["filename_prefix"] = f"sauzal/{job_id}"

    return wf


def _queue(workflow: dict, client_id: str) -> str:
    """
    Manda el workflow armado a la API de ComfyUI (POST /prompt) para que
    lo encole. Devuelve el `prompt_id` que ComfyUI asigna, necesario para
    despues consultar el progreso/resultado en `_wait`.

    Distingue dos tipos de fallo:
    - HTTP 400: ComfyUI rechazo el workflow (parametros invalidos para
      algun nodo) -> se levanta con el detalle que manda ComfyUI.
    - "node_errors" en la respuesta: el workflow se acepto sintacticamente
      pero algun nodo especifico tiene un error de configuracion.
    """
    r = requests.post(
        f"{COMFY}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )

    if r.status_code == 400:
        raise RuntimeError(f"ComfyUI rechazo el workflow: {r.text[:500]}")

    r.raise_for_status()
    body = r.json()

    if body.get("node_errors"):
        raise RuntimeError(f"Errores de nodo: {json.dumps(body['node_errors'])[:500]}")

    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI no devolvio prompt_id")

    return prompt_id


def _wait(prompt_id: str) -> list[dict]:
    """
    Hace polling a GET /history/{prompt_id} cada POLL segundos hasta que
    ComfyUI reporte que la generacion termino. Tres desenlaces posibles:

    1. Aparecen imagenes de salida -> se devuelven sus referencias
       (filename/subfolder/type), listas para pasarle a `_fetch`.
    2. ComfyUI marca la ejecucion como "error" -> se levanta con los
       mensajes de error que haya devuelto.
    3. Se cumple el TIMEOUT sin ninguna de las anteriores -> TimeoutError.
    """
    deadline = time.monotonic() + TIMEOUT

    while time.monotonic() < deadline:
        r = requests.get(f"{COMFY}/history/{prompt_id}", timeout=15)
        r.raise_for_status()
        entry = r.json().get(prompt_id)

        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"ComfyUI fallo: {json.dumps(status.get('messages', []))[:500]}"
                )
            images = [
                img
                for out in entry.get("outputs", {}).values()
                for img in out.get("images", [])
                if img.get("type") == "output"
            ]
            if images:
                return images

        time.sleep(POLL)

    raise TimeoutError(f"La generacion excedio {TIMEOUT}s")


def _fetch(ref: dict) -> bytes:
    """Descarga los bytes de una imagen generada, usando la referencia
    (filename/subfolder/type) que devolvio `_wait`, via GET /view."""
    r = requests.get(
        f"{COMFY}/view",
        params={
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.content


def generate(model: str, prompt: str) -> str:
    """
    Punto de entrada publico del modulo (lo llama agent.py). Orquesta
    todo el flujo: parsear el pedido, armar el workflow, encolarlo,
    esperar a que termine, bajar las imagenes resultantes, y devolver
    todo como un JSON serializado (con las imagenes en base64) listo
    para guardarse tal cual en la columna `result` del servidor.
    """
    started = time.monotonic()
    req = parse_request(prompt)
    job_id = uuid.uuid4().hex[:12]
    seed = req["seed"] if req["seed"] is not None else random.randint(0, 2**53)

    prompt_id = _queue(build_workflow(req, seed, job_id), f"sauzal-{job_id}")
    refs = _wait(prompt_id)
    blobs = [_fetch(ref) for ref in refs]

    return json.dumps(
        {
            "type": "image",
            "model": model,
            "format": "png",
            "images": [base64.b64encode(b).decode() for b in blobs],
            "prompt": req["prompt"],
            "width": req["width"],
            "height": req["height"],
            "steps": req["steps"],
            "seed": seed,
            "bytes": sum(len(b) for b in blobs),
            "total_duration_ns": int((time.monotonic() - started) * 1e9),
        },
        ensure_ascii=False,
    )
