"""
Sauzal · backend de imagen (ComfyUI)

Modulo del agente. No expone puertos: el agente lo llama cuando el server
le asigna un job cuyo modelo es de imagen.

Los jobs de imagen viajan por la misma tabla que los de texto. El campo
prompt admite dos formas:

    texto plano                -> se usa tal cual, con los valores por defecto
    {"prompt":"...", "width":1024, "height":1024, "steps":4, "seed":42}

El resultado es un JSON con el PNG en base64, para que entre en la
columna result sin cambiar el esquema de la base.
"""

from __future__ import annotations

import base64
import json
import random
import time
import uuid
from pathlib import Path

import requests

COMFY = "http://127.0.0.1:8188"
WORKFLOW = Path(__file__).with_name("workflow_flux_schnell.json")

# IDs de nodo del workflow exportado con Workflow -> Export (API).
# Si reexportas despues de agregar o borrar nodos, revisa que sigan siendo estos.
NODE_POSITIVE = "6"
NODE_LATENT = "27"
NODE_SAMPLER = "31"
NODE_SAVE = "9"

MODEL_NAME = "flux1-schnell-fp8"
TIMEOUT = 600
POLL = 0.5


def available() -> bool:
    """True si este nodo puede generar imagenes."""
    if not WORKFLOW.exists():
        return False
    try:
        requests.get(f"{COMFY}/system_stats", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False


def models() -> list[str]:
    return [MODEL_NAME] if available() else []


def is_image_model(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("flux") or m.startswith("sdxl") or m.startswith("sauzal-image")


def busy() -> bool:
    """ComfyUI procesa de a uno. Sirve para no encolar a ciegas."""
    try:
        q = requests.get(f"{COMFY}/queue", timeout=5).json()
        return bool(q.get("queue_running")) or bool(q.get("queue_pending"))
    except requests.RequestException:
        return False


def parse_request(prompt: str) -> dict:
    """El prompt puede ser texto plano o un JSON con parametros."""
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
    """Copia profunda de la plantilla con los valores del pedido inyectados."""
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
    """Genera y devuelve un JSON serializado, listo para la columna result."""
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
