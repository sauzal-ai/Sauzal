#!/bin/bash
set -e

# ============================================================
# Requiere la variable de entorno SAUZAL_SERVER al arrancar el contenedor:
#   docker run -e SAUZAL_SERVER=https://tu-tunel.trycloudflare.com ...
# ============================================================

if [ -z "$SAUZAL_SERVER" ]; then
  echo "ERROR: falta la variable de entorno SAUZAL_SERVER (URL del control plane)."
  exit 1
fi

echo "== Arrancando Ollama =="
ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Esperando a que Ollama responda en :11434..."
until curl -sf http://127.0.0.1:11434 > /dev/null 2>&1; do
  if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "Ollama se cayo al arrancar. Log:"
    cat /workspace/ollama.log
    exit 1
  fi
  sleep 2
done
echo "Ollama listo."

echo "== Arrancando ComfyUI =="
cd /workspace/ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
COMFY_PID=$!

echo "Esperando a que ComfyUI responda en :8188..."
until curl -sf http://127.0.0.1:8188 > /dev/null 2>&1; do
  # si ComfyUI murió, cortar acá con el log a la vista
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI se cayó al arrancar. Log:"
    cat /workspace/comfyui.log
    exit 1
  fi
  sleep 2
done
echo "ComfyUI listo."

echo "== Arrancando agente Sauzal contra $SAUZAL_SERVER =="
cd /workspace/Sauzal/agent
python3 agent.py --server "$SAUZAL_SERVER"
