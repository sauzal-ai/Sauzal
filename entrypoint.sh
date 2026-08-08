#!/bin/bash
set -e
# ============================================================
# Sauzal · entrypoint del contenedor de nodo GPU
# ============================================================
#
# Este script es el ENTRYPOINT de la imagen (ver Dockerfile). Se ejecuta
# una sola vez, al arrancar el contenedor, y hace tres cosas en orden,
# esperando a que cada una este realmente lista antes de seguir con la
# siguiente (porque el agente necesita a las otras dos ya respondiendo):
#
#   1. Arranca Ollama (backend de texto) y espera a que responda en :11434
#   2. Arranca ComfyUI (backend de imagen) y espera a que responda en :8188
#   3. Arranca el agente (agent.py), que se conecta al servidor Sauzal y
#      se queda corriendo en primer plano (por eso es el ultimo paso: si
#      el contenedor se queda "vivo" es porque este proceso sigue activo).
#
# `set -e` corta el script ante cualquier comando que falle -- salvo dentro
# de los bucles `until`, que estan pensados justamente para tolerar que el
# curl de chequeo falle mientras el servicio todavia esta levantando.
#
# Requiere la variable de entorno SAUZAL_SERVER al arrancar el contenedor:
#   docker run -e SAUZAL_SERVER=https://tu-tunel.trycloudflare.com ...
# ============================================================

if [ -z "$SAUZAL_SERVER" ]; then
  echo "ERROR: falta la variable de entorno SAUZAL_SERVER (URL del control plane)."
  exit 1
fi

echo "== Arrancando Ollama =="
# Se manda a background (&) y se redirige su salida a un log en archivo,
# porque este script necesita seguir ejecutando el resto de los pasos
# mientras Ollama sigue corriendo de fondo durante toda la vida del Pod.
ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Esperando a que Ollama responda en :11434..."
until curl -sf http://127.0.0.1:11434 > /dev/null 2>&1; do
  # Si el proceso ya no existe, Ollama se cayo al arrancar (no tiene
  # sentido seguir esperando un puerto que nunca va a responder) -- se
  # corta ahi mismo mostrando el log para poder ver el error real.
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
# --listen 0.0.0.0 para que sea alcanzable desde fuera del contenedor
# (puerto expuesto en el Dockerfile con EXPOSE 8188), no solo localhost.
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
# Sin "&": este es el proceso principal del contenedor. Corre en primer
# plano y en bucle infinito (ver agent.py); si termina o crashea, el
# contenedor entero se detiene (asi RunPod puede detectar el fallo).
python3 agent.py --server "$SAUZAL_SERVER"
