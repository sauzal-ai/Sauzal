# ============================================================
# Sauzal · imagen de nodo GPU (ComfyUI + FLUX.1-schnell + Ollama + agente)
# ============================================================
#
# Esta imagen es lo que corre en cada nodo remoto de la red Sauzal cuando
# el nodo es un contenedor (tipicamente un Pod de RunPod con GPU NVIDIA,
# pero sirve en cualquier host con NVIDIA Container Toolkit). Trae todo
# lo necesario para que, al arrancar, la maquina pueda:
#
#   - Generar texto  -> via Ollama (backend de agent.py)
#   - Generar imagen -> via ComfyUI + el modelo FLUX.1-schnell (backend
#                        de agent/comfy.py)
#   - Conectarse solita a la red Sauzal -> via agent/agent.py, que se
#     registra contra el servidor (control plane) y empieza a recibir
#     trabajos.
#
# El build se arma en GitHub Actions (ver .github/workflows/build-image.yml)
# y sube el resultado a Docker Hub. RunPod despues solo necesita esa
# imagen ya publicada, no hace falta compilar nada en el Pod.
#
# Por que se "hornean" los modelos (FLUX, gemma3:4b) DENTRO de la imagen
# en vez de descargarlos al arrancar el contenedor: para que un Pod nuevo
# este listo para trabajar en segundos en vez de tener que bajar ~20GB
# cada vez que se crea un Pod. El costo de esta decision es que la imagen
# pesa mucho (20GB+) y el build en CI tarda mas y consume mas espacio en
# disco del runner (ver los pasos de limpieza en el workflow de CI).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ---- Dependencias del sistema ----
# python3/pip/venv: para correr ComfyUI y el agente (ambos son Python).
# git: para clonar ComfyUI y este mismo repo dentro de la imagen.
# wget: para bajar el modelo FLUX (archivo grande, con progreso visible).
# curl: usado por el instalador de Ollama y por entrypoint.sh (healthchecks).
# zstd: dependencia del instalador oficial de Ollama (install.sh la
#       requiere para descomprimir su paquete; sin esto el install falla).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        zstd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- ComfyUI ----
# Clonamos el repo oficial (--depth 1: solo el ultimo commit, para no
# bajar todo el historial de git) e instalamos sus dependencias Python.
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
WORKDIR /workspace/ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Modelo FLUX.1-schnell fp8 (17.2 GB, Apache 2.0) ----
# Horneado en la imagen para no re-descargar en cada Pod nuevo.
# --progress=dot:giga hace que wget imprima muchas menos lineas de
# progreso (cada punto representa mucho mas volumen de datos), para que
# el log completo entre en el limite de salida en vivo de GitHub Actions
# sin cortarse ("output clipped") durante una descarga tan grande.
RUN mkdir -p models/checkpoints && \
    wget -q --show-progress --progress=dot:giga -O models/checkpoints/flux1-schnell-fp8.safetensors \
    https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors

# ---- Ollama ----
# Instalador oficial. Solo instala el binario (no hay systemd en este
# contenedor, asi que el "servicio" que el instalador intenta crear no
# se activa solo) — el que efectivamente arranca "ollama serve" es
# entrypoint.sh cuando corre el contenedor.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hornea el modelo de texto por defecto en la imagen (evita descargarlo
# en cada Pod nuevo, igual que con FLUX arriba).
#
# Truco de build: "ollama pull" necesita al servidor de Ollama corriendo
# para funcionar, pero en un paso RUN no hay ningun proceso de fondo
# persistente entre comandos. Por eso se arranca "ollama serve" en
# background (&) DENTRO del mismo RUN, se espera unos segundos a que
# levante, se hace el pull, y al terminar el RUN ese proceso de fondo se
# corta solo (no pasa nada, ya cumplio su proposito: dejar el modelo
# descargado en el filesystem de esta capa de la imagen).
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma3:4b && \
    sleep 2

# ---- Repo Sauzal (agente) ----
# Clonamos el propio repo DENTRO de la imagen para que el contenedor
# tenga agent.py, comfy.py y el workflow de FLUX sin depender de un bind
# mount. Ojo: esto significa que la imagen queda "congelada" con el
# commit de main que estaba disponible en el momento del build — para
# que el agente tome cambios de codigo nuevos hay que reconstruir la
# imagen (o sea, volver a pushear a main y esperar el build de CI).
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/sauzal-ai/Sauzal.git

WORKDIR /workspace/Sauzal/agent
RUN pip3 install --no-cache-dir -r requirements.txt

# ---- Entrypoint ----
# Ver entrypoint.sh: arranca Ollama, despues ComfyUI, espera a que ambos
# respondan, y recien ahi lanza el agente contra $SAUZAL_SERVER.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
# Puerto de ComfyUI (interfaz web + API). Ollama (11434) no se expone
# hacia afuera del contenedor a proposito: solo lo usa el agente,
# localmente, no hace falta que sea accesible desde internet.
EXPOSE 8188

ENTRYPOINT ["/entrypoint.sh"]
