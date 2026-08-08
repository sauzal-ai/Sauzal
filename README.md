# Sauzal PoC 0.2

Para una explicación completa de la arquitectura (qué hace cada programa,
cómo se relacionan, qué corre dónde, y el detalle de la base de datos),
ver [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

Prueba de inferencia real:

```text
PC cliente -> servidor Sauzal -> agente remoto -> Ollama/GPU -> respuesta
```

## Programas

**Servidor:** Python 3.12 o 3.13.  
**PC con GPU:** driver NVIDIA, Python y Ollama.  
**Cliente:** Python.

Para la primera prueba, usá los equipos en la misma red.

## PASO 1 — Servidor

PowerShell:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Averiguá su IP:

```powershell
ipconfig
```

Probá:

```text
http://IP_SERVIDOR:8000/health
```

## PASO 2 — PC remoto con GPU

Verificá NVIDIA:

```powershell
nvidia-smi
```

Instalá Ollama y comprobá:

```powershell
ollama --version
ollama pull gemma3:4b
ollama run gemma3:4b
```

Salí con `/bye`. Ollama deja una API local en `127.0.0.1:11434`.

## PASO 3 — Agente Sauzal

En la PC con GPU:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python agent\agent.py --server http://IP_SERVIDOR:8000 --name gpu-casa-1 --register
```

Comprobá el nodo:

```text
http://IP_SERVIDOR:8000/nodes
```

Debe figurar `online: true`.

## PASO 4 — Cliente

Desde otro PC:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python client\infer.py `
  --server http://IP_SERVIDOR:8000 `
  --model gemma3:4b `
  --prompt "Dame una receta con pollo, arroz y cebolla"
```

La respuesta debe ser producida en la GPU remota.

## Qué valida

- Registro y heartbeat del nodo.
- Conexión solo saliente desde el agente.
- Asignación de una inferencia.
- Ejecución local en la GPU remota.
- Retorno de respuesta y métricas.

## No usar en producción

Todavía no tiene HTTPS, usuarios, cifrado de payload a nivel de aplicación, sandbox fuerte, reintentos ni pagos.
