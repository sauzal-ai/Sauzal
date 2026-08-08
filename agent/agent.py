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
  2. Releva el hardware/software de la maquina (VRAM, RAM, driver, motor
     de computo, versiones, temperatura, consumo) y corre un benchmark
     fijo una sola vez, para que el panel /admin del servidor tenga datos
     reales de este nodo (ver `capabilities()` y `static_info()`).
  3. Se registra contra el servidor (control plane) la primera vez, y
     guarda las credenciales (node_id + token) en agent_config.json para
     no tener que volver a registrarse en cada arranque.
  4. Entra en un bucle infinito: cada 2 segundos manda un heartbeat
     ("sigo vivo, esto es lo que puedo hacer, esto es lo que consumi") y
     pregunta si tiene un trabajo asignado. Si lo tiene, lo ejecuta con
     el backend que corresponda segun el modelo pedido, y devuelve el
     resultado.

Importante: el agente SIEMPRE inicia la conexion hacia el servidor
(nunca al reves). Esto permite correrlo detras de NAT/firewall sin abrir
ningun puerto de entrada — solo necesita salida a internet (o a la red
local, si el servidor esta en la misma LAN).

Sobre las metricas de hardware (VRAM, temperatura, consumo, driver): se
leen con `nvidia-smi` cuando hay GPU NVIDIA. En GPUs de otros fabricantes
(por ejemplo, AMD en Windows, como se probo con una RX 6600) no existe un
equivalente estandar y sin dependencias extra para leer temperatura o
consumo -- esos campos quedan en `null` a proposito en vez de inventar un
valor. VRAM/driver si tienen una alternativa mejor que nada: WMI en
Windows (nombre y version de driver de cualquier fabricante) y el
`/system_stats` de ComfyUI (VRAM tal como la ve PyTorch, cross-vendor, si
ComfyUI esta corriendo).

Uso:
    python agent.py --server http://192.168.1.78:8000 --name mi-pc --register

    --server    URL del control plane (obligatorio)
    --name      nombre legible para identificar el nodo (default: hostname)
    --register  fuerza un registro nuevo aunque ya exista agent_config.json
                (util si se borro el nodo del lado del servidor, o si se
                quiere renombrar)
    --paused    arranca el nodo pausado (no va a recibir trabajos hasta
                que se reactive manualmente desde el panel /admin). Es
                solo la preferencia INICIAL: una vez registrado, el
                estado real de pausa se controla desde /admin.
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

# Datos que no cambian mientras el agente esta corriendo (o cambian tan
# poco que no vale la pena remedirlos en cada heartbeat): se calculan una
# sola vez y quedan cacheados aca. Para refrescarlos hay que reiniciar el
# agente (o volver a correrlo con --register).
_STATIC_CACHE: dict = {}


def cmd(args):
    """
    Ejecuta un comando de sistema y devuelve su salida como texto, o
    string vacio si falla por cualquier motivo (comando no encontrado,
    timeout, etc). Se usa para `nvidia-smi` y `wmic` sin que la falta de
    alguno (ej: nvidia-smi en una maquina con AMD, o wmic fuera de
    Windows) rompa el resto del agente.
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


def ollama_version():
    """Version del binario de Ollama, via su propio API (evita depender
    de encontrar el ejecutable en el PATH, que en Windows no siempre
    queda disponible para subprocess). None si no responde."""
    try:
        return requests.get(f"{OLLAMA}/api/version",timeout=5).json().get("version")
    except Exception:
        return None


def nvidia_query():
    """
    Una sola llamada a nvidia-smi pidiendo TODOS los campos que interesan
    de una vez (nombre, VRAM total/usada, temperatura, consumo, driver) —
    mas barato que invocar el comando varias veces por heartbeat. Si no
    hay GPU NVIDIA o el comando no existe, devuelve un dict vacio: todo lo
    que dependa de esto en capabilities() queda en None (no se inventa
    ningun valor).
    """
    out = cmd(["nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw,driver_version",
        "--format=csv,noheader,nounits"])
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 6:
        return {}
    name, mem_total, mem_used, temp, power, driver = parts
    def to_float(x):
        try:
            return float(x)
        except ValueError:
            return None
    return {
        "name": name,
        "vram_total_mb": to_float(mem_total),
        "vram_used_mb": to_float(mem_used),
        "gpu_temp_c": to_float(temp),
        "gpu_power_w": to_float(power),
        "driver_version": driver,
    }


def gpu_names_fallback():
    """
    Nombre(s) de GPU para maquinas sin nvidia-smi (ej: AMD en Windows, el
    caso probado con la RX 6600). Usa WMI (Win32_VideoController), que
    funciona para cualquier fabricante en Windows. En Linux sin
    nvidia-smi no se intenta nada mas (haria falta lspci/glxinfo, que no
    siempre estan instalados) y queda una lista vacia.
    """
    if platform.system() != "Windows":
        return []
    out = cmd(["wmic", "path", "win32_VideoController", "get", "name"])
    return [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() != "name"]


def driver_version_fallback():
    """Version de driver para GPUs no-NVIDIA en Windows, via WMI. None
    en cualquier otro caso (no hay un equivalente simple en Linux sin
    herramientas especificas por fabricante)."""
    if platform.system() != "Windows":
        return None
    out = cmd(["wmic", "path", "win32_VideoController", "get", "DriverVersion"])
    lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() != "driverversion"]
    return lines[0] if lines else None


def ram_info():
    """
    RAM total/usada del SISTEMA (no de la GPU), en MB. Windows: WMI
    (Win32_OperatingSystem). Linux: /proc/meminfo. Devuelve (None, None)
    si falla cualquiera de los dos metodos, en vez de romper el resto de
    las capabilities.
    """
    try:
        if platform.system() == "Windows":
            out = cmd(["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:list"])
            values = {}
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if v.strip().isdigit():
                        values[k] = int(v.strip())
            total_kb = values.get("TotalVisibleMemorySize")
            free_kb = values.get("FreePhysicalMemory")
            if total_kb is None:
                return None, None
            used_kb = (total_kb - free_kb) if free_kb is not None else None
            return round(total_kb/1024), (round(used_kb/1024) if used_kb is not None else None)
        else:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    parts = v.strip().split()
                    if parts and parts[0].isdigit():
                        info[k] = int(parts[0])
            total_kb = info.get("MemTotal")
            avail_kb = info.get("MemAvailable")
            if total_kb is None:
                return None, None
            used_kb = (total_kb - avail_kb) if avail_kb is not None else None
            return round(total_kb/1024), (round(used_kb/1024) if used_kb is not None else None)
    except Exception:
        return None, None


def compute_backend(gpu_names, has_nvidia):
    """
    Heuristica simple para etiquetar el "motor de computo" que probablemente
    usan Ollama/ComfyUI en esta maquina, a partir de que GPU se detecto.
    No es 100% autoritativo (Ollama podria elegir otro backend interno),
    pero da una pista util para el panel de administracion.
    """
    if has_nvidia:
        return "CUDA"
    joined = " ".join(gpu_names).lower()
    if "amd" in joined or "radeon" in joined:
        return "ROCm/Vulkan"
    if "intel" in joined:
        return "Vulkan/oneAPI"
    if gpu_names:
        return "desconocido"
    return "CPU"


def comfy_system_stats():
    """
    Consulta GET /system_stats de ComfyUI, que ya calcula (via PyTorch) la
    version de Python y la VRAM total/libre que ve el proceso. Sirve como
    fuente alternativa de VRAM cuando nvidia-smi no esta disponible (por
    ejemplo, una GPU AMD con ComfyUI corriendo sobre ROCm/DirectML).
    Dict vacio si ComfyUI no responde.
    """
    try:
        data = requests.get(f"{comfy.COMFY}/system_stats", timeout=5).json()
    except Exception:
        return {}
    out = {"python_version": (data.get("system") or {}).get("python_version")}
    devices = data.get("devices") or []
    if devices:
        dev = devices[0]
        vram_total = dev.get("vram_total")
        vram_free = dev.get("vram_free")
        out["vram_total_mb"] = round(vram_total/1024/1024) if vram_total else None
        out["vram_used_mb"] = (
            round((vram_total-vram_free)/1024/1024)
            if vram_total and vram_free is not None else None
        )
    return out


def run_benchmark(has_ollama, has_comfy):
    """
    Corre UNA sola vez (al registrarse) una prueba fija por cada backend
    disponible, para tener un numero real de desempeno del nodo en vez de
    depender solo de que se ejecuten trabajos reales con el tiempo.
    Cualquier fallo aca se guarda como texto en vez de propagarse: el
    benchmark es informativo, no debe impedir que el nodo se registre.
    """
    bench = {}
    if has_ollama:
        try:
            model = ollama_models()[0]
            r = requests.post(f"{OLLAMA}/api/generate",
                json={"model":model,"prompt":"Respondeme solo con la palabra: listo","stream":False},
                timeout=120)
            r.raise_for_status()
            b = r.json()
            eval_count, eval_ns = b.get("eval_count"), b.get("eval_duration")
            if eval_count and eval_ns:
                bench["tokens_per_sec"] = round(eval_count/(eval_ns/1e9), 1)
                bench["benchmark_model"] = model
        except Exception as exc:
            bench["tokens_per_sec_error"] = repr(exc)
    if has_comfy:
        try:
            started = time.monotonic()
            comfy.generate(comfy.MODEL_NAME, json.dumps({"prompt":"benchmark","width":256,"height":256,"steps":1}))
            bench["image_seconds"] = round(time.monotonic()-started, 2)
        except Exception as exc:
            bench["image_seconds_error"] = repr(exc)
    return bench


def measure_network(server):
    """
    Descarga UNA sola vez (al registrarse) el bloque fijo de 2MB que
    expone el servidor en GET /bench/payload, y calcula la velocidad de
    red hacia EL (en Mbps). Se mide contra el propio control plane a
    proposito: lo relevante para la red Sauzal es que tan rapido llegan
    los trabajos/resultados, no un benchmark generico de banda ancha.
    """
    try:
        started = time.monotonic()
        r = requests.get(f"{server}/bench/payload", timeout=30)
        r.raise_for_status()
        elapsed = time.monotonic() - started
        if elapsed <= 0:
            return None
        return round((len(r.content)*8/1_000_000)/elapsed, 1)
    except Exception:
        return None


def static_info(server):
    """
    Calcula (una sola vez por arranque, cacheado en _STATIC_CACHE) todo lo
    que no cambia mientras el agente esta corriendo: nombre(s) de GPU, RAM
    (foto del momento del arranque), driver, motor de computo estimado,
    versiones de Ollama/ComfyUI, benchmark, y velocidad de red hacia el
    servidor. Los campos genuinamente dinamicos (temperatura, consumo,
    VRAM usada) se recalculan aparte en cada llamada a capabilities().
    """
    if _STATIC_CACHE:
        return _STATIC_CACHE

    nv = nvidia_query()
    gpu_names = [nv["name"]] if nv else gpu_names_fallback()
    ram_total, ram_used = ram_info()
    has_ollama = bool(ollama_models())
    has_comfy = comfy.available()

    _STATIC_CACHE.update({
        "gpu_names": gpu_names,
        "ram_total_mb": ram_total,
        "ram_used_mb_at_start": ram_used,
        "driver_version": nv.get("driver_version") or driver_version_fallback(),
        "compute_backend": compute_backend(gpu_names, bool(nv)),
        "ollama_version": ollama_version() if has_ollama else None,
        "comfy_python_version": comfy_system_stats().get("python_version") if has_comfy else None,
        "network_mbps": measure_network(server),
        "benchmark": run_benchmark(has_ollama, has_comfy),
    })
    return _STATIC_CACHE


def capabilities(server):
    """
    Arma el JSON de "capabilities" que el nodo le manda al servidor en el
    registro y en cada heartbeat: todo lo que el panel /admin necesita
    para mostrar el estado de este nodo (backends, modelos, hardware,
    versiones, benchmark, temperatura/consumo en vivo).

    El campo "services" es un resumen booleano (bool(lista) es True si la
    lista no esta vacia) que usa el propio agente para decidir que
    backends anunciar como activos.

    `comfy.available()` se llama UNA sola vez aca (no dos) a proposito:
    tanto `comfy_system_stats()` como el equivalente de `comfy.models()`
    lo necesitan, y cada llamada implica un intento de conexion HTTP que,
    cuando ComfyUI no esta corriendo, tarda ~1s en fallar (ver el
    comentario de timeout en comfy.py::available()) -- llamarlo dos veces
    duplicaba ese costo en cada heartbeat.
    """
    static = static_info(server)
    nv = nvidia_query()  # se recalcula siempre: temp/consumo/VRAM usada cambian con el tiempo
    has_comfy_now = comfy.available()
    comfy_stats = comfy_system_stats() if has_comfy_now else {}

    text_models = ollama_models()
    image_models = [comfy.MODEL_NAME] if has_comfy_now else []

    return json.dumps({
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "gpu": static.get("gpu_names", []),
        "ollama_models": text_models,
        "image_models": image_models,
        "services": {"ollama": bool(text_models), "comfyui": bool(image_models)},

        "vram_total_mb": nv.get("vram_total_mb") if nv else comfy_stats.get("vram_total_mb"),
        "vram_used_mb": nv.get("vram_used_mb") if nv else comfy_stats.get("vram_used_mb"),
        "ram_total_mb": static.get("ram_total_mb"),
        "ram_used_mb": static.get("ram_used_mb_at_start"),  # foto del arranque, ver static_info()
        "driver_version": static.get("driver_version"),
        "compute_backend": static.get("compute_backend"),
        "ollama_version": static.get("ollama_version"),
        "comfy_python_version": static.get("comfy_python_version"),
        "network_mbps": static.get("network_mbps"),
        "benchmark": static.get("benchmark"),
        "gpu_temp_c": nv.get("gpu_temp_c") if nv else None,
        "gpu_power_w": nv.get("gpu_power_w") if nv else None,
    },ensure_ascii=False)


def register(server,name,paused=False):
    """
    Da de alta este nodo contra el servidor (POST /nodes/register) y
    guarda la respuesta (node_id, token) mas la URL del servidor en
    agent_config.json, para poder reusarla en el proximo arranque sin
    tener que registrarse de nuevo (cada registro crea un nodo NUEVO del
    lado del servidor, asi que no conviene llamarlo en cada inicio).

    `paused` es la preferencia de disponibilidad INICIAL (flag --paused
    del script): despues de este registro, el estado real se controla
    desde el panel /admin, no desde aca.
    """
    r=requests.post(f"{server}/nodes/register",
        json={"name":name,"capabilities":capabilities(server),"paused":paused},timeout=60)
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
       --register; esto es lo que dispara `static_info()`: benchmark fijo
       + medicion de velocidad de red, una sola vez), o reusa las
       credenciales guardadas.
    4. Entra en el bucle principal: cada 2 segundos, manda un heartbeat
       (con capabilities actualizadas, la latencia del heartbeat anterior,
       y el incremento de energia consumida desde el heartbeat anterior)
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
    p.add_argument("--paused",action="store_true",
        help="Arranca el nodo pausado (no recibe trabajos hasta reactivarlo desde /admin)")
    args=p.parse_args()

    has_ollama=bool(ollama_models())
    has_comfy=comfy.available()
    if not has_ollama and not has_comfy:
        raise SystemExit(
            "Este nodo no tiene ningun backend disponible.\n"
            f"  Ollama en {OLLAMA}: no responde o sin modelos\n"
            f"  ComfyUI en {comfy.COMFY}: no responde o falta {comfy.WORKFLOW.name}"
        )

    config=register(args.server,args.name,args.paused) if args.register or not CONFIG.exists() else json.loads(CONFIG.read_text())
    config["server"]=args.server.rstrip("/")
    auth={"node_id":config["node_id"],"token":config["token"]}

    backends=[n for n,ok in (("ollama",has_ollama),("comfyui",has_comfy)) if ok]
    print(f"Agente Sauzal conectado: {config['node_id']}")
    print(f"Backends activos: {', '.join(backends)}")

    # Se usan para calcular, entre un heartbeat y el siguiente, cuanta
    # energia (Wh) se consumio y cuanto tardo el heartbeat anterior. El
    # primer ciclo no tiene "anterior" con que comparar, quedan en None.
    last_heartbeat_mono=None
    last_latency_ms=None

    while True:
        try:
            now_mono=time.monotonic()
            nv=nvidia_query()
            power_w=nv.get("gpu_power_w") if nv else None
            energy_delta=None
            if last_heartbeat_mono is not None and power_w is not None:
                # potencia (W) x tiempo transcurrido (h) = energia (Wh)
                energy_delta=round(power_w*(now_mono-last_heartbeat_mono)/3600, 6)
            last_heartbeat_mono=now_mono

            t0=time.monotonic()
            requests.post(f"{config['server']}/nodes/heartbeat",
              json={
                **auth,"status":"available","capabilities":capabilities(config['server']),
                "latency_ms":last_latency_ms,"energy_wh_delta":energy_delta,
              },timeout=15).raise_for_status()
            last_latency_ms=round((time.monotonic()-t0)*1000, 1)

            # Pregunta si hay un trabajo en cola para este nodo puntual.
            r=requests.post(f"{config['server']}/agent/pull",json=auth,timeout=20)
            r.raise_for_status()
            job=r.json()["job"]
            if job:
                kind="imagen" if comfy.is_image_model(job["model"]) else "texto"
                print(f"Ejecutando {job['job_id']} [{kind}] con {job['model']}")
                # Cronometra la ejecucion real de punta a punta (exito o
                # fallo), independiente del backend -- a diferencia de los
                # tiempos que devuelve Ollama (que solo cubren texto), esto
                # sirve igual para jobs de texto y de imagen, y tambien
                # informa cuanto tardo en fallar/timeoutear un job.
                t_exec=time.monotonic()
                try:
                    output=execute(job)
                    duration_ms=round((time.monotonic()-t_exec)*1000, 1)
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
                        "duration_ms":duration_ms,
                    }
                except Exception as exc:
                    # Cualquier fallo durante la ejecucion (ComfyUI caido,
                    # modelo invalido, timeout, etc.) se reporta como
                    # resultado fallido en vez de tirar abajo el agente.
                    duration_ms=round((time.monotonic()-t_exec)*1000, 1)
                    print("  fallo:",repr(exc))
                    payload={**auth,"success":False,"error":repr(exc),"duration_ms":duration_ms}
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
