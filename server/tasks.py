"""
Sauzal · Task Decomposition + Aggregation
============================================

Este modulo le permite a Sauzal recibir una tarea compleja, partirla en
subtareas independientes o con dependencias entre si, repartirlas entre
los nodos disponibles EN ESE MOMENTO (los que haya, sin importar
cuantos), y combinar los resultados parciales en un resultado final --
todo reutilizando la misma infraestructura de `jobs` que ya usa una
inferencia simple. El agente (agent/agent.py) no cambia ni una linea
para soportar esto: cada subtarea le llega como un job de texto mas,
identico a los que ya procesaba antes de que este modulo existiera.

Piezas nuevas (ver el docstring de `server/main.py::startup()` para el
esquema SQL completo):
  - Tabla `tasks`: la tarea padre (modo, estrategia de agregacion,
    politica de limites, resultado final).
  - `jobs.task_id` / `depends_on` / `requires` / `subtask_key` /
    `retry_count`: cada subtarea es una fila de `jobs` como cualquier
    otra, con estos campos extra.
  - Un nuevo status de job, 'blocked': una subtarea con dependencias sin
    cumplir. No es elegible para ningun nodo hasta que sus dependencias
    terminen (ver `_unblock_dependents()`).
  - Otro status nuevo, 'cancelled': cancelacion BLANDA -- la tarea ya no
    necesita esa subtarea. Si un nodo ya la estaba ejecutando, no hay
    forma de interrumpirlo a mitad de una inferencia; simplemente se
    ignora el resultado cuando (si) llega (ver `on_subtask_completed()`,
    que corta apenas ve que la tarea ya esta en un estado terminal).

Como se reparten las subtareas entre nodos
--------------------------------------------
A diferencia de un `/infer` suelto (donde el servidor elige UN nodo
disponible en el momento y se lo asigna de una), las subtareas de una
tarea se crean con `assigned_node=NULL` y quedan en un "pool" compartido.
Cualquier nodo que haga `/agent/pull` y no tenga ya un job asignado
directamente puede reclamar una (`claim_subtask()`), siempre que cumpla
los `requires` de esa subtarea puntual y la tarea no haya llegado a su
limite de `max_parallel`/`max_nodes`. Esto es lo que permite que "las
subtareas se ejecuten en los nodos disponibles cuando sea posible",
sin importar si hay 1 nodo o 50: las que no consiguen nodo todavia
simplemente esperan en la cola.

Motor de agregacion (pluggable por `mode`/`aggregation`)
-----------------------------------------------------------
| mode            | aggregation por defecto | que hace |
|-----------------|-------------------------|----------|
| batch           | collect_all             | espera TODAS, junta los resultados en un dict {key: resultado} |
| fan_out         | first_success           | usa la primera que tenga exito, cancela el resto |
| pipeline        | collect_all             | subtareas encadenadas por `depends_on`, cada una puede usar el resultado de otra via `{{key.result}}` en su prompt |
| map_reduce      | llm_synthesize          | como batch, pero el "reduce" es un job mas (le pide a un nodo que sintetice las N respuestas en una) |
| first_success   | first_success           | igual que fan_out con ese nombre explicito |
| consensus       | vote                    | espera mayoria (o que terminen todas) y elige la respuesta mas repetida |

No se intenta nunca partir una unica inferencia entre varias GPUs (nada
de tensor parallelism): cada subtarea es una inferencia completa e
independiente en UN nodo. El paralelismo es a nivel de tareas/etapas,
que es lo que tiene sentido con nodos remotos, heterogeneos, y con
latencias variables.
"""

from __future__ import annotations
import json, re, time, uuid
from collections import Counter

DEFAULT_AGGREGATION = {
    "batch": "collect_all",
    "fan_out": "first_success",
    "pipeline": "collect_all",
    "map_reduce": "llm_synthesize",
    "first_success": "first_success",
    "consensus": "vote",
}
VALID_MODES = set(DEFAULT_AGGREGATION)

# Marca especial de subtask_key para el job de sintesis que dispara la
# estrategia 'llm_synthesize' -- se trata distinto en on_subtask_completed()
# porque su resultado ES el resultado final de la tarea, no una subtarea
# mas a agregar.
AGGREGATE_KEY = "__aggregate__"

_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_\-]+)\.result\s*\}\}")


# ----------------------------------------------------------------------
# Utilidades chicas
# ----------------------------------------------------------------------

def _policy(task) -> dict:
    try:
        return json.loads(task["policy"] or "{}")
    except json.JSONDecodeError:
        return {}


def _extract_response(result_json: str | None) -> str:
    """Saca el texto de `response` de un JSON de resultado de job (el
    mismo formato que arma agent.py). Cadena vacia si no hay nada util."""
    if not result_json:
        return ""
    try:
        return json.loads(result_json).get("response") or ""
    except json.JSONDecodeError:
        return ""


def resolve_template(prompt: str, results_by_key: dict[str, str]) -> str:
    """
    Sustituye placeholders `{{key.result}}` en el prompt de una subtarea
    de pipeline por el resultado (texto de la respuesta) de la subtarea
    con esa `key`. Se llama justo antes de pasar una subtarea de
    'blocked' a 'queued', una vez que todas sus dependencias terminaron.
    Si el placeholder referencia una key sin resultado disponible se deja
    tal cual (no deberia pasar si las dependencias se resolvieron bien).
    """
    return _TEMPLATE_RE.sub(lambda m: results_by_key.get(m.group(1), m.group(0)), prompt)


# ----------------------------------------------------------------------
# Creacion de tareas
# ----------------------------------------------------------------------

def create_task(
    conn, mode: str, subtasks_spec: list[dict],
    session_id: str | None = None, aggregation: str | None = None,
    policy: dict | None = None, tenant_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Crea una tarea compuesta y todas sus subtareas (filas en `jobs`).

    `subtasks_spec`: lista de dicts con:
        key          -- identificador corto y UNICO dentro de la tarea
        model        -- modelo a usar en esa subtarea
        prompt       -- prompt de esa subtarea (puede tener placeholders
                        `{{otra_key.result}}` si depende de otra)
        depends_on   -- lista de `key` de otras subtareas de ESTA MISMA
                        tarea que tienen que terminar antes (opcional)
        requires     -- dict con capacidades que tiene que cumplir el
                        nodo que la ejecute, ej: {"service":"comfyui"}
                        (opcional)

    Las subtareas sin `depends_on` arrancan 'queued' (elegibles de
    inmediato para cualquier nodo via `claim_subtask()`); las que
    dependen de otras arrancan 'blocked' hasta que esas terminen.

    Devuelve (task_id, {key: job_id}) -- el mapeo sirve para que el
    endpoint HTTP le devuelva al cliente que job_id le toco a cada
    subtarea que el mismo nombro, manteniendo la trazabilidad desde el
    primer momento.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode invalido: {mode!r} (validos: {sorted(VALID_MODES)})")
    if not subtasks_spec:
        raise ValueError("una tarea necesita al menos una subtarea")

    keys = [s["key"] for s in subtasks_spec]
    if len(set(keys)) != len(keys):
        raise ValueError("las keys de las subtareas tienen que ser unicas dentro de la tarea")

    key_to_job_id = {key: str(uuid.uuid4()) for key in keys}
    for spec in subtasks_spec:
        for dep in spec.get("depends_on") or []:
            if dep not in key_to_job_id:
                raise ValueError(f"depends_on de '{spec['key']}' referencia una key inexistente: {dep!r}")

    task_id = str(uuid.uuid4())
    now = time.time()
    conn.execute(
        """INSERT INTO tasks(task_id,session_id,tenant_id,mode,aggregation,status,policy,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (task_id, session_id, tenant_id, mode, aggregation or DEFAULT_AGGREGATION[mode],
         "running", json.dumps(policy or {}), now, now),
    )

    for spec in subtasks_spec:
        dep_keys = spec.get("depends_on") or []
        dep_job_ids = [key_to_job_id[k] for k in dep_keys]
        status = "blocked" if dep_job_ids else "queued"
        conn.execute(
            """INSERT INTO jobs(
                 job_id,model,prompt,status,assigned_node,task_id,depends_on,requires,
                 subtask_key,created_at,updated_at
               ) VALUES(?,?,?,?,NULL,?,?,?,?,?,?)""",
            (
                key_to_job_id[spec["key"]], spec["model"], spec["prompt"], status,
                task_id, json.dumps(dep_job_ids) if dep_job_ids else None,
                json.dumps(spec["requires"]) if spec.get("requires") else None,
                spec["key"], now, now,
            ),
        )

    return task_id, key_to_job_id


# ----------------------------------------------------------------------
# Consulta
# ----------------------------------------------------------------------

def get_task(conn, task_id: str):
    return conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()


def list_subtasks(conn, task_id: str):
    return conn.execute(
        "SELECT * FROM jobs WHERE task_id=? ORDER BY created_at ASC", (task_id,)
    ).fetchall()


def compute_metrics(conn, task_id: str) -> dict:
    """
    Junta las metricas de una tarea a partir de sus subtareas (todas ya
    viven como filas de `jobs`, no hace falta guardar nada aparte): total
    de subtareas, nodos distintos usados, exitosas, fallidas, reintentos,
    duracion promedio/maxima por subtarea, duracion de la agregacion (si
    hubo un job de sintesis), y tokens procesados.
    """
    subtasks = [s for s in list_subtasks(conn, task_id) if s["subtask_key"] != AGGREGATE_KEY]
    nodes_used = {s["assigned_node"] for s in subtasks if s["assigned_node"]}
    succeeded = [s for s in subtasks if s["status"] == "completed"]
    failed = [s for s in subtasks if s["status"] == "failed"]
    durations = [s["duration_ms"] for s in subtasks if s["duration_ms"] is not None]
    agg_job = conn.execute(
        "SELECT duration_ms FROM jobs WHERE task_id=? AND subtask_key=?", (task_id, AGGREGATE_KEY)
    ).fetchone()
    return {
        "subtask_count": len(subtasks),
        "nodes_used": len(nodes_used),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "cancelled": sum(1 for s in subtasks if s["status"] == "cancelled"),
        "retries": sum(s["retry_count"] or 0 for s in subtasks),
        "avg_subtask_duration_ms": round(sum(durations) / len(durations), 1) if durations else None,
        "max_subtask_duration_ms": max(durations) if durations else None,
        "aggregation_duration_ms": agg_job["duration_ms"] if agg_job else None,
        "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in subtasks) or None,
        "output_tokens": sum(s["output_tokens"] or 0 for s in subtasks) or None,
    }


# ----------------------------------------------------------------------
# Reclamo de subtareas (el lado "pull" del pool compartido)
# ----------------------------------------------------------------------

def _node_satisfies(capabilities: dict, requires_json: str | None) -> bool:
    """True si un nodo (por sus `capabilities`, el mismo JSON que arma
    agent.py) cumple los `requires` de una subtarea. Sin `requires`,
    cualquier nodo sirve."""
    if not requires_json:
        return True
    try:
        req = json.loads(requires_json)
    except json.JSONDecodeError:
        return True
    if "service" in req and not (capabilities.get("services") or {}).get(req["service"]):
        return False
    if "model" in req:
        available = set(capabilities.get("ollama_models") or []) | set(capabilities.get("image_models") or [])
        if req["model"] not in available:
            return False
    return True


def claim_subtask(conn, node_id: str, node_capabilities_json: str | None):
    """
    Busca en el pool compartido de subtareas sin nodo asignado
    (`task_id` seteado, `assigned_node IS NULL`, `status='queued'`) una
    que este nodo pueda tomar -- filtrando por `requires` y por las
    politicas `max_parallel`/`max_nodes` de cada tarea -- y la reclama de
    forma atomica (`UPDATE ... WHERE assigned_node IS NULL`, chequeando
    que efectivamente se haya actualizado una fila, para no pisar a otro
    nodo que la haya reclamado en el mismo instante).

    Se llama desde `POST /agent/pull` SOLO si ese nodo no tenia ya un job
    asignado directamente (el caso normal de /infer sigue teniendo
    prioridad). Devuelve un dict {job_id, model, prompt} listo para
    mandarle al agente, o None si no hay nada que este nodo pueda tomar
    ahora mismo.
    """
    try:
        caps = json.loads(node_capabilities_json or "{}")
    except json.JSONDecodeError:
        caps = {}

    candidates = conn.execute("""
        SELECT * FROM jobs
        WHERE task_id IS NOT NULL AND assigned_node IS NULL AND status='queued'
        ORDER BY created_at LIMIT 25
    """).fetchall()

    for row in candidates:
        if not _node_satisfies(caps, row["requires"]):
            continue

        task = get_task(conn, row["task_id"])
        if not task or task["status"] not in ("running", "aggregating"):
            continue
        policy = _policy(task)

        max_parallel = policy.get("max_parallel")
        if max_parallel is not None:
            running = conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE task_id=? AND status='running'", (row["task_id"],)
            ).fetchone()["n"]
            if running >= max_parallel:
                continue

        max_nodes = policy.get("max_nodes")
        if max_nodes is not None:
            distinct_nodes = conn.execute(
                "SELECT COUNT(DISTINCT assigned_node) AS n FROM jobs WHERE task_id=? AND assigned_node IS NOT NULL",
                (row["task_id"],),
            ).fetchone()["n"]
            already_used = conn.execute(
                "SELECT 1 FROM jobs WHERE task_id=? AND assigned_node=? LIMIT 1", (row["task_id"], node_id)
            ).fetchone()
            if distinct_nodes >= max_nodes and not already_used:
                continue

        cur = conn.execute(
            "UPDATE jobs SET assigned_node=?, status='running', updated_at=? WHERE job_id=? AND assigned_node IS NULL",
            (node_id, time.time(), row["job_id"]),
        )
        if cur.rowcount == 1:
            return {"job_id": row["job_id"], "model": row["model"], "prompt": row["prompt"]}
        # Otro nodo reclamo este job puntual en el medio -- se prueba el
        # siguiente candidato de la lista.

    return None


# ----------------------------------------------------------------------
# Reaccion a que una subtarea termino
# ----------------------------------------------------------------------

def on_subtask_completed(conn, job, response_text: str | None):
    """
    Se llama desde `POST /jobs/{id}/result` cuando el job que acaba de
    reportar resultado tiene `task_id` (es una subtarea). Debe llamarse
    bajo el lock de esa tarea (`_lock_for_task()` en server/main.py) para
    que dos subtareas que terminan casi juntas no disparen la agregacion
    dos veces.

    Devuelve la fila de `tasks` YA FINALIZADA si esta llamada en
    particular fue la que completo/fallo/cancelo la tarea, o None si la
    tarea sigue en curso (o si esta llamada no hizo nada porque la tarea
    ya estaba en un estado terminal de antes -- resultado tardio de una
    subtarea cancelada). El caller usa ese valor para saber si corresponde
    actualizar la memoria de la sesion con el resultado agregado.
    """
    task = get_task(conn, job["task_id"])
    if not task or task["status"] in ("completed", "failed", "cancelled"):
        return None

    if job["subtask_key"] == AGGREGATE_KEY:
        if job["status"] == "completed":
            return _finalize_task(conn, task, success=True, result={"response": response_text})
        return _finalize_task(conn, task, success=False, error=job["error"] or "La sintesis final fallo")

    policy = _policy(task)
    max_retries = policy.get("max_retries", 0)

    if job["status"] == "failed":
        if (job["retry_count"] or 0) < max_retries:
            _requeue_retry(conn, job)
            return None
        if task["mode"] in ("batch", "pipeline", "map_reduce"):
            return _finalize_task(
                conn, task, success=False,
                error=f"Subtarea '{job['subtask_key']}' fallo sin reintentos disponibles",
            )
        if _all_subtasks_terminal(conn, task["task_id"]) and not _has_successful_subtask(conn, task["task_id"]):
            return _finalize_task(conn, task, success=False, error="Ninguna subtarea tuvo exito")
        return None

    # Exito: desbloquear a quien dependia de esta subtarea, y evaluar si
    # ya se cumple el criterio de agregacion de la tarea.
    _unblock_dependents(conn, task["task_id"], job["job_id"])

    if task["aggregation"] == "first_success":
        return _aggregate_and_finalize(conn, task, trigger_job=job, response_text=response_text)
    if task["aggregation"] == "vote" and not _consensus_reached(conn, task["task_id"]):
        return None
    if task["aggregation"] != "vote" and not _all_subtasks_terminal(conn, task["task_id"]):
        return None
    return _aggregate_and_finalize(conn, task, trigger_job=None, response_text=None)


def _all_subtasks_terminal(conn, task_id: str) -> bool:
    """True si ninguna subtarea "real" (se excluye el job de agregacion,
    que se maneja aparte) sigue en queued/blocked/running."""
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM jobs
           WHERE task_id=? AND status IN ('queued','blocked','running')
             AND (subtask_key IS NULL OR subtask_key != ?)""",
        (task_id, AGGREGATE_KEY),
    ).fetchone()
    return row["n"] == 0


def _has_successful_subtask(conn, task_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE task_id=? AND status='completed'", (task_id,)
    ).fetchone()
    return row["n"] > 0


def _requeue_retry(conn, job) -> None:
    """
    Reintento = una subtarea NUEVA con el mismo spec (mismo prompt,
    requires, dependencias ya resueltas) y `retry_count` incrementado,
    devuelta al pool compartido (`assigned_node=NULL`) para que la tome
    cualquier nodo disponible.

    Limitacion aceptada: no se excluye a proposito al nodo que fallo --
    en una red chica de pocos nodos, evitarlo podria dejar la subtarea
    sin ningun candidato posible. Si hace falta esa garantia en el
    futuro, se puede guardar una lista de nodos excluidos en el job.
    """
    new_id = str(uuid.uuid4())
    now = time.time()
    conn.execute(
        """INSERT INTO jobs(
             job_id,model,prompt,status,assigned_node,task_id,depends_on,requires,
             subtask_key,retry_count,created_at,updated_at
           ) VALUES(?,?,?,?,NULL,?,?,?,?,?,?,?)""",
        (
            new_id, job["model"], job["prompt"], "queued", job["task_id"],
            job["depends_on"], job["requires"], job["subtask_key"],
            (job["retry_count"] or 0) + 1, now, now,
        ),
    )


def _unblock_dependents(conn, task_id: str, completed_job_id: str) -> None:
    """
    Busca subtareas 'blocked' de esta tarea que dependian (entre otras)
    de `completed_job_id`. Si TODAS sus dependencias ya estan
    'completed', resuelve los placeholders `{{key.result}}` de su prompt
    (ver `resolve_template()`) y las pasa a 'queued' -- recien ahi quedan
    elegibles para que cualquier nodo las reclame.
    """
    blocked = conn.execute(
        "SELECT * FROM jobs WHERE task_id=? AND status='blocked'", (task_id,)
    ).fetchall()
    for row in blocked:
        dep_ids = json.loads(row["depends_on"] or "[]")
        if completed_job_id not in dep_ids:
            continue
        placeholders = ",".join("?" for _ in dep_ids)
        dep_rows = conn.execute(f"SELECT * FROM jobs WHERE job_id IN ({placeholders})", dep_ids).fetchall()
        if not all(d["status"] == "completed" for d in dep_rows):
            continue
        results_by_key = {d["subtask_key"]: _extract_response(d["result"]) for d in dep_rows}
        new_prompt = resolve_template(row["prompt"], results_by_key)
        conn.execute(
            "UPDATE jobs SET status='queued', prompt=?, updated_at=? WHERE job_id=?",
            (new_prompt, time.time(), row["job_id"]),
        )


def _consensus_reached(conn, task_id: str) -> bool:
    """Para aggregation='vote': True si ya hay mayoria estricta entre las
    subtareas completadas hasta ahora, o si ya no queda ninguna
    pendiente (para no esperar indefinidamente si nunca hay mayoria)."""
    if _all_subtasks_terminal(conn, task_id):
        return True
    total_expected = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE task_id=? AND (subtask_key IS NULL OR subtask_key != ?)",
        (task_id, AGGREGATE_KEY),
    ).fetchone()["n"]
    completed = conn.execute(
        "SELECT result FROM jobs WHERE task_id=? AND status='completed'", (task_id,)
    ).fetchall()
    if not completed or not total_expected:
        return False
    counts = Counter(_extract_response(r["result"]).strip().lower() for r in completed)
    top_count = counts.most_common(1)[0][1]
    return top_count > total_expected / 2


def _aggregate_and_finalize(conn, task, trigger_job, response_text: str | None):
    """
    Ejecuta la estrategia de agregacion de la tarea. Para todas menos
    'llm_synthesize' finaliza la tarea de una (devuelve la fila
    finalizada). Para 'llm_synthesize' en cambio ENCOLA un job de
    sintesis mas en el pool compartido (`subtask_key='__aggregate__'`) y
    deja la tarea en status='aggregating' -- se finaliza recien cuando
    ese job especial completa (ver `on_subtask_completed()`), devolviendo
    None por ahora.
    """
    strategy = task["aggregation"]
    task_id = task["task_id"]

    if strategy == "first_success":
        result = {"response": response_text, "source_subtask": trigger_job["subtask_key"]}
        return _finalize_task(conn, task, success=True, result=result)

    subtasks = [s for s in list_subtasks(conn, task_id) if s["subtask_key"] != AGGREGATE_KEY]
    completed = [s for s in subtasks if s["status"] == "completed"]
    by_key = {s["subtask_key"]: _extract_response(s["result"]) for s in completed}

    if strategy == "collect_all":
        return _finalize_task(conn, task, success=True, result={"results": by_key})

    if strategy == "concat":
        return _finalize_task(conn, task, success=True, result={"response": "\n\n".join(by_key.values())})

    if strategy == "vote":
        texts = [t.strip() for t in by_key.values() if t.strip()]
        if not texts:
            return _finalize_task(conn, task, success=False, error="Ninguna subtarea tuvo una respuesta valida para votar")
        counts = Counter(t.lower() for t in texts)
        winner_norm, votes = counts.most_common(1)[0]
        winner = next(t for t in texts if t.lower() == winner_norm)
        return _finalize_task(conn, task, success=True, result={
            "response": winner, "votes": votes, "total": len(texts), "results": by_key,
        })

    if strategy == "llm_synthesize":
        prompt = (
            "Tenes varias respuestas parciales a la misma tarea. Sintetiza una "
            "respuesta final unica, clara y coherente a partir de ellas:\n\n"
            + "\n\n".join(f"[{key}]: {text}" for key, text in by_key.items())
            + "\n\nRespuesta final:"
        )
        model = subtasks[0]["model"] if subtasks else "gemma3:4b"
        now = time.time()
        conn.execute(
            """INSERT INTO jobs(job_id,model,prompt,status,assigned_node,task_id,subtask_key,created_at,updated_at)
               VALUES(?,?,?,?,NULL,?,?,?,?)""",
            (str(uuid.uuid4()), model, prompt, "queued", task_id, AGGREGATE_KEY, now, now),
        )
        conn.execute("UPDATE tasks SET status='aggregating', updated_at=? WHERE task_id=?", (now, task_id))
        return None

    return _finalize_task(conn, task, success=False, error=f"Estrategia de agregacion desconocida: {strategy!r}")


def _finalize_task(conn, task, success: bool, result: dict | None = None, error: str | None = None):
    """Marca la tarea como completada o fallida, y cancela (blando)
    cualquier subtarea que haya quedado sin terminar -- una vez que la
    tarea tiene un resultado (o se rindio), nada mas deberia seguir
    corriendo en su nombre. Devuelve la fila de `tasks` ya actualizada."""
    now = time.time()
    status = "completed" if success else "failed"
    conn.execute(
        "UPDATE tasks SET status=?, result=?, error=?, updated_at=?, completed_at=? WHERE task_id=?",
        (status, json.dumps(result) if result is not None else None, error, now, now, task["task_id"]),
    )
    _cancel_remaining(conn, task["task_id"])
    return get_task(conn, task["task_id"])


def _cancel_remaining(conn, task_id: str) -> None:
    """Cancelacion BLANDA: marca 'cancelled' las subtareas que no habian
    terminado. Si algun nodo ya la tenia en ejecucion, no hay forma de
    interrumpirlo -- cuando (si) mande el resultado mas tarde, se ignora
    porque `on_subtask_completed()` corta apenas ve que la tarea ya esta
    en un estado terminal."""
    conn.execute(
        "UPDATE jobs SET status='cancelled', updated_at=? WHERE task_id=? AND status IN ('queued','blocked','running')",
        (time.time(), task_id),
    )


def check_timeout(conn, task) -> bool:
    """
    Chequeo perezoso de `policy.max_time_s`: no hay ningun proceso de
    fondo corriendo, asi que el timeout se evalua cada vez que algo
    consulta o toca la tarea (GET /tasks/{id}, y cada
    on_subtask_completed()). Si se paso del tiempo maximo, finaliza la
    tarea como fallida y cancela lo que quede pendiente. Devuelve True
    si esta llamada fue la que corto la tarea por timeout.
    """
    policy = _policy(task)
    max_time = policy.get("max_time_s")
    if not max_time or task["status"] in ("completed", "failed", "cancelled"):
        return False
    if time.time() - task["created_at"] > max_time:
        _finalize_task(conn, task, success=False, error=f"Excedio el tiempo maximo de la tarea ({max_time}s)")
        return True
    return False


def cancel_task(conn, task_id: str) -> bool:
    """Cancela una tarea a pedido (no por politica). False si no existe
    o si ya habia terminado."""
    task = get_task(conn, task_id)
    if not task or task["status"] in ("completed", "failed", "cancelled"):
        return False
    conn.execute(
        "UPDATE tasks SET status='cancelled', updated_at=?, completed_at=? WHERE task_id=?",
        (time.time(), time.time(), task_id),
    )
    _cancel_remaining(conn, task_id)
    return True
