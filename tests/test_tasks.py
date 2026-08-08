"""
Tests de integracion de tareas compuestas (Task Decomposition +
Aggregation) contra la API real del servidor, simulando nodos falsos.
Igual criterio que tests/test_sessions.py: no hace falta GPU ni Ollama,
lo que se prueba es que el SERVIDOR reparte, coordina y agrega
correctamente -- la calidad de una respuesta real de un modelo no es
parte de esto.
"""
import json


def _register_node(client, name, capabilities=None):
    r = client.post("/nodes/register", json={
        "name": name,
        "capabilities": json.dumps(capabilities or {"services": {"ollama": True}}),
    })
    assert r.status_code == 200
    return {"name": name, **r.json()}


def _heartbeat(client, node):
    r = client.post("/nodes/heartbeat", json={
        "node_id": node["node_id"], "token": node["token"], "status": "available",
    })
    assert r.status_code == 200


def _pull(client, node):
    r = client.post("/agent/pull", json={"node_id": node["node_id"], "token": node["token"]})
    assert r.status_code == 200
    return r.json()["job"]


def _report(client, node, job_id, success, response_text=None, error=None):
    r = client.post(f"/jobs/{job_id}/result", json={
        "node_id": node["node_id"], "token": node["token"], "success": success,
        "result": json.dumps({"type": "text", "response": response_text}) if success else None,
        "error": error, "duration_ms": 50.0,
    })
    assert r.status_code == 200


def test_batch_task_runs_four_subtasks_in_parallel_and_aggregates(client):
    """
    Caso minimo #1 del pedido: una tarea con 4 subtareas independientes
    se reparte entre 4 nodos disponibles, se ejecutan, y el resultado
    final trae las 4 con trazabilidad de que nodo hizo cada una.
    """
    nodes = [_register_node(client, f"node-batch-{i}") for i in range(4)]
    for node in nodes:
        _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [{"key": f"part{i}", "model": "gemma3:4b", "prompt": f"parte {i}"} for i in range(4)],
    })
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert len(r.json()["subtasks"]) == 4

    # Cada nodo reclama UNA subtarea del pool compartido (ninguna tenia
    # un nodo asignado de antemano -- las 4 arrancaron assigned_node=NULL).
    claimed = {}
    for node in nodes:
        job = _pull(client, node)
        assert job is not None
        claimed[node["name"]] = job

    job_ids = {j["job_id"] for j in claimed.values()}
    assert len(job_ids) == 4, "cada nodo tiene que haberse llevado una subtarea DISTINTA"

    for node in nodes:
        _report(client, node, claimed[node["name"]]["job_id"], success=True,
                response_text=f"resultado de {node['name']}")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "completed"
    results = json.loads(task["result"])["results"]
    assert len(results) == 4
    assert task["metrics"]["subtask_count"] == 4
    assert task["metrics"]["succeeded"] == 4
    assert task["metrics"]["nodes_used"] == 4

    subtasks = client.get(f"/tasks/{task_id}/subtasks").json()["subtasks"]
    executed_by = {s["subtask_key"]: s["node_name"] for s in subtasks}
    assert len(set(executed_by.values())) == 4, "trazabilidad: 4 subtareas, 4 nodos distintos"


def test_fan_out_first_success_uses_first_and_cancels_rest(client):
    """
    Caso minimo #2 del pedido: la misma pregunta a 3 nodos; se usa la
    primera respuesta valida y se cancelan (blando) las demas.
    """
    nodes = [_register_node(client, f"node-fanout-{i}") for i in range(3)]
    for node in nodes:
        _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "fan_out",
        "subtasks": [{"key": f"try{i}", "model": "gemma3:4b", "prompt": "la misma pregunta"} for i in range(3)],
    })
    task_id = r.json()["task_id"]

    jobs = {node["name"]: _pull(client, node) for node in nodes}

    winner = nodes[0]
    _report(client, winner, jobs[winner["name"]]["job_id"], success=True, response_text="la ganadora")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "completed"
    assert json.loads(task["result"])["response"] == "la ganadora"

    subtasks = client.get(f"/tasks/{task_id}/subtasks").json()["subtasks"]
    statuses = {s["subtask_key"]: s["status"] for s in subtasks}
    assert statuses["try0"] == "completed"
    assert statuses["try1"] == "cancelled"
    assert statuses["try2"] == "cancelled"

    # Resultados tardios de los perdedores tienen que ignorarse sin romper nada.
    for node in nodes[1:]:
        r = client.post(f"/jobs/{jobs[node['name']]['job_id']}/result", json={
            "node_id": node["node_id"], "token": node["token"], "success": True,
            "result": json.dumps({"type": "text", "response": "tarde"}),
            "duration_ms": 999.0,
        })
        assert r.status_code == 200

    task_after = client.get(f"/tasks/{task_id}").json()
    assert json.loads(task_after["result"])["response"] == "la ganadora"


def test_pipeline_stage_waits_for_dependency_and_resolves_template(client):
    """Una subtarea 'blocked' se desbloquea recien cuando su dependencia
    termina, y su prompt se resuelve sustituyendo {{key.result}}."""
    node = _register_node(client, "node-pipeline")
    _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "pipeline",
        "subtasks": [
            {"key": "A", "model": "gemma3:4b", "prompt": "elegi un tema"},
            {"key": "B", "model": "gemma3:4b", "prompt": "expandi esto: {{A.result}}", "depends_on": ["A"]},
        ],
    })
    task_id = r.json()["task_id"]

    subtasks = client.get(f"/tasks/{task_id}/subtasks").json()["subtasks"]
    statuses = {s["subtask_key"]: s["status"] for s in subtasks}
    assert statuses["A"] == "queued"
    assert statuses["B"] == "blocked"

    job_a = _pull(client, node)
    assert job_a["prompt"] == "elegi un tema"
    _report(client, node, job_a["job_id"], success=True, response_text="gatos")

    subtasks = client.get(f"/tasks/{task_id}/subtasks").json()["subtasks"]
    b = next(s for s in subtasks if s["subtask_key"] == "B")
    assert b["status"] == "queued"
    assert b["prompt"] == "expandi esto: gatos"

    job_b = _pull(client, node)
    assert job_b["job_id"] == b["job_id"]
    _report(client, node, job_b["job_id"], success=True, response_text="los gatos son geniales")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "completed"
    assert json.loads(task["result"])["results"] == {"A": "gatos", "B": "los gatos son geniales"}


def test_failed_subtask_is_retried_on_another_node(client):
    node_a = _register_node(client, "node-fail")
    node_b = _register_node(client, "node-retry")
    _heartbeat(client, node_a)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [{"key": "solo", "model": "gemma3:4b", "prompt": "algo"}],
        "policy": {"max_retries": 1},
    })
    task_id = r.json()["task_id"]

    job1 = _pull(client, node_a)
    _report(client, node_a, job1["job_id"], success=False, error="boom")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "running", "todavia le quedaba un reintento, no deberia haberse rendido"

    _heartbeat(client, node_b)
    job2 = _pull(client, node_b)
    assert job2 is not None
    assert job2["job_id"] != job1["job_id"], "el reintento tiene que ser una subtarea NUEVA"
    _report(client, node_b, job2["job_id"], success=True, response_text="funciono en el segundo intento")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "completed"
    assert task["metrics"]["retries"] == 1


def test_failed_subtask_without_retries_fails_the_whole_batch_task(client):
    node = _register_node(client, "node-nofail-retry")
    _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [
            {"key": "ok", "model": "gemma3:4b", "prompt": "esto va a andar"},
            {"key": "mal", "model": "gemma3:4b", "prompt": "esto va a fallar"},
        ],
    })
    task_id = r.json()["task_id"]

    job1 = _pull(client, node)
    _heartbeat(client, node)
    job2 = _pull(client, node)
    ok_job = job1 if job1["prompt"] == "esto va a andar" else job2
    bad_job = job2 if ok_job is job1 else job1

    _report(client, node, bad_job["job_id"], success=False, error="fallo permanente")
    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "failed"

    # Un resultado tardio de la subtarea "ok" (que nunca reporto nada en
    # este test) se ignora -- la tarea ya esta en un estado terminal.
    _report(client, node, ok_job["job_id"], success=True, response_text="tarde")
    task_after = client.get(f"/tasks/{task_id}").json()
    assert task_after["status"] == "failed"


def test_node_capability_requirements_are_respected(client):
    """Una subtarea que pide requires={"service":"comfyui"} no deberia
    poder ser tomada por un nodo que solo tiene Ollama."""
    ollama_only = _register_node(client, "node-ollama-only", {"services": {"ollama": True}})
    comfy_node = _register_node(client, "node-comfy", {"services": {"comfyui": True}})
    _heartbeat(client, ollama_only)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [{
            "key": "img", "model": "flux1-schnell-fp8", "prompt": "un gato",
            "requires": {"service": "comfyui"},
        }],
    })
    task_id = r.json()["task_id"]

    assert _pull(client, ollama_only) is None, "no cumple el requires, no deberia poder reclamarla"

    _heartbeat(client, comfy_node)
    job = _pull(client, comfy_node)
    assert job is not None
    _report(client, comfy_node, job["job_id"], success=True, response_text="imagen generada")

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "completed"


def test_max_parallel_limits_concurrent_claims(client):
    nodes = [_register_node(client, f"node-limit-{i}") for i in range(3)]
    for node in nodes:
        _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [{"key": f"p{i}", "model": "gemma3:4b", "prompt": f"p{i}"} for i in range(3)],
        "policy": {"max_parallel": 1},
    })
    task_id = r.json()["task_id"]

    job1 = _pull(client, nodes[0])
    assert job1 is not None
    assert _pull(client, nodes[1]) is None, "max_parallel=1 y ya hay una corriendo"
    assert _pull(client, nodes[2]) is None

    _report(client, nodes[0], job1["job_id"], success=True, response_text="uno")
    job2 = _pull(client, nodes[1])
    assert job2 is not None, "se libero un lugar, ahora si deberia poder reclamar"


def test_cancel_task_endpoint_stops_pending_subtasks(client):
    node = _register_node(client, "node-cancel")
    _heartbeat(client, node)

    r = client.post("/tasks", json={
        "mode": "batch",
        "subtasks": [{"key": "a", "model": "gemma3:4b", "prompt": "algo"}],
    })
    task_id = r.json()["task_id"]

    assert client.post(f"/tasks/{task_id}/cancel").status_code == 200

    task = client.get(f"/tasks/{task_id}").json()
    assert task["status"] == "cancelled"
    assert _pull(client, node) is None


def test_task_with_unknown_mode_is_rejected(client):
    r = client.post("/tasks", json={
        "mode": "no-existe",
        "subtasks": [{"key": "a", "model": "gemma3:4b", "prompt": "algo"}],
    })
    assert r.status_code == 400


def test_task_result_updates_session_memory_only_once_on_completion(client):
    """El resultado AGREGADO de la tarea (no cada subtarea individual)
    es lo unico que se guarda en la memoria de la sesion, y solo cuando
    la tarea entera termina."""
    nodes = [_register_node(client, f"node-sess-{i}") for i in range(2)]
    for node in nodes:
        _heartbeat(client, node)

    session_id = client.post("/sessions", json={}).json()["session_id"]

    r = client.post("/tasks", json={
        "mode": "batch",
        "session_id": session_id,
        "subtasks": [{"key": f"p{i}", "model": "gemma3:4b", "prompt": f"parte {i}"} for i in range(2)],
    })
    task_id = r.json()["task_id"]
    jobs = [_pull(client, node) for node in nodes]

    _report(client, nodes[0], jobs[0]["job_id"], success=True, response_text="uno")
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    assert not any(m["role"] == "assistant" for m in messages), "la tarea no termino, no deberia haber mensaje todavia"

    _report(client, nodes[1], jobs[1]["job_id"], success=True, response_text="dos")
    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant_msgs) == 1, "recien al completar la tarea se guarda UN mensaje con el resultado agregado"


def test_simple_infer_is_unaffected_by_the_tasks_feature(client):
    """Regresion: /infer sin task_id sigue funcionando exactamente igual
    que antes (test_sessions.py ya cubre esto a fondo; aca solo se
    confirma que las columnas nuevas de jobs no rompen el flujo normal)."""
    node = _register_node(client, "node-plain")
    _heartbeat(client, node)

    r = client.post("/infer", json={"model": "gemma3:4b", "prompt": "Hola"})
    assert r.status_code == 200
    job_full = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job_full["task_id"] is None
    assert job_full["prompt"] == "Hola"
