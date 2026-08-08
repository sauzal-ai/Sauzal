"""
Tests de integracion de sesiones/memoria contra la API real del servidor
(via FastAPI TestClient), simulando nodos falsos. No hace falta GPU ni
Ollama para correr esto: lo que se prueba es que el SERVIDOR arma y
persiste el contexto correcto, no la calidad de una respuesta de un
modelo real -- esa frontera de responsabilidad es justamente la que pide
el diseño ("el contexto lo administra Sauzal, no el nodo").
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import memory


def _register_node(client, name):
    r = client.post("/nodes/register", json={
        "name": name,
        "capabilities": json.dumps({"services": {"ollama": True}}),
    })
    assert r.status_code == 200
    return {"name": name, **r.json()}  # {"name": ..., "node_id": ..., "token": ...}


def _heartbeat(client, node):
    r = client.post("/nodes/heartbeat", json={
        "node_id": node["node_id"], "token": node["token"], "status": "available",
    })
    assert r.status_code == 200


def _complete_job(client, node, job_id, response_text):
    r = client.post(f"/jobs/{job_id}/result", json={
        "node_id": node["node_id"], "token": node["token"], "success": True,
        "result": json.dumps({
            "type": "text", "response": response_text,
            "prompt_tokens": 10, "output_tokens": 5,
        }),
        "duration_ms": 123.4,
    })
    assert r.status_code == 200


def test_stateless_infer_is_unaffected_by_sessions(client):
    """Un pedido SIN session_id tiene que seguir funcionando exactamente
    igual que antes de que existiera este modulo: el prompt no se toca."""
    node = _register_node(client, "node-stateless")
    _heartbeat(client, node)

    r = client.post("/infer", json={"model": "gemma3:4b", "prompt": "Hola"})
    assert r.status_code == 200
    job = r.json()
    assert job["assigned_node"] == node["node_id"]

    job_full = client.get(f"/jobs/{job['job_id']}").json()
    assert job_full["prompt"] == "Hola"
    assert job_full["session_id"] is None
    assert job_full["context_tokens_estimate"] is None


def test_session_lifecycle_create_get_delete(client):
    r = client.post("/sessions", json={})
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    info = client.get(f"/sessions/{session_id}")
    assert info.status_code == 200
    assert info.json()["message_count"] == 0

    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_infer_with_unknown_session_id_returns_404(client):
    node = _register_node(client, "node-x")
    _heartbeat(client, node)
    r = client.post("/infer", json={
        "model": "gemma3:4b", "prompt": "Hola", "session_id": "no-existe",
    })
    assert r.status_code == 404


def test_conversation_survives_across_different_nodes(client):
    """
    El caso central del pedido: se cuenta un dato en el nodo A: varios
    turnos de relleno despues (para sacar el dato de la ventana de corto
    plazo), se pregunta por ese dato y el pedido lo atiende el nodo B --
    el prompt que arma el servidor para el nodo B tiene que traer el dato
    de todas formas, sin que el nodo A haya tenido que guardar nada.
    """
    node_a = _register_node(client, "node-a")
    node_b = _register_node(client, "node-b")
    session_id = client.post("/sessions", json={}).json()["session_id"]

    # Turno 1, atendido por el nodo A (el unico disponible en este punto).
    _heartbeat(client, node_a)
    r = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id,
        "prompt": "Mi auto favorito es un BMW Isetta",
    })
    job1 = r.json()
    assert job1["assigned_node"] == node_a["node_id"]
    _complete_job(client, node_a, job1["job_id"], "Que interesante, el BMW Isetta es un microauto clasico.")

    # Turnos de relleno (siguen en el nodo A), para que el dato salga de
    # la ventana de corto plazo (memory.SHORT_TERM_WINDOW = 6 mensajes).
    for i in range(4):
        _heartbeat(client, node_a)
        r = client.post("/infer", json={
            "model": "gemma3:4b", "session_id": session_id,
            "prompt": f"Contame algo sin relacion, numero {i}",
        })
        _complete_job(client, node_a, r.json()["job_id"], f"Respuesta de relleno {i}.")

    # Se pausa el nodo A a proposito, para que el siguiente turno SI o SI
    # tenga que ir a un nodo distinto (asi el test no depende de timing).
    client.post(f"/admin/nodes/{node_a['node_id']}/pause")
    _heartbeat(client, node_b)

    r = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id,
        "prompt": "Cual es mi auto favorito?",
    })
    assert r.status_code == 200
    job2 = r.json()
    assert job2["assigned_node"] == node_b["node_id"], "el segundo turno tiene que poder ir a un nodo distinto"

    job2_full = client.get(f"/jobs/{job2['job_id']}").json()
    assert "BMW Isetta" in job2_full["prompt"], "el servidor le tiene que haber inyectado el dato al nodo B"
    assert job2_full["context_semantic_hits"] >= 1


def test_sessions_are_isolated_from_each_other(client):
    node = _register_node(client, "node-iso")
    _heartbeat(client, node)

    s1 = client.post("/sessions", json={}).json()["session_id"]
    s2 = client.post("/sessions", json={}).json()["session_id"]

    r = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": s1, "prompt": "Mi color favorito es el verde",
    })
    _complete_job(client, node, r.json()["job_id"], "Que lindo el verde.")

    _heartbeat(client, node)
    r = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": s2, "prompt": "Cual es mi color favorito?",
    })
    job_full = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert "verde" not in job_full["prompt"].lower(), "la sesion 2 no deberia ver nada de la sesion 1"


def test_image_model_does_not_get_conversational_context_injected(client):
    """Para modelos de imagen, no tiene sentido inyectar una
    transcripcion de texto en el prompt -- se omite la memoria, aunque
    el mensaje se guarda igual para que el historial quede completo."""
    node = _register_node(client, "node-img")
    _heartbeat(client, node)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    r = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id,
        "prompt": "Mi auto favorito es un BMW Isetta",
    })
    _complete_job(client, node, r.json()["job_id"], "Que interesante.")

    _heartbeat(client, node)
    r = client.post("/infer", json={
        "model": "flux1-schnell-fp8", "session_id": session_id, "prompt": "un auto rojo",
    })
    job_full = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert job_full["prompt"] == "un auto rojo"


def test_long_conversation_triggers_automatic_summary(client):
    """
    Cuando se acumulan demasiados mensajes viejos sin resumir
    (memory.SUMMARIZE_TRIGGER), el servidor tiene que encolar SOLO UN
    job interno kind='summarize' (no uno por turno). Al completarse (via
    el mismo POST /jobs/{id}/result de siempre -- el agente no distingue
    nada especial), el resumen queda guardado en la sesion y los
    mensajes que cubria quedan marcados como resumidos.
    """
    node = _register_node(client, "node-summary")
    session_id = client.post("/sessions", json={}).json()["session_id"]

    for i in range(12):
        _heartbeat(client, node)
        r = client.post("/infer", json={
            "model": "gemma3:4b", "session_id": session_id,
            "prompt": f"Mensaje numero {i} con contenido variado para no ser trivial",
        })
        _complete_job(client, node, r.json()["job_id"], f"Respuesta numero {i}.")

    # El resumen automatico quedo encolado para el mismo nodo (es el
    # unico disponible); lo "pulleamos" tal como haria el agente real.
    _heartbeat(client, node)
    pulled = client.post("/agent/pull", json={
        "node_id": node["node_id"], "token": node["token"],
    }).json()
    job = pulled["job"]
    assert job is not None, "tendria que haberse encolado un job de resumen automatico"

    _complete_job(client, node, job["job_id"], "Resumen: se hablo de varios temas variados sin nada en particular.")

    info = client.get(f"/sessions/{session_id}").json()
    assert info["summary"] and "Resumen" in info["summary"]

    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    summarized_count = sum(1 for m in messages if m["summarized"])
    assert summarized_count >= memory.SUMMARIZE_TRIGGER

    # No debe haber quedado ningun otro job de resumen duplicado encolado.
    _heartbeat(client, node)
    pulled_again = client.post("/agent/pull", json={
        "node_id": node["node_id"], "token": node["token"],
    }).json()
    assert pulled_again["job"] is None


def test_delete_session_via_admin_panel_removes_it(client):
    node = _register_node(client, "node-admin-del")
    _heartbeat(client, node)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    client.post("/admin/sessions/{}/delete".format(session_id))
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_multiple_sessions_run_concurrently_without_cross_contamination(client):
    """
    Un mismo cliente tiene que poder mantener N sesiones activas al
    mismo tiempo: dispara varias conversaciones con THREADS REALES (no
    una despues de la otra) y verifica que ninguna vea el contexto de
    las demas, y que ninguna request falle por contencion de la base
    (esto es lo que habilita el WAL mode de startup()).

    Cada conversacion usa su propio nodo dedicado para que la
    disponibilidad de nodos no sea un cuello de botella artificial que
    confunda la prueba de concurrencia real entre sesiones.
    """
    facts = [
        ("verde", "Mi color favorito es el verde"),
        ("Isetta", "Mi auto favorito es un BMW Isetta"),
        ("ajedrez", "Mi hobby favorito es el ajedrez"),
        ("pizza", "Mi comida favorita es la pizza"),
    ]
    nodes = [_register_node(client, f"node-parallel-{i}") for i in range(len(facts))]
    for node in nodes:
        _heartbeat(client, node)
    session_ids = [client.post("/sessions", json={}).json()["session_id"] for _ in facts]

    def run_conversation(node, session_id, fact_text):
        r = client.post("/infer", json={
            "model": "gemma3:4b", "session_id": session_id,
            "prompt": fact_text, "node": node["name"],
        })
        assert r.status_code == 200, r.text
        job = r.json()
        assert job["assigned_node"] == node["node_id"]
        _complete_job(client, node, job["job_id"], f"Anotado: {fact_text}")

    with ThreadPoolExecutor(max_workers=len(facts)) as pool:
        futures = [
            pool.submit(run_conversation, node, session_id, fact_text)
            for node, session_id, (_, fact_text) in zip(nodes, session_ids, facts)
        ]
        for f in futures:
            f.result()  # relanza cualquier excepcion/assert que haya pasado en el thread

    # Cada sesion tiene que ver SOLO su propio dato, nada de las demas.
    for session_id, (keyword, _fact_text) in zip(session_ids, facts):
        messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
        contents = " ".join(m["content"] for m in messages)
        assert keyword in contents
        for other_keyword, _ in facts:
            if other_keyword != keyword:
                assert other_keyword not in contents


def test_same_session_concurrent_requests_do_not_corrupt_history(client):
    """
    Si el cliente dispara dos pedidos de LA MISMA sesion en paralelo
    (por ejemplo, sin esperar la respuesta del primero), el lock por
    sesion (`_lock_for_session`) tiene que evitar que se pisen: los dos
    mensajes de usuario tienen que terminar guardados, sin perder
    ninguno y sin que el servidor tire una excepcion.
    """
    node_a = _register_node(client, "node-concurrent-a")
    node_b = _register_node(client, "node-concurrent-b")
    _heartbeat(client, node_a)
    _heartbeat(client, node_b)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    def send(node, i):
        r = client.post("/infer", json={
            "model": "gemma3:4b", "session_id": session_id,
            "prompt": f"Mensaje concurrente numero {i}", "node": node["name"],
        })
        assert r.status_code == 200, r.text
        return r.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(lambda args: send(*args), [(node_a, 1), (node_b, 2)]))

    for job, node in zip(jobs, (node_a, node_b)):
        _complete_job(client, node, job["job_id"], f"Respuesta a {job['job_id'][:6]}")

    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assert "Mensaje concurrente numero 1" in user_msgs
    assert "Mensaje concurrente numero 2" in user_msgs
    assert len(user_msgs) == 2, "ningun mensaje deberia perderse ni pisarse"


def test_immediate_retry_with_same_prompt_returns_same_job(client):
    """
    Si el cliente reintenta el mismo pedido (timeout de red, doble
    click) -- mismo session_id, mismo texto, casi al instante -- el
    servidor tiene que devolver el job que ya existia, sin crear un
    segundo job ni duplicar el mensaje en el historial. Esto se prueba
    a proposito SIN completar el primer job (el nodo sigue "busy"): si
    el dedupe no funcionara, la segunda llamada fallaria con 503 por no
    haber ningun nodo disponible.
    """
    node = _register_node(client, "node-dedupe")
    _heartbeat(client, node)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    r1 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Hola, como estas?",
    })
    assert r1.status_code == 200
    job1 = r1.json()
    assert not job1.get("deduplicated")

    r2 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Hola, como estas?",
    })
    assert r2.status_code == 200
    job2 = r2.json()

    assert job2["job_id"] == job1["job_id"]
    assert job2.get("deduplicated") is True

    messages = client.get(f"/sessions/{session_id}/messages").json()["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 1, "no deberia haberse duplicado el mensaje"


def test_different_prompt_in_same_session_is_not_deduplicated(client):
    node = _register_node(client, "node-dedupe-diff")
    _heartbeat(client, node)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    r1 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Primera pregunta",
    })
    _complete_job(client, node, r1.json()["job_id"], "Respuesta 1")

    _heartbeat(client, node)
    r2 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Segunda pregunta, distinta",
    })

    assert r1.json()["job_id"] != r2.json()["job_id"]
    assert not r2.json().get("deduplicated")


def test_same_prompt_outside_dedupe_window_runs_again(client, monkeypatch):
    """Pasada la ventana de deduplicacion, la misma pregunta se ejecuta
    de cero -- esto NO es un cache de respuestas."""
    import time as time_module

    from server import main as server_main
    monkeypatch.setattr(server_main, "DEDUPE_WINDOW_SECONDS", 0.05)

    node = _register_node(client, "node-dedupe-window")
    _heartbeat(client, node)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    r1 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Que hora es?",
    })
    _complete_job(client, node, r1.json()["job_id"], "Respuesta 1")
    time_module.sleep(0.15)

    _heartbeat(client, node)
    r2 = client.post("/infer", json={
        "model": "gemma3:4b", "session_id": session_id, "prompt": "Que hora es?",
    })

    assert r1.json()["job_id"] != r2.json()["job_id"]
    assert not r2.json().get("deduplicated")
