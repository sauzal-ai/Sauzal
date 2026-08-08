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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import memory


def _register_node(client, name):
    r = client.post("/nodes/register", json={
        "name": name,
        "capabilities": json.dumps({"services": {"ollama": True}}),
    })
    assert r.status_code == 200
    return r.json()  # {"node_id": ..., "token": ...}


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
