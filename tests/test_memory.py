"""
Tests unitarios de server/memory.py: estimacion de tokens, deteccion de
mensajes triviales, y sobre todo la recuperacion semantica (top_similar)
y el armado de contexto (build_context), que es el corazon de la memoria
de conversacion de Sauzal. No necesitan la API HTTP ni una base real: se
usa una base SQLite en memoria con el esquema minimo de sessions/messages.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import memory


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE sessions(session_id TEXT PRIMARY KEY, tenant_id TEXT,
          created_at REAL, last_used_at REAL, summary TEXT);
        CREATE TABLE messages(message_id TEXT PRIMARY KEY, session_id TEXT,
          role TEXT, content TEXT, created_at REAL,
          trivial INTEGER DEFAULT 0, token_estimate INTEGER,
          summarized INTEGER DEFAULT 0);
    """)
    return conn


def test_estimate_tokens_is_positive_and_scales_with_length():
    assert memory.estimate_tokens("hola") >= 1
    assert memory.estimate_tokens("a" * 400) > memory.estimate_tokens("a" * 40)


def test_is_trivial_detects_greetings_and_short_replies():
    assert memory.is_trivial("hola")
    assert memory.is_trivial("gracias")
    assert memory.is_trivial("ok")
    assert memory.is_trivial("")
    assert not memory.is_trivial("Mi auto favorito es un BMW Isetta")


def test_top_similar_finds_the_relevant_fact_among_unrelated_messages():
    """Caso central del pedido: dado un mensaje nuevo, la busqueda
    semantica tiene que traer el mensaje viejo relacionado (mismo tema)
    por encima de mensajes viejos sin relacion."""
    candidates = [
        ("m1", "Mi auto favorito es un BMW Isetta"),
        ("m2", "El clima hoy esta nublado"),
        ("m3", "Me gusta jugar al ajedrez los domingos"),
    ]
    hits = memory.top_similar("Cual es mi auto favorito?", candidates, k=1)
    assert hits
    assert hits[0][0] == "m1"


def test_top_similar_returns_nothing_below_min_score():
    candidates = [("m1", "El clima hoy esta nublado")]
    hits = memory.top_similar("Cual es mi auto favorito?", candidates)
    assert hits == []


def test_top_similar_with_no_candidates_returns_empty_list():
    assert memory.top_similar("cualquier cosa", []) == []


def test_build_context_includes_summary_and_semantic_hit():
    conn = _conn()
    session_id = memory.create_session(conn)
    # El hecho relevante, y despues varios turnos de relleno para
    # sacarlo de la ventana de corto plazo (SHORT_TERM_WINDOW).
    memory.add_message(conn, session_id, "user", "Mi auto favorito es un BMW Isetta")
    memory.add_message(conn, session_id, "assistant", "Que interesante, el BMW Isetta es un microauto clasico.")
    for i in range(4):
        memory.add_message(conn, session_id, "user", f"Relleno numero {i}, sin relacion")
        memory.add_message(conn, session_id, "assistant", f"Respuesta de relleno {i}")

    prompt, metrics = memory.build_context(conn, session_id, "Cual es mi auto favorito?")

    assert "BMW Isetta" in prompt
    assert metrics["context_semantic_hits"] >= 1
    assert metrics["context_tokens_estimate"] > 0


def test_build_context_does_not_need_semantic_hit_when_still_in_short_term_window():
    conn = _conn()
    session_id = memory.create_session(conn)
    memory.add_message(conn, session_id, "user", "Mi auto favorito es un BMW Isetta")
    # Sin relleno: el mensaje sigue DENTRO de la ventana de corto plazo,
    # ya va a aparecer en la transcripcion reciente sin hacer falta
    # "recuperarlo" aparte via busqueda semantica.
    prompt, metrics = memory.build_context(conn, session_id, "Cual es mi auto favorito?")
    assert "BMW Isetta" in prompt
    assert metrics["context_semantic_hits"] == 0


def test_new_message_is_not_persisted_by_build_context():
    """build_context() no debe guardar el mensaje nuevo -- eso es
    responsabilidad del caller, DESPUES de armar el contexto (ver el
    aviso de orden en el docstring de la funcion)."""
    conn = _conn()
    session_id = memory.create_session(conn)
    memory.build_context(conn, session_id, "Mensaje que todavia no se guardo")
    assert memory.list_messages(conn, session_id) == []


def test_trivial_messages_are_excluded_from_summary_candidates_but_kept_in_history():
    conn = _conn()
    session_id = memory.create_session(conn)
    memory.add_message(conn, session_id, "user", "hola")
    for i in range(6):
        memory.add_message(conn, session_id, "user", f"Relleno {i}")
        memory.add_message(conn, session_id, "assistant", f"Respuesta {i}")

    all_messages = memory.list_messages(conn, session_id)
    assert any(m["content"] == "hola" for m in all_messages)  # se guardo igual

    pending = memory.messages_needing_summary(conn, session_id)
    hola_pending = [m for m in pending if m["content"] == "hola"]
    assert hola_pending and hola_pending[0]["trivial"] == 1


def test_delete_session_removes_messages_too():
    conn = _conn()
    session_id = memory.create_session(conn)
    memory.add_message(conn, session_id, "user", "hola")
    memory.delete_session(conn, session_id)
    assert memory.get_session(conn, session_id) is None
    assert memory.list_messages(conn, session_id) == []
