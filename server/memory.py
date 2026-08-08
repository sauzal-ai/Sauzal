"""
Sauzal · Memoria de conversacion (sesiones, mensajes, recuperacion semantica)
================================================================================

Este modulo es lo que le permite a una conversacion mantener contexto aunque
cada mensaje lo procese un nodo/GPU distinto: TODO el estado vive aca, del
lado del SERVIDOR, nunca en el agente. El agente sigue recibiendo un prompt
de texto plano como siempre (agent/agent.py no cambia en absoluto para
soportar esto) -- lo unico que cambia es que ese prompt, cuando el pedido
pertenece a una sesion, lo arma este modulo en vez de ser el texto crudo
que mando el cliente. Para el nodo, cada job sigue siendo una request de
texto suelta sin memoria de nada.

Piezas:
  - Tablas `sessions` y `messages` (creadas/migradas en server/main.py::startup()).
  - `build_context()`: dado un session_id y el mensaje nuevo del usuario,
    arma el prompt final a mandarle al modelo, combinando:
      1. El resumen de la sesion, si ya se genero uno (memoria de largo plazo).
      2. Los mensajes semanticamente mas relevantes de ANTES de la ventana
         reciente (memoria semantica/vectorial).
      3. La ventana de mensajes mas recientes, tal cual (memoria de corto plazo).
      4. El mensaje nuevo del usuario.
  - `messages_needing_summary()` + `build_summary_prompt()`: la mitad
    "decisoria" del resumen automatico. La mitad "ejecutora" (elegir un
    nodo y encolar el job) vive en server/main.py, porque ese es el
    dominio de la tabla `jobs` -- este modulo no sabe nada de nodos.

Motor de similitud semantica
-----------------------------
No se usa un modelo de embeddings neuronal: se evito a proposito agregarle
al SERVIDOR una dependencia pesada (PyTorch/sentence-transformers), que
ademas no tendria sentido si el control plane corre en una maquina sin
GPU. En cambio se usa TF-IDF + similitud coseno, implementado en Python
puro (sin librerias nuevas): es el modelo vectorial clasico de
recuperacion de informacion, funciona perfecto para buscar DENTRO de una
sola conversacion (un espacio de busqueda chico -- decenas o cientos de
mensajes, no millones), y se recalcula al vuelo en cada busqueda sin
necesidad de persistir ni cachear vectores.

Si en el futuro hace falta mas precision semantica, alcanza con
reemplazar `_vectorize()` por embeddings reales (por ejemplo, pidiendoselos
a algun nodo via el /api/embeddings de Ollama) sin tocar el resto de este
modulo ni la logica de server/main.py que lo llama -- toda la busqueda pasa
por `top_similar()`.
"""

from __future__ import annotations
import math, re, time, uuid
from collections import Counter

# ----------------------------------------------------------------------
# Configuracion (numeros elegidos para una conversacion tipica de chat;
# no hay nada magico, son los puntos de ajuste si hace falta afinar).
# ----------------------------------------------------------------------

SHORT_TERM_WINDOW = 6       # mensajes recientes (usuario+asistente) que se incluyen siempre, tal cual
SEMANTIC_TOP_K = 3          # maximo de mensajes viejos relevantes a recuperar
SEMANTIC_MIN_SCORE = 0.05   # similitud minima para considerar relevante un mensaje viejo (evita ruido)
SUMMARIZE_TRIGGER = 12      # a partir de cuantos mensajes viejos sin resumir se dispara un resumen automatico
TRIVIAL_MAX_WORDS = 3       # mensajes con esta cantidad de palabras o menos no entran al banco semantico

_TRIVIAL_PHRASES = {
    "ok", "okay", "listo", "gracias", "de nada", "hola", "chau", "bien",
    "genial", "perfecto", "dale", "si", "sí", "no", "entendido", "bueno",
}

_WORD_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    """
    Estimacion aproximada de tokens (no un conteo exacto: el servidor no
    tiene el tokenizador real de cada modelo). Se usa la heuristica
    estandar de ~4 caracteres por token -- alcanza para las metricas de
    "cuanto contexto se esta usando" que pide este feature, no para
    facturacion ni limites duros.
    """
    return max(1, len(text) // 4)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def is_trivial(text: str) -> bool:
    """
    True si un mensaje no aporta informacion util para guardar como
    memoria de largo plazo (saludos, confirmaciones cortas, etc). Estos
    mensajes IGUAL se guardan en `messages` (hacen falta para la memoria
    de corto plazo / continuidad inmediata de la charla), pero se
    excluyen del banco de candidatos para la busqueda semantica, para no
    llenarlo de ruido irrelevante.
    """
    clean = (text or "").strip().lower().strip(".!¡?¿")
    if not clean:
        return True
    if clean in _TRIVIAL_PHRASES:
        return True
    return len(_tokenize(text)) <= TRIVIAL_MAX_WORDS


def _vectorize(docs: list[list[str]]) -> list[dict[str, float]]:
    """
    TF-IDF clasico sobre una lista de documentos ya tokenizados. Devuelve
    un vector (dict termino->peso) por documento. Se recalcula desde cero
    en cada busqueda -- el corpus es chico (los mensajes de UNA sesion),
    asi que no hace falta persistir ni cachear vectores.
    """
    n_docs = len(docs)
    df = Counter()
    for tokens in docs:
        for term in set(tokens):
            df[term] += 1
    idf = {term: math.log((n_docs + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    vectors = []
    for tokens in docs:
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        vectors.append({term: (count / total) * idf.get(term, 0.0) for term, count in tf.items()})
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def top_similar(
    query: str,
    candidates: list[tuple[str, str]],
    k: int = SEMANTIC_TOP_K,
    min_score: float = SEMANTIC_MIN_SCORE,
) -> list[tuple[str, str, float]]:
    """
    Dado un texto de consulta y una lista de candidatos [(id, texto), ...],
    devuelve los `k` mas similares a la consulta (por coseno sobre
    TF-IDF), como [(id, texto, score), ...], ordenados de mas a menos
    relevante. Descarta los que quedan por debajo de `min_score` (evita
    traer "el menos distinto de un lote de nada relacionado").
    """
    if not candidates:
        return []
    docs = [_tokenize(query)] + [_tokenize(text) for _, text in candidates]
    vectors = _vectorize(docs)
    query_vec, cand_vecs = vectors[0], vectors[1:]

    scored = [
        (cid, text, _cosine(query_vec, vec))
        for (cid, text), vec in zip(candidates, cand_vecs)
    ]
    scored = [s for s in scored if s[2] >= min_score]
    scored.sort(key=lambda s: s[2], reverse=True)
    return scored[:k]


# ----------------------------------------------------------------------
# Acceso a datos. Todas las funciones toman una conexion SQLite ya
# abierta (el caller la maneja con el `db()` de server/main.py) en vez de
# abrir la suya propia, para poder participar de la misma transaccion
# que el resto del endpoint que las llama.
# ----------------------------------------------------------------------

def create_session(conn, tenant_id: str | None = None) -> str:
    """
    Crea una sesion nueva y devuelve su session_id (UUID4). `tenant_id`
    queda reservado para el futuro (ver docs/ARQUITECTURA.md): hoy
    siempre es NULL y no se usa para filtrar nada, pero la columna ya
    existe para no tener que migrar el esquema el dia que haga falta
    aislar sesiones por cliente/organizacion.
    """
    session_id = str(uuid.uuid4())
    now = time.time()
    conn.execute(
        "INSERT INTO sessions(session_id,tenant_id,created_at,last_used_at,summary) VALUES(?,?,?,?,?)",
        (session_id, tenant_id, now, now, None),
    )
    return session_id


def get_session(conn, session_id: str):
    return conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()


def list_messages(conn, session_id: str):
    return conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC", (session_id,)
    ).fetchall()


def add_message(conn, session_id: str, role: str, content: str) -> str:
    """Guarda un mensaje (rol 'user' o 'assistant') y actualiza
    `last_used_at` de la sesion. Devuelve el message_id generado."""
    message_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO messages(message_id,session_id,role,content,created_at,trivial,token_estimate,summarized)
           VALUES(?,?,?,?,?,?,?,0)""",
        (message_id, session_id, role, content, time.time(), int(is_trivial(content)), estimate_tokens(content)),
    )
    conn.execute("UPDATE sessions SET last_used_at=? WHERE session_id=?", (time.time(), session_id))
    return message_id


def delete_session(conn, session_id: str) -> None:
    """Borra la sesion Y todos sus mensajes. No toca la tabla `jobs`
    (los jobs historicos de esa sesion quedan, con session_id apuntando
    a una sesion que ya no existe -- igual criterio que cuando se borra
    un nodo desde /admin y sus jobs viejos quedan con assigned_node
    huerfano)."""
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


def build_context(conn, session_id: str, new_message: str) -> tuple[str, dict]:
    """
    Arma el prompt final que va a recibir el modelo para un mensaje nuevo
    dentro de una sesion, y un dict de metricas sobre que se uso.

    IMPORTANTE (orden de llamada): se espera que `new_message` TODAVIA NO
    este guardado en `messages` cuando se llama a esta funcion -- el
    caller (POST /infer) debe llamar a build_context() primero y recien
    despues, con el prompt ya armado, hacer `add_message(..., "user",
    new_message)`. Si se invirtiera el orden, el mensaje nuevo aparecería
    duplicado (una vez dentro de la ventana reciente leida de la base, y
    otra vez al final del prompt armado aca).

    Estructura del prompt final:
        Resumen de la conversacion hasta ahora: ...      (si existe)

        Contexto relevante de mensajes anteriores:
        - ...                                             (si hubo hits)

        Conversacion reciente:
        Usuario: ...
        Asistente: ...
        Usuario: {new_message}
        Asistente:
    """
    session = get_session(conn, session_id)
    all_messages = list_messages(conn, session_id)

    recent = all_messages[-SHORT_TERM_WINDOW:]
    older = all_messages[:-SHORT_TERM_WINDOW] if len(all_messages) > SHORT_TERM_WINDOW else []
    # Solo los mensajes no triviales entran como candidatos a memoria semantica.
    candidates = [(m["message_id"], m["content"]) for m in older if not m["trivial"]]
    hits = top_similar(new_message, candidates)

    parts = []
    if session and session["summary"]:
        parts.append(f"Resumen de la conversacion hasta ahora: {session['summary']}")
    if hits:
        bullets = "\n".join(f"- {text}" for _, text, _score in hits)
        parts.append(f"Contexto relevante de mensajes anteriores:\n{bullets}")

    transcript = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
        for m in recent
    )
    tail = f"Usuario: {new_message}\nAsistente:"
    parts.append(f"Conversacion reciente:\n{transcript}\n{tail}" if transcript else tail)

    prompt = "\n\n".join(parts)
    metrics = {
        "context_tokens_estimate": estimate_tokens(prompt),
        "context_messages_used": len(recent),
        "context_semantic_hits": len(hits),
        "context_summary_used": bool(session and session["summary"]),
    }
    return prompt, metrics


def messages_needing_summary(conn, session_id: str):
    """
    Mensajes "viejos" (fuera de la ventana de corto plazo) que todavia no
    fueron incorporados a ningun resumen. Si hay SUMMARIZE_TRIGGER o mas,
    el caller (server/main.py) debe encolar un job de resumen con
    `build_summary_prompt()`.
    """
    all_messages = list_messages(conn, session_id)
    older = all_messages[:-SHORT_TERM_WINDOW] if len(all_messages) > SHORT_TERM_WINDOW else []
    return [m for m in older if not m["summarized"]]


def build_summary_prompt(session, pending_messages) -> str:
    """
    Arma el prompt que se le manda a un nodo para comprimir los mensajes
    viejos de una sesion en un resumen nuevo, incorporando el resumen
    previo si ya existia (para que el resumen se vaya "acumulando" en vez
    de perder lo que ya se habia comprimido antes).
    """
    prior = session["summary"] if session and session["summary"] else ""
    transcript = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
        for m in pending_messages
    )
    prefix = f"Resumen previo: {prior}\n\n" if prior else ""
    return (
        f"{prefix}Resumi de forma breve y concisa la siguiente parte de una "
        "conversacion, preservando los datos, preferencias y decisiones "
        "importantes que se mencionaron (nombres, gustos, numeros, decisiones "
        f"tomadas). No agregues informacion que no este en el texto.\n\n{transcript}\n\nResumen:"
    )
