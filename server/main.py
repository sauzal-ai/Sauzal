from __future__ import annotations
import sqlite3, time, uuid
from contextlib import contextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).with_name("sauzal.db")
app = FastAPI(title="Sauzal Control Plane", version="0.2.0")

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

@app.on_event("startup")
def startup():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
          node_id TEXT PRIMARY KEY, name TEXT, token TEXT, status TEXT,
          last_seen REAL, capabilities TEXT);
        CREATE TABLE IF NOT EXISTS jobs(
          job_id TEXT PRIMARY KEY, model TEXT, prompt TEXT, status TEXT,
          assigned_node TEXT, result TEXT, error TEXT,
          created_at REAL, updated_at REAL);
        """)

class Register(BaseModel):
    name: str
    capabilities: str

class Auth(BaseModel):
    node_id: str
    token: str

class Heartbeat(Auth):
    status: str = "available"
    capabilities: str | None = None

class Infer(BaseModel):
    model: str = "gemma3:4b"
    prompt: str = Field(min_length=1)
    node: str | None = None

class Result(Auth):
    success: bool
    result: str | None = None
    error: str | None = None

def authenticate(conn, node_id, token):
    node = conn.execute(
        "SELECT * FROM nodes WHERE node_id=? AND token=?",
        (node_id, token)
    ).fetchone()
    if not node:
        raise HTTPException(401, "Invalid node credentials")
    return node

@app.get("/health")
def health():
    return {"status":"ok","service":"sauzal-control-plane","version":"0.2.0"}

@app.post("/nodes/register")
def register(req: Register):
    node_id, token = str(uuid.uuid4()), str(uuid.uuid4())
    with db() as conn:
        conn.execute(
            "INSERT INTO nodes VALUES(?,?,?,?,?,?)",
            (node_id, req.name, token, "available", time.time(), req.capabilities)
        )
    return {"node_id":node_id,"token":token}

@app.post("/nodes/heartbeat")
def heartbeat(req: Heartbeat):
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        if req.capabilities is None:
            conn.execute(
                "UPDATE nodes SET status=?,last_seen=? WHERE node_id=?",
                (req.status,time.time(),req.node_id)
            )
        else:
            conn.execute(
                "UPDATE nodes SET status=?,last_seen=?,capabilities=? WHERE node_id=?",
                (req.status,time.time(),req.capabilities,req.node_id)
            )
    return {"ok":True}

@app.get("/nodes")
def list_nodes():
    now=time.time()
    with db() as conn:
        rows=conn.execute(
            "SELECT node_id,name,status,last_seen,capabilities FROM nodes ORDER BY last_seen DESC"
        ).fetchall()
    return [{**dict(r),"online":now-r["last_seen"]<30} for r in rows]

@app.post("/infer")
def infer(req: Infer):
    now=time.time()
    with db() as conn:
        if req.node:
            node=conn.execute("""
              SELECT node_id FROM nodes
              WHERE status='available' AND last_seen>? AND (node_id=? OR name=?)
              ORDER BY last_seen DESC LIMIT 1
            """,(now-30,req.node,req.node)).fetchone()
        else:
            node=conn.execute("""
              SELECT node_id FROM nodes
              WHERE status='available' AND last_seen>?
              ORDER BY last_seen DESC LIMIT 1
            """,(now-30,)).fetchone()
        if not node:
            raise HTTPException(503,"No available Sauzal nodes")
        job_id=str(uuid.uuid4())
        conn.execute("""
          INSERT INTO jobs(job_id,model,prompt,status,assigned_node,created_at,updated_at)
          VALUES(?,?,?,'queued',?,?,?)
        """,(job_id,req.model,req.prompt,node["node_id"],now,now))
        conn.execute("UPDATE nodes SET status='busy' WHERE node_id=?",(node["node_id"],))
    return {"job_id":job_id,"assigned_node":node["node_id"],"status":"queued"}

@app.post("/agent/pull")
def pull(req: Auth):
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        job=conn.execute("""
          SELECT job_id,model,prompt FROM jobs
          WHERE assigned_node=? AND status='queued'
          ORDER BY created_at LIMIT 1
        """,(req.node_id,)).fetchone()
        if not job:
            return {"job":None}
        conn.execute(
            "UPDATE jobs SET status='running',updated_at=? WHERE job_id=?",
            (time.time(),job["job_id"])
        )
    return {"job":dict(job)}

@app.post("/jobs/{job_id}/result")
def result(job_id: str, req: Result):
    with db() as conn:
        authenticate(conn, req.node_id, req.token)
        status="completed" if req.success else "failed"
        conn.execute("""
          UPDATE jobs SET status=?,result=?,error=?,updated_at=? WHERE job_id=?
        """,(status,req.result,req.error,time.time(),job_id))
        conn.execute(
            "UPDATE nodes SET status='available',last_seen=? WHERE node_id=?",
            (time.time(),req.node_id)
        )
    return {"ok":True}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with db() as conn:
        job=conn.execute("""
          SELECT jobs.*, nodes.name AS node_name FROM jobs
          LEFT JOIN nodes ON nodes.node_id = jobs.assigned_node
          WHERE jobs.job_id=?
        """,(job_id,)).fetchone()
    if not job:
        raise HTTPException(404,"Job not found")
    return dict(job)
