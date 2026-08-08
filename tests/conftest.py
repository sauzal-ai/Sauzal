"""
Configuracion compartida de los tests de Sauzal.

Lo mas importante de este archivo: setea SAUZAL_DB_PATH ANTES de que se
importe server.main en cualquier lado. server/main.py lee esa variable
de entorno una sola vez, al importarse, para decidir donde vive la base
SQLite -- si esto no se hiciera aca (antes de cualquier `import server.main`),
los tests terminarian leyendo y escribiendo la base real
(server/sauzal.db), lo cual seria un desastre.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_TEST_DB = Path(tempfile.mkdtemp(prefix="sauzal-tests-")) / "sauzal_test.db"
os.environ["SAUZAL_DB_PATH"] = str(_TEST_DB)


@pytest.fixture()
def client():
    """
    TestClient de FastAPI contra una base SQLite temporal, recreada
    (vacia) en cada test para que no quede estado compartido entre tests
    (nodos/sesiones/jobs de un test no deben poder afectar a otro).
    """
    if _TEST_DB.exists():
        _TEST_DB.unlink()

    from fastapi.testclient import TestClient
    from server import main

    with TestClient(main.app) as test_client:
        yield test_client
