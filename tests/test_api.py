import json
import os

import pytest
from fastapi.testclient import TestClient

from pqm.config import CostosCalidad
from pqm.data import load_secom


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PQM_DB", str(tmp_path / "t.db"))
    import importlib
    from pqm.api import app as mod
    importlib.reload(mod)
    return TestClient(mod.app), mod


def test_health_y_modelo(client):
    c, _ = client
    assert c.get("/health").json()["estado"] == "ok"
    info = c.get("/model-info").json()
    assert 0 < info["umbral"] < 1 and info["sensores_en_modelo"] > 100


def test_predict_devuelve_decision_y_factores_y_registra(client):
    c, _ = client
    X, y, _t = load_secom()
    fila = {k: (None if v != v else float(v)) for k, v in X.iloc[1500].items()}
    r = c.post("/predict", json={"lote_id": "L-1500", "sensores": fila})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["probabilidad_falla"] <= 1 and d["accion"].split()[0] in {"INSPECCIONAR", "LIBERAR:"} or d["accion"].startswith(("INSPECCIONAR", "LIBERAR"))
    assert len(d["factores"]) == 5 and d["sensores_desconocidos"] == []
    assert c.get("/predicciones").json()[0]["lote_id"] == "L-1500"


def test_predict_rechaza_entrada_sin_sensores_validos(client):
    c, _ = client
    assert c.post("/predict", json={"lote_id": "x", "sensores": {"sensor_raro": 1.0}}).status_code == 422
    assert c.post("/predict", json={"sensores": {}}).status_code == 422


def test_predict_tolera_sensores_faltantes_y_extra(client):
    c, _ = client
    r = c.post("/predict", json={"lote_id": "parcial", "sensores": {"f59": 1.0, "f0": 3000.0, "no_existe": 5.0}})
    assert r.status_code == 200 and r.json()["sensores_desconocidos"] == ["no_existe"]


def test_drift_requiere_muestra_suficiente(client):
    c, _ = client
    assert c.get("/drift").json()["estado"] == "datos_insuficientes"


def test_drift_detecta_corrimiento_fuerte(client):
    c, _ = client
    X, _, _t = load_secom()
    meta = json.loads((__import__("pathlib").Path(os.environ.get("PQM_MODEL_DIR", "models")) / "metadata.json").read_text()) if False else None
    for i in range(40):
        fila = {k: (None if v != v else float(v) * 5 + 100) for k, v in X.iloc[i].items()}
        assert c.post("/predict", json={"lote_id": f"d{i}", "sensores": fila}).status_code == 200
    d = c.get("/drift").json()
    assert d["estado"] == "ok" and d["con_deriva"] > 50
