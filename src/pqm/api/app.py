"""API de calidad predictiva. Ejecutar: uvicorn pqm.api.app:app --port 8000"""
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scipy.stats import ks_2samp

from pqm import drift, explain
from pqm.config import MODELS

DB_PATH = Path(os.environ.get("PQM_DB", "predicciones.db"))
MODEL_DIR = Path(os.environ.get("PQM_MODEL_DIR", MODELS))

app = FastAPI(title="Calidad predictiva SECOM", version="1.0.0",
              description="Riesgo de falla por lote, decisión por costo y explicación por sensor.")
_state: dict = {}


class Lote(BaseModel):
    lote_id: str = Field(..., description="Identificador del lote")
    sensores: dict[str, float | None] = Field(..., description="Lecturas por sensor (f0..f589). Los faltantes se omiten o van en null")


class Prediccion(BaseModel):
    lote_id: str
    probabilidad_falla: float
    umbral: float
    accion: str
    sensores_recibidos: int
    sensores_desconocidos: list[str]
    factores: list[dict]


def _cargar() -> None:
    if _state:
        return
    _state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    _state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    _state["alias"] = explain.cargar_alias()
    _state["cols"] = _state["meta"]["todas_las_features"]
    _state["ref"] = pd.read_csv(MODEL_DIR / "referencia.csv")


@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, lote_id TEXT, probabilidad REAL, accion TEXT, version TEXT, sensores_recibidos INTEGER, lecturas TEXT)""")
    try:
        yield con
        con.commit()
    finally:
        con.close()


@app.get("/health")
def health() -> dict:
    _cargar()
    return {"estado": "ok", "version": _state["meta"]["version"]}


@app.get("/model-info")
def model_info() -> dict:
    _cargar()
    m = _state["meta"]
    return {"version": m["version"], "umbral": m["umbral"], "sensores_en_modelo": len(m["features"]),
            "entrenado_con_lotes": m["entrenado_con"], "costos_supuestos": m["costos"],
            "aviso": "Alias de sensores ilustrativos; ver config/sensor_aliases.yaml"}


@app.post("/predict", response_model=Prediccion)
def predict(lote: Lote) -> Prediccion:
    _cargar()
    cols = _state["cols"]
    desconocidos = [s for s in lote.sensores if s not in cols]
    conocidos = {k: v for k, v in lote.sensores.items() if k in cols}
    if not conocidos:
        raise HTTPException(422, "Ningún sensor reconocido (se esperan nombres f0..f589)")
    fila = pd.DataFrame([{c: conocidos.get(c, np.nan) if conocidos.get(c) is not None else np.nan for c in cols}])
    pipe = _state["model"]
    p = float(pipe.predict_proba(fila)[0, 1])
    umbral = _state["meta"]["umbral"]
    accion = "INSPECCIONAR el lote antes de liberarlo" if p >= umbral else "LIBERAR: riesgo bajo el umbral de costo"
    sv, Z, _ = explain.shap_values(pipe, fila)
    factores = explain.explicar_lote(sv.iloc[0], Z.iloc[0], _state["alias"], 5)
    with _db() as con:
        con.execute("INSERT INTO predicciones (ts, lote_id, probabilidad, accion, version, sensores_recibidos, lecturas) VALUES (?,?,?,?,?,?,?)",
                    (time.time(), lote.lote_id, p, accion, _state["meta"]["version"], len(conocidos), json.dumps(conocidos)))
    return Prediccion(lote_id=lote.lote_id, probabilidad_falla=p, umbral=umbral, accion=accion,
                      sensores_recibidos=len(conocidos), sensores_desconocidos=desconocidos, factores=factores)


@app.get("/predicciones")
def historial(limite: int = 50) -> list[dict]:
    with _db() as con:
        filas = con.execute("SELECT ts, lote_id, probabilidad, accion, version, sensores_recibidos FROM predicciones ORDER BY id DESC LIMIT ?",
                            (min(limite, 500),)).fetchall()
    return [dict(zip(["ts", "lote_id", "probabilidad", "accion", "version", "sensores_recibidos"], f)) for f in filas]


@app.get("/drift")
def drift_reciente(ultimas: int = 200, minimo: int = 100) -> dict:
    """PSI por sensor entre la muestra de entrenamiento y las últimas lecturas recibidas.
    Con menos de `minimo` predicciones no se calcula (el PSI no es confiable con muestras chicas)."""
    _cargar()
    with _db() as con:
        filas = con.execute("SELECT lecturas, probabilidad FROM predicciones ORDER BY id DESC LIMIT ?", (ultimas,)).fetchall()
    if len(filas) < minimo:
        return {"estado": "datos_insuficientes", "predicciones": len(filas), "minimo": minimo}
    nuevo = pd.DataFrame([json.loads(f[0]) for f in filas])
    ref = _state["ref"]
    cols = [c for c in ref.columns if c in nuevo.columns]
    t = drift.psi_table(ref, nuevo, cols)
    # Sin etiquetas: (a) PSI de sensores críticos y (b) KS entre la distribución de puntajes de referencia y la reciente
    criticos = [c for c in _state["meta"].get("sensores_criticos", []) if c in nuevo.columns]
    n_crit = int(t[t.sensor.isin(criticos)].psi.gt(0.2).sum())
    if "ref_scores" not in _state:
        _state["ref_scores"] = _state["model"][-1].predict_proba(_state["ref"])[:, 1]
    ks = ks_2samp(_state["ref_scores"], [f[1] for f in filas])
    motivos = (["PSI>0.2 en más de 3 sensores críticos"] if n_crit > 3 else []) + (["cambió la distribución de puntajes (KS p<0.05)"] if ks.pvalue < 0.05 else [])
    return {"estado": "ok", "predicciones": len(filas), "sensores_comparados": len(cols),
            "con_deriva": int((t.psi > 0.25).sum()), "vigilar": int(((t.psi > 0.1) & (t.psi <= 0.25)).sum()),
            "criticos_con_psi_mayor_0.2": n_crit, "ks_puntajes": {"estadistico": round(float(ks.statistic), 3), "p": float(ks.pvalue)},
            "alerta_reentrenar": bool(motivos), "motivos": motivos,
            "top": t.head(10).assign(psi=lambda d: d.psi.round(3)).to_dict("records")}
