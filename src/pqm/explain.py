"""Explicabilidad con SHAP y traducción de sensores anónimos a alias legibles (ilustrativos)."""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml

from pqm.config import ROOT

ALIAS_PATH = ROOT / "config/sensor_aliases.yaml"


def cargar_alias(path: Path = ALIAS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("alias", {})


def nombre(sensor: str, alias: dict[str, str]) -> str:
    return f"{alias[sensor]} ({sensor})" if sensor in alias else sensor


def shap_values(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Devuelve (valores SHAP, datos transformados, valor base) en escala logit para la clase 'falla'."""
    Z = pipe[0].transform(X)
    ex = shap.TreeExplainer(pipe[-1])
    sv = ex.shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    base = float(np.ravel(ex.expected_value)[-1])
    return pd.DataFrame(sv, columns=Z.columns, index=Z.index), Z, base


def resumen_global(sv: pd.DataFrame, Z: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """Importancia media |SHAP| y dirección del efecto (correlación valor-SHAP): + sube el riesgo al subir el sensor."""
    imp = sv.abs().mean().sort_values(ascending=False).head(k)
    filas = []
    for s, v in imp.items():
        ok = Z[s].notna()
        r = np.corrcoef(Z.loc[ok, s], sv.loc[ok, s])[0, 1] if ok.sum() > 5 and Z.loc[ok, s].nunique() > 1 else np.nan
        filas.append({"sensor": s, "shap_medio_abs": float(v), "direccion": "↑ valor alto sube el riesgo" if r > 0.1 else "↓ valor alto baja el riesgo" if r < -0.1 else "no monótono"})
    return pd.DataFrame(filas)


def explicar_lote(sv_fila: pd.Series, valores: pd.Series, alias: dict[str, str], k: int = 5) -> list[dict]:
    top = sv_fila.abs().sort_values(ascending=False).head(k).index
    return [{"sensor": s, "nombre": nombre(s, alias), "valor": None if pd.isna(valores[s]) else float(valores[s]),
             "aporte": float(sv_fila[s])} for s in top]


_ACCIONES = [("Presion", "verificar sellos/fugas y calibración del transmisor de presión"),
             ("Temperatura", "verificar la calibración del termopar y el control de la zona"),
             ("Potencia", "revisar suministro RF y conexiones del electrodo"), ("Voltaje", "revisar suministro y conexiones del electrodo"),
             ("Corriente", "revisar el accionamiento del motor"), ("Flujo", "verificar el controlador de flujo másico"),
             ("Humedad", "revisar el acondicionamiento de la sala limpia"), ("Vibracion", "inspeccionar rodamientos de la bomba")]


def accion_sugerida(nombre_sensor: str) -> str | None:
    """Sugerencia de verificación según la familia del alias. Es HIPOTÉTICA: depende del diccionario real de tags."""
    return next((a for k, a in _ACCIONES if nombre_sensor.startswith(k)), None)
