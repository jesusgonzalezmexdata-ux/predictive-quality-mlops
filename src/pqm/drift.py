"""Deriva de datos: índice de estabilidad poblacional (PSI) entre un periodo de referencia y uno nuevo."""
import numpy as np
import pandas as pd


def psi(ref: pd.Series, new: pd.Series, bins: int = 10) -> float:
    ref, new = ref.dropna(), new.dropna()
    if len(ref) < 20 or len(new) < 20 or ref.nunique() < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(new, edges)[0] / len(new)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def psi_table(ref: pd.DataFrame, new: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = list(cols) if cols is not None else list(ref.columns)
    t = pd.DataFrame({"sensor": cols, "psi": [psi(ref[c], new[c]) for c in cols]})
    t["estado"] = pd.cut(t["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["estable", "vigilar", "deriva"])
    return t.sort_values("psi", ascending=False, ignore_index=True)
