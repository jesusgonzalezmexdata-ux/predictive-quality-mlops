"""Carga de SECOM (UCI): 1,567 lotes, 590 sensores anónimos, etiqueta -1 pasa / 1 falla, con fecha."""
import numpy as np
import pandas as pd

from pqm.config import RAW_SECOM


def load_secom(path=RAW_SECOM) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve X (590 columnas f0..f589), y (1 = falla) y t (marca de tiempo), ordenados por tiempo."""
    X = pd.read_csv(path / "secom.data", sep=r"\s+", header=None).astype(float)
    X.columns = [f"f{i}" for i in X.columns]
    lab = pd.read_csv(path / "secom_labels.data", sep=r"\s+", header=None, names=["y", "t"], quotechar='"')
    t = pd.to_datetime(lab["t"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["y"] == 1).astype(int)
    orden = np.argsort(t.values, kind="stable")
    X, y, t = X.iloc[orden].reset_index(drop=True), y.iloc[orden].reset_index(drop=True), t.iloc[orden].reset_index(drop=True)
    return X, y, t


def split_temporal(n: int, frac_holdout: float = 0.2) -> tuple[slice, slice]:
    """Último bloque cronológico como holdout. Nunca se mezcla con el entrenamiento."""
    corte = int(round(n * (1 - frac_holdout)))
    return slice(0, corte), slice(corte, n)
