"""Validación temporal, métricas para clase rara y decisión por costo."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pqm.config import CostosCalidad


def expanding_window_folds(n: int, n_folds: int = 4, min_train: float = 0.4):
    """Entrena con todo lo anterior y prueba en el siguiente bloque cronológico."""
    inicio = int(n * min_train)
    cortes = np.linspace(inicio, n, n_folds + 1).astype(int)
    for a, b in zip(cortes[:-1], cortes[1:]):
        yield np.arange(0, a), np.arange(a, b)


def pr_auc(y, p) -> float:
    return float(average_precision_score(y, p)) if np.sum(y) > 0 else float("nan")


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if 0 < np.sum(y) < len(y) else float("nan")


def lift_at_k(y, p, frac: float = 0.1) -> float:
    """Cuántas veces más fallas hay en el 10 % de lotes con mayor riesgo que en el promedio."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-p)[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def bootstrap_ci(y, p, fn=pr_auc, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def best_threshold(y, p, costos: CostosCalidad | None = None) -> tuple[float, float]:
    """Umbral que minimiza el costo esperado en los datos dados. Devuelve (umbral, costo)."""
    costos = costos or CostosCalidad()
    candidatos = np.unique(np.quantile(p, np.linspace(0, 1, 201)))
    mejor = min(((costos.costo(y, p >= t), t) for t in candidatos), key=lambda x: x[0])
    return float(mejor[1]), float(mejor[0])


def monetizar(y, p, umbral: float, costos: CostosCalidad | None = None) -> dict:
    costos = costos or CostosCalidad()
    y = np.asarray(y).astype(bool)
    f = np.asarray(p) >= umbral
    return {
        "lotes": int(len(y)), "fallas": int(y.sum()), "marcados": int(f.sum()),
        "detectadas": int((y & f).sum()), "no_detectadas": int((y & ~f).sum()), "falsas_alarmas": int((~y & f).sum()),
        "costo_con_modelo": costos.costo(y, f), "costo_sin_modelo": costos.sin_modelo(y),
        "costo_inspeccionar_todo": costos.inspeccionar_todo(y),
        "costo_mejor_baseline": min(costos.sin_modelo(y), costos.inspeccionar_todo(y)),
        "ahorro_vs_sin_modelo": costos.sin_modelo(y) - costos.costo(y, f),
        "ahorro_vs_mejor_baseline": min(costos.sin_modelo(y), costos.inspeccionar_todo(y)) - costos.costo(y, f),
        "ahorro_vs_inspeccionar_todo": costos.inspeccionar_todo(y) - costos.costo(y, f),
    }
