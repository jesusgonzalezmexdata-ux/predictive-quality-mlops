"""NASA C-MAPSS (turbofan): vida útil restante (RUL), probabilidad de falla en una ventana y política de mantenimiento.

Ejecutar desde src/:  python -m pqm.rul.cmapss
NO implementa modelos de supervivencia (Weibull / RSF): la probabilidad de falla viene de un clasificador sobre RUL<=H.
"""
import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from pqm.config import RAW_CMAPSS, ROOT, SEED, CostosMantenimiento

CAP = 125
VENTANA = 10
COLS = ["motor", "ciclo", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OPS = ["op1", "op2", "op3"]


def cargar(fd: str, raw=RAW_CMAPSS):
    tr = pd.read_csv(raw / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(raw / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_te = pd.read_csv(raw / f"RUL_{fd}.txt", header=None, names=["rul"])["rul"].to_numpy()
    return tr, te, rul_te


def rul_train(df: pd.DataFrame, cap: int = CAP) -> pd.Series:
    """RUL con tope: la degradación es imperceptible al inicio de la vida del motor."""
    return (df.groupby("motor")["ciclo"].transform("max") - df["ciclo"]).clip(upper=cap)


class Normalizador:
    """Normaliza sensores por régimen operativo (KMeans sobre los 3 ajustes). Con 1 régimen equivale a z-score global."""

    def __init__(self, n_reg: int = 1):
        self.n_reg = n_reg

    def fit(self, df):
        self.sensores_ = [c for c in df.columns if c.startswith("s") and df[c].std() > 1e-6]
        self.km_ = KMeans(self.n_reg, n_init=5, random_state=SEED).fit(df[OPS]) if self.n_reg > 1 else None
        reg = self._reg(df)
        g = df[self.sensores_].groupby(reg)
        self.mu_, self.sd_ = g.mean(), g.std().replace(0, 1)
        return self

    def _reg(self, df):
        return self.km_.predict(df[OPS]) if self.km_ is not None else np.zeros(len(df), dtype=int)

    def transform(self, df):
        reg = self._reg(df)
        Z = (df[self.sensores_].to_numpy() - self.mu_.loc[reg].to_numpy()) / self.sd_.loc[reg].to_numpy()
        return pd.DataFrame(Z, columns=self.sensores_, index=df.index)


def _pendiente(x: np.ndarray, w: int) -> np.ndarray:
    xp = np.concatenate([np.repeat(x[:1], w - 1), x])
    v = sliding_window_view(xp, w)
    t = np.arange(w) - (w - 1) / 2
    return (v * t).sum(1) / (t ** 2).sum()


def ventanas(df: pd.DataFrame, Z: pd.DataFrame, w: int = VENTANA) -> pd.DataFrame:
    """Por motor y ciclo: valor normalizado, media móvil y pendiente de degradación (solo pasado, sin fuga)."""
    out = []
    for _, idx in df.groupby("motor").indices.items():
        z = Z.iloc[idx]
        f = {"ciclo": df["ciclo"].iloc[idx].to_numpy()}
        for c in Z.columns:
            x = z[c].to_numpy()
            xp = np.concatenate([np.repeat(x[:1], w - 1), x])
            f[c] = x
            f[f"{c}_mm"] = sliding_window_view(xp, w).mean(1)
            f[f"{c}_pend"] = _pendiente(x, w)
        out.append(pd.DataFrame(f, index=df.index[idx]))
    return pd.concat(out).loc[df.index]


def score_nasa(real, pred) -> float:
    """Puntuación asimétrica de NASA (penaliza más predecir RUL mayor al real: se falla antes de lo previsto)."""
    d = np.asarray(pred, float) - np.asarray(real, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def costo_politica(rul_real, rul_pred, margen: int, c: CostosMantenimiento) -> np.ndarray:
    """Costo por motor al reemplazar cuando falten `margen` ciclos según la predicción.
    Si el reemplazo planeado cae antes de la falla: preventivo + vida útil desperdiciada; si no: correctivo."""
    plan = np.maximum(np.asarray(rul_pred) - margen, 0)
    real = np.asarray(rul_real)
    return np.where(plan <= real, c.preventivo + c.vida_perdida_por_ciclo * (real - plan), c.correctivo)


def evaluar(fd: str, n_reg: int, c: CostosMantenimiento = CostosMantenimiento(), verbose: bool = True) -> dict:
    tr, te, rul_te = cargar(fd)
    norm = Normalizador(n_reg).fit(tr)
    Ftr = ventanas(tr, norm.transform(tr))
    Fte_all = ventanas(te, norm.transform(te))
    y = rul_train(tr)
    y_real = tr.groupby("motor")["ciclo"].transform("max") - tr["ciclo"]
    g = tr["motor"].to_numpy()
    prm = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.5, reg_lambda=5, n_jobs=2, random_state=SEED, verbose=-1)

    # Validación agrupada por motor (nunca se reparte un motor entre entrenamiento y validación)
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(Ftr, y, g):
        oof[b] = LGBMRegressor(**prm).fit(Ftr.iloc[a], y.iloc[a]).predict(Ftr.iloc[b])
    rmse_cv = float(np.sqrt(np.mean((oof - y) ** 2)))

    # Margen óptimo elegido SOLO con predicciones fuera de muestra del entrenamiento
    ciclo_ok = tr["ciclo"].to_numpy() >= 30
    margenes = list(range(0, 61, 5))
    cv_cost = {m: float(costo_politica(y_real[ciclo_ok], oof[ciclo_ok], m, c).mean()) for m in margenes}
    margen = min(cv_cost, key=cv_cost.get)

    # Evaluación en el conjunto de prueba oficial (último ciclo observado de cada motor vs. RUL oficial)
    reg = LGBMRegressor(**prm).fit(Ftr, y)
    ultimo = te.groupby("motor")["ciclo"].idxmax()
    Fte = Fte_all.loc[ultimo.values]
    pred = np.clip(reg.predict(Fte), 0, CAP)
    real = np.minimum(rul_te, CAP * 10)
    rmse = float(np.sqrt(np.mean((pred - np.minimum(rul_te, CAP)) ** 2)))
    rmse_sin_tope = float(np.sqrt(np.mean((pred - rul_te) ** 2)))

    # Probabilidad de falla dentro del horizonte
    H = c.horizonte
    yb = (y_real <= H).astype(int)
    clf_prm = {**prm, "scale_pos_weight": float((yb == 0).sum() / max(yb.sum(), 1))}
    oofp = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(Ftr, yb, g):
        oofp[b] = LGBMClassifier(**clf_prm).fit(Ftr.iloc[a], yb.iloc[a]).predict_proba(Ftr.iloc[b])[:, 1]
    clf = LGBMClassifier(**clf_prm).fit(Ftr, yb)
    p_te = clf.predict_proba(Fte)[:, 1]
    yb_te = (rul_te <= H).astype(int)

    # Negocio
    n = len(rul_te)
    corr = n * c.correctivo
    res_pol = {}
    for m in [0, margen, 30]:
        res_pol[str(m)] = float(costo_politica(rul_te, pred, m, c).sum())
    # Alternativa sin predicción: mismo plazo de reemplazo para todos los motores, elegido mirando el test (ventaja para el baseline)
    fija = {}
    for k in range(0, 126, 5):
        fija[k] = float(costo_politica(rul_te, np.full(n, CAP), CAP - k, c).sum())
    mejor_fija = min(fija.values())
    out = {
        "fd": fd, "regimenes": n_reg, "motores_train": int(tr.motor.nunique()), "motores_test": int(n),
        "rmse_cv_agrupado": rmse_cv, "rmse_test": rmse, "rmse_test_sin_tope": rmse_sin_tope, "score_nasa": score_nasa(np.minimum(rul_te, CAP), pred),
        "clasificador": {"horizonte": H, "pr_auc_cv": float(average_precision_score(yb, oofp)), "prevalencia_cv": float(yb.mean()),
                         "pr_auc_test": float(average_precision_score(yb_te, p_te)) if yb_te.sum() else None,
                         "positivos_test": int(yb_te.sum())},
        "mantenimiento": {"margen_elegido_cv": margen, "costo_cv_por_margen": cv_cost,
                          "costo_correr_hasta_fallar": float(corr),
                          "costo_con_modelo": res_pol[str(margen)],
                          "costo_mejor_plazo_fijo_oraculo": mejor_fija,
                          "ahorro_vs_correr_hasta_fallar": float(corr - res_pol[str(margen)]),
                          "fallas_no_evitadas": int((np.maximum(pred - margen, 0) > rul_te).sum())},
    }
    if verbose:
        print(json.dumps({k: v for k, v in out.items() if k != "mantenimiento"}, indent=1))
        print({k: v for k, v in out["mantenimiento"].items() if k != "costo_cv_por_margen"})
    pd.DataFrame({"real": rul_te, "pred": pred, "p_falla_H": p_te}).to_csv(ROOT / f"reports/rul_test_{fd}.csv", index=False)
    return out


def main():
    warnings.filterwarnings("ignore")
    res = {"FD001": evaluar("FD001", 1), "FD002": evaluar("FD002", 6)}
    (ROOT / "reports/rul_metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
