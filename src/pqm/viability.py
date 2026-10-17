"""Dictamen de viabilidad de un modelo de calidad: ¿conviene desplegarlo, descartarlo o juntar más datos?

Aplica a la tabla de lotes de cualquier proceso el mismo protocolo que el caso SECOM:

1. Partición cronológica: el último 20 % de los lotes se evalúa una sola vez.
2. Dentro del pasado, ventanas crecientes: los puntajes fuera de muestra eligen el modelo (gradient boosting
   o regresión logística) y el umbral de menor costo, sin ver el periodo evaluado.
3. Señal: PR-AUC en lotes futuros con intervalo bootstrap contra la tasa de falla (lo que daría el azar).
4. Dinero: ahorro contra la mejor alternativa simple (no inspeccionar o inspeccionar todo) con intervalo
   bootstrap, y cuántos lotes y meses más harían falta para que el intervalo excluya cero.
5. Riesgos: brecha entre validación al azar y temporal (el proceso cambia), deriva de variables y cambio
   en la tasa de falla, sensibilidad a la razón de costos.

Todo corre con scikit-learn para que funcione en el navegador (Pyodide).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pqm.config import CostosCalidad
from pqm.data import split_temporal
from pqm.drift import psi
from pqm.evaluation import (
    best_threshold,
    bootstrap_ci,
    expanding_window_folds,
    lift_at_k,
    monetizar,
    pr_auc,
)

MIN_FAILURES, MIN_HOLDOUT_FAILURES = 20, 5
DESPLEGAR, NO_DESPLEGAR, RECOLECTAR = "DESPLEGAR", "NO DESPLEGAR", "RECOLECTAR MÁS DATOS"


@dataclass(frozen=True)
class Ajustes:
    scrap_no_detectado: float = 5000.0
    inspeccion: float = 250.0
    retrabajo_detectado: float = 500.0
    frac_holdout: float = 0.2
    n_folds: int = 4
    nivel: float = 0.95
    remuestreos: int = 1000
    semilla: int = 0
    meses_max: float = 24.0   # plazo razonable para juntar evidencia

    @property
    def costos(self) -> CostosCalidad:
        return CostosCalidad(self.scrap_no_detectado, self.inspeccion, self.retrabajo_detectado)


def _candidatos(seed: int) -> dict:
    return {
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                                            l2_regularization=1.0, class_weight="balanced", early_stopping=False,
                                                            random_state=seed),
        "Regresión logística": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
    }


def _fresh(name: str, seed: int):
    return _candidatos(seed)[name]


def _oof(X: pd.DataFrame, y: np.ndarray, name: str, aj: Ajustes):
    idx, scores = [], []
    for tr, te in expanding_window_folds(len(X), aj.n_folds):
        if y[tr].sum() < 3:
            continue
        m = _fresh(name, aj.semilla).fit(X.iloc[tr], y[tr])
        idx.append(te)
        scores.append(m.predict_proba(X.iloc[te])[:, 1])
    if not idx:
        return np.array([], int), np.array([])
    return np.concatenate(idx), np.concatenate(scores)


# Sin train_test_split ni permutation_importance de scikit-learn: en Pyodide su chequeo de pyarrow falla.
def _stratified_split(X: pd.DataFrame, y: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        test[rng.choice(idx, size=max(1, int(round(len(idx) * frac))), replace=False)] = True
    return X.iloc[~test], X.iloc[test], y[~test], y[test]


def _permutation_importance(model, X: pd.DataFrame, y: np.ndarray, base: float, seed: int, repeats: int = 3) -> np.ndarray:
    """Caída media de PR-AUC al revolver cada variable en el periodo evaluado."""
    rng = np.random.default_rng(seed)
    out = np.zeros(X.shape[1])
    for j, col in enumerate(X.columns):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - pr_auc(y, model.predict_proba(Xp)[:, 1]))
        out[j] = float(np.mean(drops))
    return out


def _per_lot(y, flag, c: CostosCalidad):
    y, f = np.asarray(y, bool), np.asarray(flag, bool)
    model = np.where(y & f, c.inspeccion + c.retrabajo_detectado, 0) + np.where(~y & f, c.inspeccion, 0) + np.where(y & ~f, c.scrap_no_detectado, 0)
    return model, y * c.scrap_no_detectado, c.inspeccion + y * c.retrabajo_detectado


def _saving_ci(y, flag, aj: Ajustes):
    model, none, all_ = _per_lot(y, flag, aj.costos)
    rng = np.random.default_rng(aj.semilla)
    b = rng.integers(0, len(y), size=(aj.remuestreos, len(y)))
    saving = np.minimum(none[b].sum(1), all_[b].sum(1)) - model[b].sum(1)
    a = (1 - aj.nivel) / 2
    lo, hi = np.quantile(saving, [a, 1 - a])
    return float(lo), float(hi), float(saving.std(ddof=1)), float((saving > 0).mean())


def evaluate(df: pd.DataFrame, variables: list[str], aj: Ajustes | None = None) -> dict:
    aj = aj or Ajustes()
    X, y = df[variables], df["y"].to_numpy(int)
    tr, ho = split_temporal(len(X), aj.frac_holdout)
    Xtr, ytr, Xho, yho = X.iloc[tr], y[tr], X.iloc[ho], y[ho]
    meses = max((df["fecha"].iloc[-1] - df["fecha"].iloc[0]).days / 30.44, 1 / 30)
    base = dict(lotes=len(df), fallas=int(y.sum()), variables=len(variables), lotes_por_mes=len(df) / meses,
                desde=df["fecha"].iloc[0], hasta=df["fecha"].iloc[-1], corte=df["fecha"].iloc[tr.stop],
                lotes_eval=int(len(yho)), fallas_eval=int(yho.sum()), prevalencia_eval=float(yho.mean()), ajustes=aj)
    if y.sum() < MIN_FAILURES or yho.sum() < MIN_HOLDOUT_FAILURES or ytr.sum() < 10:
        motivo = (f"El histórico tiene {int(y.sum())} fallas ({int(yho.sum())} en el periodo de evaluación). Se necesitan al menos "
                  f"{MIN_FAILURES} en total y {MIN_HOLDOUT_FAILURES} en el último 20 % para evaluar algo.")
        return {**base, "evaluable": False, "dictamen": RECOLECTAR, "motivo": motivo, "hallazgos": [], "riesgos": []}

    # 1) Selección del modelo y del umbral con puntajes fuera de muestra del pasado.
    oof = {name: _oof(Xtr, ytr, name, aj) for name in _candidatos(aj.semilla)}
    cv = {name: pr_auc(ytr[i], s) if len(i) and ytr[i].sum() else float("nan") for name, (i, s) in oof.items()}
    elegido = max(cv, key=lambda k: -1 if np.isnan(cv[k]) else cv[k])
    i_oof, s_oof = oof[elegido]
    umbral, _ = best_threshold(ytr[i_oof], s_oof, aj.costos)
    cv_ic = bootstrap_ci(ytr[i_oof], s_oof, n=aj.remuestreos, seed=aj.semilla, alpha=1 - aj.nivel)

    # 2) Una sola evaluación en el periodo reservado.
    model = _fresh(elegido, aj.semilla).fit(Xtr, ytr)
    p = model.predict_proba(Xho)[:, 1]
    pr = pr_auc(yho, p)
    pr_lo, pr_hi = bootstrap_ci(yho, p, n=aj.remuestreos, seed=aj.semilla, alpha=1 - aj.nivel)
    mon = monetizar(yho, p, umbral, aj.costos)
    lo, hi, _, prob = _saving_ci(yho, p >= umbral, aj)
    s = mon["ahorro_vs_mejor_baseline"]
    need_lots = need_fail = need_months = None
    if s > 0 and lo <= 0:
        # distancia de la media al límite inferior observado; el intervalo se angosta con la raíz del tamaño
        need_lots = int(np.ceil(len(yho) * ((s - lo) / s) ** 2))
        need_fail = int(np.ceil(need_lots * yho.mean()))
        need_months = max(0.0, (need_lots - len(yho)) / base["lotes_por_mes"])

    # 3) Riesgos: fuga por partición aleatoria, deriva y cambio de la tasa de falla, costos.
    Xa, Xb, ya, yb = _stratified_split(Xtr, ytr, 0.25, aj.semilla)
    pr_random = pr_auc(yb, _fresh(elegido, aj.semilla).fit(Xa, ya).predict_proba(Xb)[:, 1])
    psis = pd.Series({c: psi(Xtr[c], Xho[c]) for c in variables}).dropna()
    deriva = float((psis > 0.25).mean()) if len(psis) else 0.0
    tasa_p = float(fisher_exact([[ytr.sum(), len(ytr) - ytr.sum()], [yho.sum(), len(yho) - yho.sum()]])[1])
    sens = []
    for k in (0.5, 1.0, 2.0):
        c = Ajustes(aj.scrap_no_detectado * k, aj.inspeccion, aj.retrabajo_detectado, semilla=aj.semilla)
        u, _ = best_threshold(ytr[i_oof], s_oof, c.costos)
        sens.append(dict(factor=k, razon=c.scrap_no_detectado / c.inspeccion,
                         ahorro=monetizar(yho, p, u, c.costos)["ahorro_vs_mejor_baseline"]))
    top = (pd.DataFrame({"variable": variables, "importancia": _permutation_importance(model, Xho, yho, pr, aj.semilla), "psi": [psis.get(v, np.nan) for v in variables]})
           .sort_values("importancia", ascending=False).head(8).reset_index(drop=True))

    r = {**base, "evaluable": True, "modelo": elegido, "cv_pr_auc": cv, "umbral": umbral, "pr_auc": pr, "pr_auc_ic": (pr_lo, pr_hi),
         "lift10": lift_at_k(yho, p), "mon": mon, "ahorro": s, "ahorro_ic": (lo, hi), "prob_ahorro": prob,
         "lotes_necesarios": need_lots, "fallas_necesarias": need_fail, "meses_necesarios": need_months,
         "pr_auc_aleatorio": pr_random, "pr_auc_temporal_cv": cv[elegido], "pr_auc_temporal_ic": cv_ic, "deriva": deriva, "psi_top": psis.sort_values(ascending=False).head(5),
         "prevalencia_entreno": float(ytr.mean()), "p_cambio_tasa": tasa_p, "sensibilidad": sens, "importancia": top,
         "puntajes": pd.DataFrame({"fecha": df["fecha"].iloc[ho].to_numpy(), "y": yho, "p": p})}
    r["dictamen"], r["motivo"] = verdict(r)
    r["hallazgos"], r["riesgos"] = findings(r)
    return r


def verdict(r) -> tuple[str, str]:
    lo, hi = r["ahorro_ic"]
    signal = r["pr_auc_ic"][0] > r["prevalencia_eval"]
    nivel = f"{r['ajustes'].nivel:.0%}"
    if signal and lo > 0:
        return DESPLEGAR, f"Hay señal en lotes futuros y el ahorro es positivo con confianza del {nivel}."
    if hi < 0:
        return NO_DESPLEGAR, "El modelo cuesta más que la mejor alternativa simple en todo el intervalo."
    if r["ahorro"] <= 0:
        why = "ordena mejor que el azar, pero" if signal else "no ordena mejor que el azar y"
        return NO_DESPLEGAR, f"El modelo {why} con estos costos no ahorra frente a la mejor alternativa simple."
    if r["meses_necesarios"] is not None and r["meses_necesarios"] <= r["ajustes"].meses_max:
        return RECOLECTAR, (f"El ahorro estimado es positivo, pero su intervalo incluye cero. Con ≈ {r['fallas_necesarias']} fallas en el "
                            f"periodo de evaluación ({_meses(r['meses_necesarios'])} más al ritmo actual) se podría demostrar.")
    return NO_DESPLEGAR, (f"El ahorro estimado es positivo pero tan pequeño que demostrarlo tomaría ≈ {r['meses_necesarios']:.0f} meses más, "
                          f"por encima del plazo de {r['ajustes'].meses_max:.0f}.")


def _meses(m: float) -> str:
    return "menos de un mes" if m < 1 else f"≈ {m:.0f} meses"


def _f(titulo, criterio, evidencia, resultado, riesgo=""):
    return dict(titulo=titulo, criterio=criterio, evidencia=evidencia, resultado=resultado, riesgo=riesgo)


def findings(r):
    aj, mon = r["ajustes"], r["mon"]
    nivel = f"{aj.nivel:.0%}"
    lo, hi = r["ahorro_ic"]
    h = []
    gap = r["pr_auc_aleatorio"] / r["pr_auc_temporal_cv"] if r["pr_auc_temporal_cv"] else np.nan
    fuga = r["pr_auc_aleatorio"] > r["pr_auc_temporal_ic"][1]  # por encima de lo que explica el ruido
    h.append(_f("1 · Validación sin información del futuro",
                "El modelo se evalúa como se usaría: entrenado con lotes anteriores y probado en posteriores (ISO 9001 · 9.1.3).",
                f"Evaluación única en los {r['lotes_eval']} lotes posteriores al {r['corte']:%d/%m/%Y}. PR-AUC con partición al azar "
                f"{r['pr_auc_aleatorio']:.3f} contra {r['pr_auc_temporal_cv']:.3f} con ventanas cronológicas "
                f"(IC {nivel} {r['pr_auc_temporal_ic'][0]:.3f}–{r['pr_auc_temporal_ic'][1]:.3f}).",
                "Observación" if fuga else "Conforme",
                "El proceso cambia con el tiempo: una validación al azar habría inflado el desempeño "
                f"{gap - 1:.0%}." if fuga else ""))
    signal = r["pr_auc_ic"][0] > r["prevalencia_eval"]
    h.append(_f("2 · Señal predictiva",
                "En lotes futuros, el modelo ordena las fallas mejor que el azar: PR-AUC con su intervalo por encima de la tasa de falla.",
                f"PR-AUC {r['pr_auc']:.3f} (IC {nivel} {r['pr_auc_ic'][0]:.3f}–{r['pr_auc_ic'][1]:.3f}) contra tasa de falla "
                f"{r['prevalencia_eval']:.3f}. El 10 % de lotes con mayor riesgo concentra {r['lift10']:.1f} veces la tasa promedio. "
                f"Modelo: {r['modelo']}.",
                "Conforme" if signal else "No conforme", "" if signal else "La señal no se distingue del azar con estos datos."))
    ok = lo > 0
    h.append(_f("3 · Beneficio económico",
                "Ahorro contra la mejor alternativa simple (no inspeccionar o inspeccionar todo) con intervalo por encima de cero.",
                f"Con el umbral elegido en el pasado: {mon['detectadas']} de {mon['fallas']} fallas detectadas, {mon['falsas_alarmas']} falsas "
                f"alarmas. Ahorro {r['ahorro']:,.0f} MXN (IC {nivel} {lo:,.0f} a {hi:,.0f}; probabilidad de ahorro positivo {r['prob_ahorro']:.0%}).",
                "Conforme" if ok else "No conforme", "" if ok else "El ahorro no está demostrado."))
    if r["lotes_necesarios"]:
        ev = (f"{r['fallas_eval']} fallas en el periodo evaluado. Para que el intervalo excluya cero harían falta ≈ {r['lotes_necesarios']:,} "
              f"lotes evaluados (≈ {r['fallas_necesarias']} fallas): {_meses(r['meses_necesarios'])} más a {r['lotes_por_mes']:.0f} lotes por mes.")
    else:
        ev = f"{r['fallas_eval']} fallas en el periodo evaluado de un total de {r['fallas']}."
    h.append(_f("4 · Suficiencia de la evidencia", "El número de fallas evaluadas permite separar el ahorro de cero.", ev,
                "Conforme" if ok else ("Observación" if r["lotes_necesarios"] else "No conforme")))
    cambio = r["p_cambio_tasa"] < 0.05
    h.append(_f("5 · Estabilidad del proceso",
                "Las variables y la tasa de falla del periodo evaluado se parecen a las del entrenamiento (ISO 9001 · 7.1.5).",
                f"{r['deriva']:.0%} de las variables con PSI > 0.25. Tasa de falla {r['prevalencia_entreno']:.1%} en entrenamiento contra "
                f"{r['prevalencia_eval']:.1%} evaluada (Fisher p = {r['p_cambio_tasa']:.3f}).",
                "Observación" if (r["deriva"] > 0.2 or cambio) else "Conforme",
                "Reentrenar periódicamente y verificar calibración de instrumentos." if (r["deriva"] > 0.2 or cambio) else ""))
    s = r["sensibilidad"]
    signs = {x["ahorro"] > 0 for x in s}
    h.append(_f("6 · Sensibilidad a los costos", "La conclusión no depende de un supuesto de costo frágil.",
                "Ahorro con scrap a la mitad, igual y al doble: " + ", ".join(f"{x['ahorro']:,.0f}" for x in s) + " MXN.",
                "Conforme" if len(signs) == 1 else "Observación",
                "" if len(signs) == 1 else "El signo del ahorro cambia con la razón de costos: validar el costo real del scrap."))
    riesgos = ["El puntaje ordena lotes; no es una probabilidad calibrada.",
               "Las variables deben medirse antes de conocer el resultado; una variable posterior a la inspección inflaría todo.",
               "Si el proceso cambia (material, molde, receta), el modelo pierde validez: monitorear deriva y reentrenar."]
    return h, riesgos
