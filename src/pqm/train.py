"""Entrenamiento y evaluación de extremo a extremo: python -m pqm.train"""
import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pqm import drift, explain
from pqm.config import FIGURES, MODELS, ROOT, SEED, CostosCalidad
from pqm.data import load_secom, split_temporal
from pqm.evaluation import (best_threshold, bootstrap_ci, expanding_window_folds, lift_at_k, monetizar, pr_auc,
                            roc_auc)
from pqm.models import build_pipeline, peso_positivo

warnings.filterwarnings("ignore")
IMPUTADORES = ["none", "median", "knn"]


def oof_temporal(X, y, imputer: str, n_folds: int = 4):
    idx, p = [], []
    for a, b in expanding_window_folds(len(X), n_folds):
        m = build_pipeline(imputer, peso_positivo(y.iloc[a])).fit(X.iloc[a], y.iloc[a])
        idx += list(b)
        p += list(m.predict_proba(X.iloc[b])[:, 1])
    return np.array(idx), np.array(p)


def oof_aleatorio(X, y, imputer: str):
    idx, p = [], []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = build_pipeline(imputer, peso_positivo(y.iloc[a])).fit(X.iloc[a], y.iloc[a])
        idx += list(b)
        p += list(m.predict_proba(X.iloc[b])[:, 1])
    return np.array(idx), np.array(p)


def main() -> dict:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("secom-calidad-predictiva")
    costos = CostosCalidad()
    X, y, t = load_secom()
    tr, ho = split_temporal(len(X), 0.2)
    Xtr, ytr, Xho, yho = X.iloc[tr], y.iloc[tr], X.iloc[ho], y.iloc[ho]
    res: dict = {"n_lotes": len(X), "fallas": int(y.sum()), "prevalencia": float(y.mean()),
                 "periodo": [str(t.min()), str(t.max())], "n_train": len(Xtr), "n_holdout": len(Xho),
                 "fallas_holdout": int(yho.sum()), "prevalencia_holdout": float(yho.mean())}

    # 1) Comparación de imputación con validación temporal (solo con datos de entrenamiento)
    comp = {}
    for imp in IMPUTADORES:
        with mlflow.start_run(run_name=f"cv-temporal-{imp}"):
            idx, p = oof_temporal(Xtr, ytr, imp)
            yy = ytr.iloc[idx].values
            comp[imp] = {"pr_auc": pr_auc(yy, p), "roc_auc": roc_auc(yy, p), "lift10": lift_at_k(yy, p),
                         "pr_auc_ic95": bootstrap_ci(yy, p, n=500), "prevalencia": float(yy.mean())}
            mlflow.log_param("imputer", imp)
            mlflow.log_metrics({k: v for k, v in comp[imp].items() if isinstance(v, float)})
    res["comparacion_imputacion"] = comp
    mejor = max(comp, key=lambda k: comp[k]["pr_auc"])
    res["imputacion_elegida"] = mejor

    # 2) Fuga de información: misma configuración con partición aleatoria
    idx, p = oof_aleatorio(Xtr, ytr, mejor)
    res["validacion_aleatoria"] = {"pr_auc": pr_auc(ytr.iloc[idx].values, p), "roc_auc": roc_auc(ytr.iloc[idx].values, p)}

    # 3) Umbral por costo con predicciones fuera de muestra del periodo de entrenamiento
    idx, p_oof = oof_temporal(Xtr, ytr, mejor)
    umbral, _ = best_threshold(ytr.iloc[idx].values, p_oof, costos)
    res["umbral"] = umbral

    # 4) Holdout cronológico (se evalúa una sola vez)
    with mlflow.start_run(run_name="holdout-final"):
        model = build_pipeline(mejor, peso_positivo(ytr)).fit(Xtr, ytr)
        p = model.predict_proba(Xho)[:, 1]
        mon = monetizar(yho.values, p, umbral, costos)
        res["holdout"] = {"pr_auc": pr_auc(yho, p), "pr_auc_ic95": bootstrap_ci(yho, p, n=1000), "roc_auc": roc_auc(yho, p),
                          "lift10": lift_at_k(yho, p), **mon, "sensores_usados": len(model[0].final_),
                          "sensores_colineales_eliminados": model[0].n_colineales_}
        rng = np.random.default_rng(SEED)
        ah = []
        for _ in range(1000):
            i = rng.integers(0, len(yho), len(yho))
            yb = yho.values[i]
            ah.append(min(costos.sin_modelo(yb), costos.inspeccionar_todo(yb)) - costos.costo(yb, p[i] >= umbral))
        res["holdout"]["ahorro_ic95"] = [float(np.percentile(ah, 2.5)), float(np.percentile(ah, 97.5))]
        res["holdout"]["prob_ahorro_positivo"] = float(np.mean(np.array(ah) > 0))
        mlflow.log_params({"imputer": mejor, "umbral": umbral})
        mlflow.log_metrics({"holdout_pr_auc": res["holdout"]["pr_auc"], "ahorro": mon["ahorro_vs_mejor_baseline"]})

    # 5) Sensibilidad del ahorro a la razón costo de scrap / costo de inspección
    sens = []
    for razon in [5, 10, 20, 40, 80]:
        c = CostosCalidad(scrap_no_detectado=250 * razon)
        u, _ = best_threshold(ytr.iloc[idx].values, p_oof, c)
        m2 = monetizar(yho.values, p, u, c)
        sens.append({"razon_scrap_inspeccion": razon, "umbral": u, "ahorro": m2["ahorro_vs_mejor_baseline"],
                     "ahorro_pct": m2["ahorro_vs_mejor_baseline"] / m2["costo_mejor_baseline"]})
    res["sensibilidad_costos"] = sens

    # 6) SHAP sobre el holdout y estabilidad del ranking
    sv, Z, base = explain.shap_values(model, Xho)
    glob = explain.resumen_global(sv, Z, 15)
    res["shap_top"] = glob.to_dict("records")
    top10 = set(glob.sensor[:10])
    solapes = []
    for k in range(10):
        r = np.random.default_rng(k)
        i = np.sort(r.choice(len(Xtr), int(0.8 * len(Xtr)), replace=False))
        mk = build_pipeline("median", peso_positivo(ytr.iloc[i])).fit(Xtr.iloc[i], ytr.iloc[i])
        svk, _, _ = explain.shap_values(mk, Xho)
        solapes.append(len(top10 & set(svk.abs().mean().sort_values(ascending=False).head(10).index)) / 10)
    res["estabilidad_shap_top10"] = {"solape_medio": float(np.mean(solapes)), "min": float(np.min(solapes)), "max": float(np.max(solapes))}

    # 7) Deriva: entrenamiento vs holdout y por mes
    usados = list(model[0].final_)
    pt = drift.psi_table(Xtr, Xho, usados)
    res["deriva"] = {"sensores_con_deriva": int((pt.psi > 0.25).sum()), "sensores_vigilar": int(((pt.psi > 0.1) & (pt.psi <= 0.25)).sum()),
                     "total": len(usados), "top": pt.head(10).to_dict("records")}
    res["tasa_falla_mensual"] = y.groupby(t.dt.to_period("M").astype(str)).mean().round(4).to_dict()

    # 8) Modelo de despliegue: reentrenado con todos los lotes; umbral de la validación temporal sobre todos los datos
    idx_all, p_all = oof_temporal(X, y, mejor)
    umbral_final, _ = best_threshold(y.iloc[idx_all].values, p_all, costos)
    final = build_pipeline(mejor, peso_positivo(y)).fit(X, y)
    Zf = final[0].transform(X)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "model.joblib", compress=3)
    meta = {"version": "1.0.0", "imputer": mejor, "umbral": umbral_final, "features": list(Zf.columns),
            "todas_las_features": list(X.columns), "costos": costos.__dict__, "entrenado_con": len(X), "prevalencia": float(y.mean()),
            "sensores_criticos": list(glob.sensor[:10])}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # muestra de referencia (datos de entrenamiento ya transformados) para vigilar deriva en producción
    Zf.sample(min(len(Zf), 600), random_state=SEED).to_csv(MODELS / "referencia.csv", index=False)

    pd.DataFrame({"timestamp": t.iloc[ho].values, "y": yho.values, "p": p}).to_csv(ROOT / "reports/holdout_predictions.csv", index=False)
    # figuras
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_figuras(res, yho.values, p, sv, Z, glob)
    (ROOT / "reports/metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def plot_figuras(res, yho, p, sv, Z, glob):
    from sklearn.metrics import precision_recall_curve
    az, gr, rj = "#0B5394", "#8A94A3", "#C62828"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    pr, rc, _ = precision_recall_curve(yho, p)
    ax.plot(rc, pr, color=az, lw=2, label=f"Modelo (PR-AUC {res['holdout']['pr_auc']:.3f})")
    ax.axhline(yho.mean(), color=gr, ls="--", label=f"Azar = prevalencia ({yho.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precisión"); ax.set_title("Holdout cronológico (último 20 % de lotes)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "pr_holdout.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    vals = [res["validacion_aleatoria"]["pr_auc"], res["comparacion_imputacion"][res["imputacion_elegida"]]["pr_auc"]]
    bars = ax.bar(["Partición aleatoria\n(filtra el futuro)", "Validación temporal\n(honesta)"], vals, color=[rj, az], width=.5)
    ax.axhline(res["prevalencia"], color=gr, ls="--"); ax.text(1.32, res["prevalencia"], "azar", color=gr, va="center")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .003, f"{v:.3f}", ha="center")
    ax.set_ylabel("PR-AUC"); ax.set_title("Misma configuración, distinta validación"); fig.tight_layout(); fig.savefig(FIGURES / "fuga_vs_temporal.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    s = pd.DataFrame(res["sensibilidad_costos"])
    ax.bar(s.razon_scrap_inspeccion.astype(str), s.ahorro, color=[az if a > 0 else rj for a in s.ahorro])
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("Costo de scrap ÷ costo de inspección"); ax.set_ylabel("Ahorro vs mejor política sin modelo (MXN)")
    ax.set_title("¿Cuándo conviene el modelo?"); fig.tight_layout(); fig.savefig(FIGURES / "sensibilidad_costos.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    g = glob.head(10).iloc[::-1]
    ax.barh(g.sensor, g.shap_medio_abs, color=az); ax.set_xlabel("|SHAP| medio (holdout)"); ax.set_title("Sensores que más mueven el riesgo")
    fig.tight_layout(); fig.savefig(FIGURES / "shap_top.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
