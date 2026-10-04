"""Evidencia para 'Decisiones de diseño': costo vs umbral, valor del recall, estrés (ruido y deriva) y explicaciones del holdout.
Reentrena el modelo SOLO con el periodo de entrenamiento (igual que train.py) y verifica que reproduce reports/holdout_predictions.csv.
Ejecutar desde src/:  python -m pqm.evidence
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from pqm import drift, explain
from pqm.config import FIGURES, ROOT, SEED, CostosCalidad
from pqm.data import load_secom, split_temporal
from pqm.evaluation import pr_auc, roc_auc
from pqm.models import build_pipeline, peso_positivo


def main():
    warnings.filterwarnings("ignore")
    m = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    costos, umbral = CostosCalidad(), m["umbral"]
    X, y, t = load_secom()
    tr, ho = split_temporal(len(X), 0.2)
    Xtr, ytr, Xho, yho = X.iloc[tr], y.iloc[tr], X.iloc[ho], y.iloc[ho]
    model = build_pipeline(m["imputacion_elegida"], peso_positivo(ytr)).fit(Xtr, ytr)
    p = model.predict_proba(Xho)[:, 1]
    ref = pd.read_csv(ROOT / "reports/holdout_predictions.csv")
    assert np.allclose(p, ref.p.values, atol=1e-6), "el reentrenamiento no reproduce las predicciones del holdout"
    res = {}

    # 1) Costo total vs umbral y valor marginal del recall
    ths = np.unique(np.concatenate([[0], np.sort(p), [1.01]]))
    filas = []
    for u in ths:
        f = p >= u
        tp, fp = int((yho.values.astype(bool) & f).sum()), int((~yho.values.astype(bool) & f).sum())
        filas.append({"umbral": float(u), "recall": tp / yho.sum(), "detectadas": tp, "falsas_alarmas": fp, "costo": costos.costo(yho.values, f)})
    cv = pd.DataFrame(filas)
    tabla = cv.sort_values("costo").groupby("detectadas", as_index=False).first().sort_values("detectadas")  # mejor costo por nivel de recall
    tabla["delta_costo_vs_anterior"] = tabla.costo.diff()
    res["recall_vs_costo"] = tabla[["detectadas", "recall", "falsas_alarmas", "costo", "delta_costo_vs_anterior"]].round(3).to_dict("records")
    n, prev = len(yho), yho.mean()
    res["valor_por_punto_recall_por_1000_lotes"] = {
        "fallas_por_1000": 1000 * prev, "ahorro_bruto_por_punto": 1000 * prev * 0.01 * (costos.scrap_no_detectado - costos.inspeccion - costos.retrabajo_detectado),
        "nota": "ahorro bruto por cada punto de recall con 0 falsas alarmas adicionales; cada falsa alarma extra cuesta " + str(int(costos.inspeccion))}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(cv.umbral.clip(lower=1e-3, upper=1), cv.costo, color="tab:blue", label="costo con modelo")
    ax.axhline(costos.sin_modelo(yho.values), color="gray", ls="--", label="sin modelo")
    ax.axhline(costos.inspeccionar_todo(yho.values), color="gray", ls=":", label="inspeccionar todo")
    ax.axvline(umbral, color="tab:red", label=f"umbral elegido en entrenamiento ({umbral:.2f})")
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_xlabel("umbral de alarma"); ax.set_ylabel("costo total MXN (holdout)"); ax.legend(fontsize=8)
    ax.set_title("El umbral se elige por costo, no por F1")
    plt.tight_layout(); plt.savefig(FIGURES / "costo_vs_umbral.png", dpi=130); plt.close()

    # 2) Distribución temporal de las fallas y partición
    mes = pd.DataFrame({"mes": t.dt.to_period("M").astype(str), "y": y})
    g = mes.groupby("mes").y.agg(["mean", "sum", "size"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(g.index, g["mean"] * 100, color="tab:blue")
    for i, (a, b) in enumerate(zip(g["sum"], g["size"])):
        ax.text(i, g["mean"].iloc[i] * 100 + 0.3, f"{a}/{b}", ha="center", fontsize=8)
    ax.set_ylabel("% de lotes con falla"); ax.set_title(f"Tasa de falla por mes · holdout desde {t.iloc[ho.start]:%d-%b}")
    plt.tight_layout(); plt.savefig(FIGURES / "distribucion_temporal.png", dpi=130); plt.close()

    # 3) Pruebas de estrés: ruido gaussiano y corrimiento de sensores críticos
    crit = [c for c in meta["sensores_criticos"] if c in Xho.columns]
    sd = Xtr.std()
    rng = np.random.default_rng(SEED)
    base_scores = p
    estres = {"ruido": [], "corrimiento_criticos": []}

    def evaluar(Xn, etiqueta, clave):
        pn = model.predict_proba(Xn)[:, 1]
        psis = {c: drift.psi(Xho[c], Xn[c]) for c in crit}
        ks = ks_2samp(base_scores, pn)
        estres[clave].append({**etiqueta, "pr_auc": pr_auc(yho.values, pn), "roc_auc": roc_auc(yho.values, pn),
                              "criticos_psi_gt_0.2": int(sum(v > 0.2 for v in psis.values() if v == v)),
                              "ks_p_scores": float(ks.pvalue), "ks_stat": float(ks.statistic),
                              "alerta": bool(sum(v > 0.2 for v in psis.values() if v == v) > 3 or ks.pvalue < 0.05)})

    for nivel in [0, 0.05, 0.1, 0.25, 0.5, 1.0]:
        Xn = Xho + rng.normal(0, 1, Xho.shape) * sd.values * nivel
        evaluar(Xn, {"ruido_sigma": nivel}, "ruido")
    for k in [0, 0.5, 1, 2, 3]:
        Xn = Xho.copy()
        Xn[crit[:5]] = Xn[crit[:5]] + k * sd[crit[:5]]
        evaluar(Xn, {"corrimiento_sigma_5_criticos": k}, "corrimiento_criticos")
    res["estres"] = estres
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, clave, xk in zip(ax, ["ruido", "corrimiento_criticos"], ["ruido_sigma", "corrimiento_sigma_5_criticos"]):
        d = pd.DataFrame(estres[clave])
        a.plot(d[xk], d.pr_auc, "-", color="gray", zorder=1)
        a.scatter(d[xk], d.pr_auc, c=np.where(d.alerta, "tab:red", "tab:green"), s=60, zorder=2)
        a.axhline(yho.mean(), color="gray", ls=":")
        a.text(d[xk].iloc[-1], yho.mean() + 0.001, "azar (prevalencia)", ha="right", fontsize=7)
        a.set_xlabel("ruido añadido (× desv. est. de cada sensor)" if clave == "ruido" else "corrimiento de 5 sensores críticos (× desv. est.)")
        a.set_ylabel("PR-AUC en holdout"); a.set_title("rojo = el monitoreo sin etiquetas alerta · verde = no alerta", fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / "estres.png", dpi=130); plt.close()

    # 4) Explicaciones del holdout para el dashboard (SHAP por lote)
    sv, Z, _ = explain.shap_values(model, Xho)
    np.savez_compressed(ROOT / "reports/holdout_explain.npz", sv=sv.round(4).to_numpy(), z=Z.round(4).to_numpy(), cols=np.array(Z.columns))
    (ROOT / "reports/evidence.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "recall_vs_costo"}, indent=1, default=float)[:3500])
    print(pd.DataFrame(r["recall_vs_costo"]).to_string())
