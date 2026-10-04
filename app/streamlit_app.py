"""Dashboard para el Gerente de Planta: ¿cuánto habríamos ahorrado con el modelo? (SECOM, holdout cronológico)."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pqm.config import CostosCalidad  # noqa: E402
from pqm.evaluation import monetizar  # noqa: E402
from pqm.explain import cargar_alias, nombre  # noqa: E402

st.set_page_config(page_title="Calidad predictiva SECOM", layout="wide")


@st.cache_data
def cargar():
    m = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    h = pd.read_csv(ROOT / "reports/holdout_predictions.csv", parse_dates=["timestamp"])
    return m, h


m, h = cargar()
alias = cargar_alias()
ho = m["holdout"]

st.title("Calidad predictiva · ¿Qué habría pasado si usábamos el modelo?")
st.caption(f"Periodo evaluado (holdout cronológico, nunca visto en entrenamiento): {h.timestamp.min():%d-%b-%Y} a "
           f"{h.timestamp.max():%d-%b-%Y} · {len(h)} lotes · {int(h.y.sum())} fallas. "
           "Costos en MXN, supuestos editables a la izquierda.")

with st.sidebar:
    st.header("Supuestos de costo (MXN/lote)")
    scrap = st.number_input("Scrap no detectado", 100, 100000, 5000, 250)
    insp = st.number_input("Inspección de un lote marcado", 10, 5000, 250, 50)
    retr = st.number_input("Retrabajo de una falla detectada", 0, 20000, 500, 100)
    costos = CostosCalidad(scrap, insp, retr)
    umbral = st.slider("Umbral de alarma (prob.)", 0.01, 0.60, float(round(m["umbral"], 3)), 0.001, format="%.3f",
                       help="Valor óptimo en validación temporal previa; moverlo muestra el compromiso.")

mon = monetizar(h.y.values, h.p.values, umbral, costos)
a = mon["ahorro_vs_mejor_baseline"]
t1, t2, t3, t4 = st.tabs(["Impacto económico", "Qué sensores importan", "Deriva", "Honestidad estadística"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ahorro vs mejor alternativa simple", f"${a:,.0f}", f"{a / mon['costo_mejor_baseline']:+.1%}")
    c2.metric("Fallas detectadas", f"{mon['detectadas']} de {mon['fallas']}")
    c3.metric("Falsas alarmas", mon["falsas_alarmas"], f"${mon['falsas_alarmas'] * costos.inspeccion:,.0f}", delta_color="off")
    c4.metric("Scrap no detectado", mon["no_detectadas"], f"${mon['no_detectadas'] * costos.scrap_no_detectado:,.0f}", delta_color="off")
    lo, hi = ho["ahorro_ic95"]
    st.info(f"**Lectura para el gerente:** con el umbral óptimo (costos base) el modelo habría ahorrado "
            f"**\\${ho['ahorro_vs_mejor_baseline']:,.0f}** en {len(h)} lotes, pero el intervalo de confianza 95 % va de "
            f"**{lo:,.0f} a {hi:,.0f} MXN**: con 17 fallas en el holdout **no podemos afirmar** que el ahorro sea positivo "
            f"(probabilidad bootstrap de ahorro > 0: {ho['prob_ahorro_positivo']:.0%}).")
    z = [[mon["verdaderos_negativos"] if "verdaderos_negativos" in mon else len(h) - mon["marcados"] - mon["no_detectadas"], mon["falsas_alarmas"]],
         [mon["no_detectadas"], mon["detectadas"]]]
    costo_cel = [[0, mon["falsas_alarmas"] * costos.inspeccion],
                 [mon["no_detectadas"] * costos.scrap_no_detectado, mon["detectadas"] * (costos.inspeccion + costos.retrabajo_detectado)]]
    txt = [[f"Lote bueno, sin alarma<br><b>{z[0][0]}</b><br>$0", f"Falsa alarma<br><b>{z[0][1]}</b><br>${costo_cel[0][1]:,.0f}"],
           [f"Scrap no detectado<br><b>{z[1][0]}</b><br>${costo_cel[1][0]:,.0f}", f"Falla detectada<br><b>{z[1][1]}</b><br>${costo_cel[1][1]:,.0f}"]]
    fig = go.Figure(go.Heatmap(z=costo_cel, text=txt, texttemplate="%{text}", x=["Sin alarma", "Alarma"],
                               y=["Lote bueno", "Lote defectuoso"], colorscale="Reds", showscale=False))
    fig.update_layout(title="Matriz de confusión monetizada", height=360, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    cmp = pd.DataFrame({"Estrategia": ["Sin modelo", "Inspeccionar todo", "Con modelo"],
                        "Costo MXN": [mon["costo_sin_modelo"], mon["costo_inspeccionar_todo"], mon["costo_con_modelo"]]})
    st.plotly_chart(px.bar(cmp, x="Estrategia", y="Costo MXN", text_auto=",.0f", title="Costo total de calidad en el periodo"), width="stretch")
    st.subheader("Sensibilidad: ¿y si el scrap fuera más (o menos) caro?")
    s = pd.DataFrame(m["sensibilidad_costos"])
    st.plotly_chart(px.bar(s, x="razon_scrap_inspeccion", y="ahorro", text_auto=",.0f",
                           labels={"razon_scrap_inspeccion": "Costo scrap / costo inspección", "ahorro": "Ahorro MXN"}), width="stretch")
    st.caption("El beneficio cambia de signo según la razón de costos: el modelo solo conviene en una banda intermedia.")

with t2:
    st.caption("⚠️ Los nombres físicos son **alias ilustrativos**: SECOM no publica qué mide cada sensor. "
               "Sirven para mostrar cómo se vería con el diccionario real de tags de la planta.")
    sh = pd.DataFrame(m["shap_top"])
    sh["Sensor (alias)"] = sh.sensor.map(lambda s: nombre(s, alias))
    st.plotly_chart(px.bar(sh.sort_values("shap_medio_abs"), x="shap_medio_abs", y="Sensor (alias)", orientation="h",
                           title="Importancia SHAP media (holdout)"), width="stretch")
    st.dataframe(sh[["Sensor (alias)", "sensor", "shap_medio_abs", "direccion"]], hide_index=True)
    e = m["estabilidad_shap_top10"]
    st.warning(f"Estabilidad: al reentrenar con 80 % de los datos, solo {e['solape_medio']:.0%} del top-10 se mantiene "
               f"(rango {e['min']:.0%}–{e['max']:.0%}). El ranking es una pista a investigar con ingeniería, no una causa raíz.")

with t3:
    d = m["deriva"]
    c1, c2 = st.columns(2)
    c1.metric("Sensores con deriva (PSI>0.25)", f"{d['sensores_con_deriva']} de {d['total']}")
    c2.metric("En vigilancia (0.1–0.25)", d["sensores_vigilar"])
    td = pd.DataFrame(d["top"])
    td["Sensor (alias)"] = td.sensor.map(lambda s: nombre(s, alias))
    st.dataframe(td, hide_index=True)
    tm = pd.Series(m["tasa_falla_mensual"]).rename("tasa de falla").reset_index()
    st.plotly_chart(px.bar(tm, x="index", y="tasa de falla", title="Tasa de falla por mes (el proceso no es estacionario)"), width="stretch")
    st.caption("Esto justifica reentrenar periódicamente y monitorear PSI vía el endpoint /drift de la API.")

with t4:
    c = m["comparacion_imputacion"]
    st.write(f"**PR-AUC en holdout:** {ho['pr_auc']:.3f} (IC95 {ho['pr_auc_ic95'][0]:.3f}–{ho['pr_auc_ic95'][1]:.3f}) "
             f"vs. prevalencia {m['prevalencia_holdout']:.3f}. **ROC-AUC:** {ho['roc_auc']:.2f}.")
    st.write(f"**Fuga por partición aleatoria:** PR-AUC {m['validacion_aleatoria']['pr_auc']:.3f} con split aleatorio vs "
             f"{c[m['imputacion_elegida']]['pr_auc']:.3f} con validación temporal: el split aleatorio infla el resultado.")
    st.image(str(ROOT / "reports/figures/fuga_vs_temporal.png"))
    st.write("Conclusión: hay señal débil, pero con estos datos el beneficio económico no está demostrado. "
             "Se publica así a propósito.")
