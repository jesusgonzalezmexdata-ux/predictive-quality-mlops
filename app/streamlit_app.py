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
t0, t1, t2, t3, t4 = st.tabs(["Qué hacer con cada lote", "Impacto económico", "Qué sensores importan", "Deriva", "Honestidad estadística"])

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - r), min(1.0, c + r)


with t0:
    st.caption("Regla de decisión por lote. Cada acción lleva su referencia al sistema de gestión (ISO 9001:2015).")
    hh = h.reset_index(drop=True)
    i = st.selectbox("Lote del holdout", hh.index[::-1], format_func=lambda k: f"{hh.timestamp[k]:%d-%b %H:%M} · riesgo {hh.p[k]:.1%}")
    pr = hh.p[i]
    vec = hh.assign(d=(hh.p - pr).abs()).nsmallest(40, "d")  # 40 lotes con puntaje más parecido
    lo_, hi_ = wilson(int(vec.y.sum()), len(vec))
    if pr >= umbral:
        color, estado, accion = "🔴", "ALTO RIESGO", ("Retener el lote, inspección al 100 % y registrarlo como posible salida no conforme "
                                                       "(ISO 9001 · 8.7 Control de salidas no conformes).")
    elif pr >= umbral / 2:
        color, estado, accion = "🟡", "VIGILAR", "Muestreo reforzado y registro del resultado para el seguimiento del desempeño (ISO 9001 · 9.1.3)."
    else:
        color, estado, accion = "🟢", "NORMAL", "Liberar con el flujo habitual."
    st.subheader(f"{color} {estado}")
    st.write(f"**Acción requerida:** {accion}")
    c1, c2 = st.columns(2)
    c1.metric("Puntaje del modelo", f"{pr:.1%}", f"umbral de alarma {umbral:.1%}", delta_color="off")
    c2.metric("Tasa observada de falla en lotes con puntaje parecido", f"{vec.y.mean():.0%}",
              f"IC95 % {lo_:.0%}–{hi_:.0%} (n={len(vec)})", delta_color="off")
    st.caption("El puntaje no es una probabilidad calibrada: por eso se muestra, junto a él, qué fracción de lotes con puntaje similar falló "
               "realmente y su intervalo (Wilson). Con pocas fallas el intervalo es amplio y debe leerse como tal.")
    st.markdown("##### Reglas a nivel proceso")
    d_ = m["deriva"]
    pct = d_["sensores_con_deriva"] / d_["total"]
    st.write(f"{'🔴' if pct > 0.2 else '🟢'} **Deriva de sensores:** {pct:.0%} con PSI>0.25. " +
             ("**Acción requerida:** reentrenar el modelo y revisar calibración de instrumentos (ISO 9001 · 7.1.5 Recursos de seguimiento y medición)."
              if pct > 0.2 else "Sin acción."))
    tm_ = pd.Series(m["tasa_falla_mensual"])
    mx = tm_.idxmax()
    st.write(f"{'🔴' if tm_.max() > 2 * m['prevalencia'] else '🟢'} **Tasa de falla mensual máxima:** {tm_.max():.1%} ({mx}). " +
             ("**Acción requerida:** análisis de causa raíz y acción correctiva (ISO 9001 · 10.2 No conformidad y acción correctiva)."
              if tm_.max() > 2 * m["prevalencia"] else "Sin acción."))
    st.caption("Los umbrales de estas reglas (20 % de sensores, 2× la tasa media) son criterios de ejemplo a validar con ingeniería de calidad.")

with t1:
    st.caption("Datos trazables para auditoría: ISO 9001 · 8.7 (salidas no conformes) y 9.1.3 (análisis y evaluación). "
               "El costo de la no calidad (COPQ) incluye scrap no detectado, inspección y retrabajo.")
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
