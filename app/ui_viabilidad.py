"""Modo «Evalúa tu proceso»: dictamen de viabilidad de un modelo de calidad con los lotes del usuario, en el navegador."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pqm import viability as vb
from pqm import viability_intake as vi
from pqm import viability_report as rp

NEGRO, GRIS, LINEA, VERDE, ROJO, AMARILLO = "#0E0F11", "#4A4E54", "#C9CDD1", "#1F7A3D", "#D7261E", "#FFD21F"
BADGE = {"Conforme": ":green[**Conforme**]", "No conforme": ":red[**No conforme**]", "Observación": ":orange[**Observación**]"}
CAJA = {vb.DESPLEGAR: st.success, vb.NO_DESPLEGAR: st.error, vb.RECOLECTAR: st.warning}


def _md(text: str) -> str:
    return text.replace("$", "\\$")


@st.cache_data(show_spinner=False)
def _template():
    return vi.template_bytes()


@st.cache_data(show_spinner=False)
def _example_bytes(name: str) -> bytes:
    return vi.to_csv_bytes(vi.example_with_errors() if name == "errores" else vi.example(name))


@st.cache_data(show_spinner="Evaluando con validación cronológica… (en el navegador puede tardar un minuto)")
def _run(content: bytes, name: str, aj: vb.Ajustes):
    raw = vi.read_table(content, name)
    df, variables, issues = vi.validate(raw)
    return len(raw), len(df), issues, vb.evaluate(df, variables, aj)


def _strategies_fig(r):
    m = r["mon"]
    names = ["Sin modelo", "Inspeccionar todo", "Con modelo"]
    vals = [m["costo_sin_modelo"], m["costo_inspeccionar_todo"], m["costo_con_modelo"]]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[LINEA, LINEA, NEGRO], text=[f"${v:,.0f}" for v in vals], textposition="outside",
                           hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=f"Costo de calidad en los {r['lotes_eval']} lotes evaluados (en negro, con modelo)", height=330,
                      margin=dict(l=70, r=10, t=50, b=40), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color=NEGRO,
                      font_family="Barlow, system-ui, sans-serif", yaxis=dict(tickprefix="$", gridcolor="#E9EBE7", range=[0, max(vals) * 1.18]))
    return fig


def render():
    st.subheader("Dictamen de viabilidad de un modelo de calidad")
    st.markdown("Antes de invertir en un modelo predictivo, sube el histórico de lotes de tu proceso y obtén un dictamen de una página: "
                "**desplegar, no desplegar o recolectar más datos**, con hallazgos en lenguaje de auditoría (criterio, evidencia, "
                "resultado, riesgo). **El archivo se procesa en tu navegador: no se envía a ningún servidor.**")
    a, b = st.columns([1, 2])
    a.download_button("Descargar plantilla de Excel (con datos de ejemplo)", _template(), "plantilla_lotes.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    ejemplo = b.selectbox("O prueba con un proceso de ejemplo", ["(ninguno)", *vi.EXAMPLES, "Inyección de plástico · con errores de captura"])
    st.caption("Columnas: *fecha, resultado (pasa/falla)*, opcional *lote*, y una columna numérica por variable de proceso medida antes "
               "del resultado. Acepta .xlsx o .csv. Mínimo 200 lotes y 20 fallas; ideal 1,000 lotes.")
    up = st.file_uploader("Histórico de lotes (.xlsx o .csv)", type=["xlsx", "csv"])
    proceso = st.text_input("Nombre del proceso (aparece en el dictamen)", "Mi proceso")
    c1, c2, c3 = st.columns(3)
    scrap = c1.number_input("Dejar pasar un lote malo (MXN)", 1, 10_000_000, 5_000, 250,
                            help="Scrap, reclamo o retrabajo en cliente de un lote defectuoso que no se detectó.")
    insp = c2.number_input("Inspeccionar un lote marcado (MXN)", 0, 1_000_000, 250, 50)
    retr = c3.number_input("Retrabajar una falla detectada (MXN)", 0, 1_000_000, 500, 50)

    if up is not None:
        content, name = up.getvalue(), up.name
    elif ejemplo != "(ninguno)":
        key = "errores" if "errores" in ejemplo else ejemplo
        content, name = _example_bytes(key), "ejemplo.csv"
    else:
        st.info("Para probar, elige un proceso de ejemplo (hay uno por cada dictamen posible) o sube la plantilla.")
        return

    aj = vb.Ajustes(float(scrap), float(insp), float(retr))
    try:
        n_raw, n_ok, issues, r = _run(content, name, aj)
    except vi.DataError as exc:
        st.error(str(exc))
        return
    errors = [i for i in issues if i.nivel == "error"]
    st.markdown(f"**{n_ok:,} de {n_raw:,} lotes válidos** · {len(errors)} filas excluidas · {r['variables']} variables · "
                f"{r['fallas']} fallas ({r['fallas_eval']} en el periodo evaluado)")
    if issues:
        with st.expander("Ver errores y avisos de los datos", expanded=bool(errors)):
            st.dataframe(pd.DataFrame([dict(Nivel=i.nivel, Fila=str(i.fila) if i.fila else "—", Detalle=i.mensaje) for i in issues]),
                         hide_index=True)

    CAJA[r["dictamen"]](_md(f"**Dictamen: {r['dictamen']}.** {r['motivo']}"))
    if r.get("evaluable"):
        lo, hi = r["ahorro_ic"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("PR-AUC en lotes futuros", f"{r['pr_auc']:.2f}", border=True)
        k2.metric("Fallas detectadas", f"{r['mon']['detectadas']} de {r['mon']['fallas']}", border=True)
        k3.metric("Ahorro vs mejor alternativa", _md(f"${r['ahorro']:,.0f}"), border=True)
        k4.metric(f"Intervalo {aj.nivel:.0%} del ahorro", f"{lo / 1e3:+,.0f} a {hi / 1e3:+,.0f} mil", border=True)
        st.caption(f"Azar (tasa de falla del periodo evaluado) = {r['prevalencia_eval']:.3f} · {r['mon']['falsas_alarmas']} falsas alarmas · "
                   f"probabilidad de ahorro positivo {r['prob_ahorro']:.0%}.")
    if r["hallazgos"]:
        st.markdown("#### Hallazgos")
        for h in r["hallazgos"]:
            st.markdown(_md(f"**{h['titulo']}** · {BADGE[h['resultado']]}  \n{h['evidencia']}"))
            st.caption(_md(f"Criterio: {h['criterio']}" + (f" Riesgo: {h['riesgo']}" if h["riesgo"] else "")))
    if r.get("evaluable"):
        left, right = st.columns([3, 2])
        left.plotly_chart(_strategies_fig(r), theme=None)
        right.markdown("**Variables que más pesan** (pista, no causa)")
        imp = r["importancia"][r["importancia"]["importancia"] > 0]
        right.dataframe(pd.DataFrame({"Variable": imp["variable"], "Importancia": imp["importancia"].round(3),
                                      "Deriva (PSI)": imp["psi"].round(2)}), hide_index=True)
        st.caption(f"Modelo elegido con ventanas cronológicas del pasado: {r['modelo']} (PR-AUC "
                   + ", ".join(f"{k} {v:.3f}" for k, v in r["cv_pr_auc"].items()) + f"). Umbral por costo: {r['umbral']:.3f}.")
    html_report = rp.report_html(r, issues, proceso, (n_raw, n_ok))
    st.download_button("Descargar dictamen de una página (HTML, imprimible a PDF)", html_report.encode("utf-8"),
                       "dictamen_viabilidad.html", "text/html", type="primary")
