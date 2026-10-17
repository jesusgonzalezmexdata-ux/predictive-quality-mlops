"""Acabado «Señal de campo» que el tema no cubre en todas partes: tipografía Barlow (también en la demo del navegador),
botón principal amarillo, cifras condensadas y pie de página con el aviso de derechos."""
import streamlit as st

CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;600&display=swap");
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th { font-family: Barlow, system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: "Barlow Condensed", "Arial Narrow", sans-serif; font-weight: 700; }
[data-testid="stBaseButton-primary"], button[kind="primary"] { background: #FFD21F !important; color: #0E0F11 !important; border: 0 !important; font-weight: 600 !important; }
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover { background: #F2C400 !important; }
[data-testid="stMetricValue"] { font-family: "Barlow Condensed", "Arial Narrow", sans-serif; font-weight: 600; }
[data-testid="stMetric"] { background: #FFFFFF; border-radius: 4px; }
/* Pie de página: se declara al inicio y CSS lo manda al final, así aparece aunque la página se detenga antes. */
[data-testid="stElementContainer"]:has(.pie) { order: 9999; }
.pie { margin-top: 48px; padding: 16px 0 8px; border-top: 1px solid #C9CDD1; color: #4A4E54; font: 400 13px/1.5 Barlow, system-ui, sans-serif; }
.pie a { color: #0E0F11; font-weight: 600; }
</style>
"""

PIE = (
    '<footer class="pie">© 2026 Jesús R. L. González · Código bajo '
    '<a href="https://github.com/jesusgonzalezmexdata-ux/predictive-quality-mlops/blob/main/LICENSE" target="_blank" rel="noopener">licencia MIT</a>'
    ' · Caso SECOM: datos públicos UCI; procesos de ejemplo simulados · '
    '<a href="https://github.com/jesusgonzalezmexdata-ux/predictive-quality-mlops" target="_blank" rel="noopener">Código fuente</a></footer>'
)



def aplicar():
    st.html(CSS)
    st.html(PIE)
