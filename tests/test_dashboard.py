from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")


def test_dashboard_arranca_y_muestra_accion_con_referencia_iso():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception
    texto = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.subheader)
    estados = " ".join(str(x.delta) for x in at.metric)
    assert "ISO 9001" in texto and any(s in estados for s in ("ALTO RIESGO", "VIGILAR", "NORMAL"))
