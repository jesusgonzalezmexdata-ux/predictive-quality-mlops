"""MVP «Dictamen de viabilidad»: entrada, protocolo sin fuga, dictamen por caso y reporte."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pqm import viability as vb
from pqm import viability_intake as vi
from pqm import viability_report as rp

warnings.filterwarnings("ignore")
FAST = vb.Ajustes(remuestreos=300)


@pytest.fixture(scope="module")
def results():
    out = {}
    for name in vi.EXAMPLES:
        df, variables, issues = vi.validate(vi.example(name))
        out[name] = (df, variables, issues, vb.evaluate(df, variables, FAST))
    return out


def test_template_round_trip():
    pytest.importorskip("openpyxl")
    df, variables, _ = vi.validate(vi.read_table(vi.template_bytes(), "plantilla.xlsx"))
    assert len(df) == 1500 and "temp_molde_c" in variables and df["fecha"].is_monotonic_increasing


def test_errors_reported_by_row_and_text_columns_dropped():
    df, variables, issues = vi.validate(vi.read_table(vi.to_csv_bytes(vi.example_with_errors()), "x.csv"))
    errors = [i for i in issues if i.nivel == "error"]
    assert len(errors) == 2 and {i.fila for i in errors} == {9, 32}
    assert "comentario" not in variables and any("no numéricas" in i.mensaje for i in issues)
    assert len(df) == 1498


def test_semicolon_csv_spanish_labels_and_decimal_comma():
    lines = ["Fecha;Estatus;Temp;Presión"] + [f"{1 + i % 28:02d}/{1 + i // 28 % 12:02d}/2025 08:00;{'NG' if i % 9 == 0 else 'OK'};"
                                               f"{60 + i % 5},5;{400 + i}" for i in range(300)]
    df, variables, _ = vi.validate(vi.read_table("\n".join(lines).encode(), "export.csv"))
    assert set(variables) == {"temp", "presion"} and df["y"].sum() == 34 and df["temp"].iloc[0] % 1 == 0.5


def test_too_few_lots_or_failures():
    with pytest.raises(vi.DataError, match="200 lotes"):
        vi.validate(vi.example("Inyección de plástico · señal clara").head(150))
    df = vi.example("Inyección de plástico · señal clara")
    df["resultado"] = np.where(np.arange(len(df)) < 10, "falla", "pasa")
    d, v, issues = vi.validate(df)
    r = vb.evaluate(d, v, FAST)
    assert not r["evaluable"] and r["dictamen"] == vb.RECOLECTAR and any("no hay nada que evaluar" in i.mensaje for i in issues)


def test_each_example_gets_its_verdict(results):
    verdicts = {name: res[3]["dictamen"] for name, res in results.items()}
    assert list(verdicts.values()) == [vb.DESPLEGAR, vb.RECOLECTAR, vb.NO_DESPLEGAR]


def test_clear_signal_is_found_and_ranked(results):
    r = results["Inyección de plástico · señal clara"][3]
    assert r["pr_auc_ic"][0] > r["prevalencia_eval"] and r["ahorro_ic"][0] > 0
    top4 = set(r["importancia"]["variable"].head(4))
    assert len(top4 & {"temp_molde_c", "presion_sostenimiento_bar", "humedad_resina_pct", "tiempo_ciclo_s"}) >= 3
    assert all(h["resultado"] == "Conforme" for h in r["hallazgos"][1:4])


def test_collect_more_data_estimates_sample_size(results):
    r = results["Inyección de plástico · señal moderada, 10 meses de datos"][3]
    assert r["ahorro"] > 0 and r["ahorro_ic"][0] <= 0
    assert r["lotes_necesarios"] > r["lotes_eval"] and 0 < r["meses_necesarios"] <= r["ajustes"].meses_max
    assert r["fallas_necesarias"] > r["fallas_eval"]


def test_weak_signal_is_not_deployed(results):
    r = results["Proceso con 120 sensores y señal débil (como SECOM)"][3]
    assert r["dictamen"] == vb.NO_DESPLEGAR and r["hallazgos"][1]["resultado"] == "No conforme"


def test_threshold_and_model_never_see_the_evaluated_period(results):
    df, variables, _, r = results["Inyección de plástico · señal clara"]
    mutated = df.copy()
    cut = int(round(len(df) * 0.8))
    mutated.loc[cut:, "y"] = 1 - mutated.loc[cut:, "y"]  # cambiar el futuro no puede cambiar lo elegido con el pasado
    r2 = vb.evaluate(mutated, variables, FAST)
    assert r2["umbral"] == r["umbral"] and r2["modelo"] == r["modelo"]
    assert np.allclose(r2["puntajes"]["p"], r["puntajes"]["p"])


def test_verdict_rules():
    base = dict(prevalencia_eval=0.05, ahorro=1000.0, meses_necesarios=6.0, fallas_necesarias=40, ajustes=vb.Ajustes())
    assert vb.verdict({**base, "pr_auc_ic": (0.2, 0.4), "ahorro_ic": (100, 2000)})[0] == vb.DESPLEGAR
    assert vb.verdict({**base, "pr_auc_ic": (0.2, 0.4), "ahorro_ic": (-3000, -100)})[0] == vb.NO_DESPLEGAR
    assert vb.verdict({**base, "pr_auc_ic": (0.2, 0.4), "ahorro_ic": (-500, 2000)})[0] == vb.RECOLECTAR
    assert vb.verdict({**base, "pr_auc_ic": (0.2, 0.4), "ahorro_ic": (-500, 2000), "meses_necesarios": 60.0})[0] == vb.NO_DESPLEGAR
    assert vb.verdict({**base, "ahorro": -10.0, "pr_auc_ic": (0.2, 0.4), "ahorro_ic": (-500, 200)})[0] == vb.NO_DESPLEGAR


def test_report_has_audit_structure(results):
    df, _, issues, r = results["Inyección de plástico · señal moderada, 10 meses de datos"]
    html = rp.report_html(r, issues, "Inyección L3", (len(df), len(df)))
    for text in ("Dictamen de viabilidad", vb.RECOLECTAR, "Criterio", "Evidencia", "Resultado", "ISO 9001", "licencia MIT"):
        assert text in html
    assert "<script" not in html


def test_ui_mode_renders_example():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=300).run()
    at.button_group[0].set_value("Evalúa tu proceso").run()
    at.selectbox[0].set_value("Inyección de plástico · señal clara").run()
    assert not at.exception
    assert any("DESPLEGAR" in s.value for s in at.success)
    assert isinstance(pd.DataFrame(at.dataframe[-1].value), pd.DataFrame)


def test_runs_with_the_stub_pyarrow_of_the_browser(monkeypatch):
    """stlite trae un pyarrow incompleto; train_test_split y permutation_importance de scikit-learn fallaban con él."""
    import types

    stub = types.ModuleType("pyarrow")  # como el de stlite: tiene Table y Array, pero no RecordBatch
    stub.Table, stub.Array, stub.ChunkedArray = (type(n, (), {}) for n in ("Table", "Array", "ChunkedArray"))
    monkeypatch.setitem(__import__("sys").modules, "pyarrow", stub)
    import importlib

    importlib.reload(vb)
    df, variables, _ = vi.validate(vi.example("Inyección de plástico · señal moderada, 10 meses de datos"))
    assert vb.evaluate(df, variables, FAST)["evaluable"]
