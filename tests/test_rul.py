import numpy as np
import pandas as pd

from pqm.config import CostosMantenimiento
from pqm.rul.cmapss import Normalizador, cargar, costo_politica, rul_train, score_nasa, ventanas


def test_score_nasa_penaliza_mas_sobreestimar():
    assert score_nasa([50], [50]) == 0
    assert score_nasa([50], [70]) > score_nasa([50], [30])


def test_rul_train_con_tope():
    tr, _, _ = cargar("FD001")
    r = rul_train(tr)
    assert r.max() == 125 and r.min() == 0


def test_politica_costos():
    c = CostosMantenimiento()
    # plan antes de la falla: preventivo + vida desperdiciada; plan después de la falla: correctivo
    assert np.isclose(costo_politica([40], [50], 20, c)[0], c.preventivo + c.vida_perdida_por_ciclo * 10)
    assert costo_politica([10], [50], 20, c)[0] == c.correctivo


def test_ventanas_no_usan_el_futuro():
    df = pd.DataFrame({"motor": [1] * 30, "ciclo": range(1, 31)})
    Z = pd.DataFrame({"s2": np.arange(30, dtype=float)})
    a = ventanas(df, Z)
    Z2 = Z.copy()
    Z2.loc[20:, "s2"] = 999.0  # alterar solo el futuro
    b = ventanas(df, Z2)
    assert np.allclose(a.loc[:19, "s2_pend"], b.loc[:19, "s2_pend"]) and np.allclose(a.loc[:19, "s2_mm"], b.loc[:19, "s2_mm"])
    assert np.isclose(a["s2_pend"].iloc[-1], 1.0)


def test_normalizador_por_regimen_centra_cada_regimen():
    tr, _, _ = cargar("FD002")
    n = Normalizador(6).fit(tr)
    Z = n.transform(tr)
    assert abs(Z.mean().mean()) < 0.05 and len(n.sensores_) >= 10
