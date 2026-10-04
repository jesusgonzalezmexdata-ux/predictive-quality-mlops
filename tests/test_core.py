import numpy as np
import pandas as pd

from pqm.config import CostosCalidad
from pqm.data import load_secom, split_temporal
from pqm.drift import psi
from pqm.evaluation import best_threshold, expanding_window_folds, lift_at_k, monetizar, pr_auc
from pqm.features import SensorCleaner


def test_datos_secom_ordenados_y_conteos():
    X, y, t = load_secom()
    assert X.shape == (1567, 590) and int(y.sum()) == 104
    assert t.is_monotonic_increasing


def test_holdout_es_el_ultimo_bloque_cronologico():
    tr, ho = split_temporal(100, 0.2)
    assert tr.stop == ho.start == 80 and ho.stop == 100


def test_folds_expanding_no_mezclan_futuro():
    for a, b in expanding_window_folds(1000):
        assert a.max() < b.min()


def test_cleaner_ajusta_solo_con_train_y_quita_constantes_y_colineales():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    X = pd.DataFrame({"const": 1.0, "a": a, "a2": a * 2 + 1e-3 * rng.normal(size=300), "b": rng.normal(size=300),
                      "mucho_nan": np.where(rng.random(300) < 0.7, np.nan, 1.0)})
    c = SensorCleaner(imputer="median").fit(X.iloc[:200])
    Z = c.transform(X.iloc[200:])
    assert "const" not in Z and "mucho_nan" not in Z and not ({"a", "a2"} <= set(Z.columns)) and "b" in Z
    assert not Z.isna().any().any()


def test_costos_y_ahorro_consistentes():
    c = CostosCalidad()
    y = np.array([1, 0, 0, 1, 0])
    f = np.array([1, 1, 0, 0, 0])
    assert c.costo(y, f) == (250 + 500) + 250 + 5000
    m = monetizar(y, np.array([0.9, 0.9, 0.1, 0.1, 0.1]), 0.5, c)
    assert m["detectadas"] == 1 and m["falsas_alarmas"] == 1 and m["no_detectadas"] == 1
    assert m["costo_mejor_baseline"] == min(c.sin_modelo(y), c.inspeccionar_todo(y))


def test_umbral_optimo_es_perfecto_con_separacion_perfecta():
    y = np.array([0] * 90 + [1] * 10)
    p = np.where(y == 1, 0.9, 0.1)
    u, costo = best_threshold(y, p, CostosCalidad())
    assert 0.1 < u <= 0.9 and pr_auc(y, p) == 1.0 and lift_at_k(y, p) > 9


def test_psi_detecta_cambio_y_estabilidad():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(size=2000))
    assert psi(r, pd.Series(rng.normal(size=2000))) < 0.1
    assert psi(r, pd.Series(rng.normal(2, 1, size=2000))) > 0.25
