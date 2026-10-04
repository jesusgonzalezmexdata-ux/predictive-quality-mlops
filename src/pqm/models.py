from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from pqm.config import SEED
from pqm.features import SensorCleaner


def build_pipeline(imputer: str = "knn", scale_pos_weight: float | None = None, **lgbm) -> Pipeline:
    """Limpieza + LightGBM. El desbalance se maneja con pesos de clase, no con sobremuestreo."""
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, random_state=SEED, n_jobs=2, verbose=-1)
    params.update(lgbm)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return Pipeline([("clean", SensorCleaner(imputer=imputer)), ("model", LGBMClassifier(**params))])


def peso_positivo(y) -> float:
    y = list(y)
    p = sum(y)
    return (len(y) - p) / max(p, 1)
