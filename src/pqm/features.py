"""Limpieza de sensores ajustada SOLO con datos de entrenamiento (sin fuga de información)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer


class SensorCleaner(BaseEstimator, TransformerMixin):
    """1) quita sensores constantes, 2) quita los muy incompletos, 3) imputa, 4) quita colineales (|r| > umbral).

    imputer: 'median' | 'knn' | 'none' (deja los vacíos para que LightGBM los trate de forma nativa).
    """

    def __init__(self, imputer: str = "knn", max_missing: float = 0.5, corr_max: float = 0.95, n_neighbors: int = 5):
        self.imputer, self.max_missing, self.corr_max, self.n_neighbors = imputer, max_missing, corr_max, n_neighbors

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        ok = (X.nunique(dropna=True) > 1) & (X.isna().mean() <= self.max_missing)
        self.kept_ = list(X.columns[ok])
        Z = X[self.kept_]
        self.imp_ = None
        if self.imputer == "median":
            self.imp_ = SimpleImputer(strategy="median").fit(Z)
        elif self.imputer == "knn":
            self.imp_ = KNNImputer(n_neighbors=self.n_neighbors).fit(Z)
        Zi = self._imputar(Z)
        corr = Zi.corr().abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0)
        faltan = Z.isna().mean().values
        eliminar: set[int] = set()
        for i in range(corr.shape[0]):
            if i in eliminar:
                continue
            for j in np.where(corr[i] > self.corr_max)[0]:
                if j > i and j not in eliminar:
                    # entre dos colineales se conserva el sensor con menos vacíos
                    eliminar.add(j if faltan[i] <= faltan[j] else i)
        self.final_ = [c for k, c in enumerate(self.kept_) if k not in eliminar]
        self.n_colineales_ = len(eliminar)
        return self

    def _imputar(self, Z: pd.DataFrame) -> pd.DataFrame:
        if self.imp_ is None:
            return Z.fillna(Z.median())  # solo para medir correlación
        return pd.DataFrame(self.imp_.transform(Z), columns=Z.columns, index=Z.index)

    def transform(self, X):
        X = pd.DataFrame(X)
        Z = X.reindex(columns=self.kept_)
        if self.imp_ is not None:
            Z = self._imputar(Z)
        return Z[self.final_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.final_)
