"""Rutas y supuestos de negocio. Los costos NO vienen de los datos: son supuestos editables."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_SECOM = ROOT / "data/raw/secom"
RAW_CMAPSS = ROOT / "data/raw/cmapss"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports/figures"
SEED = 42


@dataclass(frozen=True)
class CostosCalidad:
    """Matriz de costos por lote (MXN). Supuestos ilustrativos."""
    scrap_no_detectado: float = 5000.0   # defecto que pasa sin detectarse
    inspeccion: float = 250.0            # inspección adicional de un lote marcado
    retrabajo_detectado: float = 500.0   # costo de corregir un defecto detectado a tiempo

    def costo(self, y_true, flagged) -> float:
        import numpy as np
        y, f = np.asarray(y_true).astype(bool), np.asarray(flagged).astype(bool)
        tp, fp, fn = (y & f).sum(), (~y & f).sum(), (y & ~f).sum()
        return float(tp * (self.inspeccion + self.retrabajo_detectado) + fp * self.inspeccion + fn * self.scrap_no_detectado)

    def sin_modelo(self, y_true) -> float:
        return float(self.scrap_no_detectado * int(sum(y_true)))

    def inspeccionar_todo(self, y_true) -> float:
        n, p = len(y_true), int(sum(y_true))
        return float(n * self.inspeccion + p * self.retrabajo_detectado)


@dataclass(frozen=True)
class CostosMantenimiento:
    """Costos de reemplazo de motor (unidades arbitrarias). Supuestos ilustrativos."""
    correctivo: float = 100.0      # falla en servicio
    preventivo: float = 15.0       # reemplazo planeado
    vida_perdida_por_ciclo: float = 0.12  # valor de cada ciclo útil que se desecha
    horizonte: int = 30            # ciclos hasta la siguiente oportunidad de revisión
