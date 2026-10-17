"""Entrada del dictamen de viabilidad: tabla de lotes con fecha, variables de proceso y resultado pasa / falla.

Acepta Excel (hoja «lotes» o la primera) o CSV. Toda columna numérica que no sea la fecha, el resultado o el
identificador se trata como variable de proceso. Los mensajes de validación van en español con número de fila.
"""
from __future__ import annotations

import io
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {"date": "fecha", "fecha_hora": "fecha", "timestamp": "fecha", "fecha_produccion": "fecha",
           "result": "resultado", "estatus": "resultado", "status": "resultado", "dictamen": "resultado", "falla": "resultado",
           "rechazo": "resultado", "defecto": "resultado", "lote_id": "lote", "id_lote": "lote", "batch": "lote", "id": "lote"}
FAIL = {"falla", "f", "1", "si", "rechazo", "rechazado", "ng", "no_pasa", "nok", "malo", "fail", "true", "defectuoso"}
PASS = {"pasa", "p", "0", "no", "aceptado", "ok", "bueno", "pass", "false", "conforme", "liberado"}
MIN_LOTS, MIN_FAILURES = 200, 20
MAX_MISSING = 0.5

INSTRUCTIONS = [
    ("Columna", "Obligatoria", "Qué poner"),
    ("fecha", "Sí", "Fecha (y hora si existe) en que se fabricó el lote: dd/mm/aaaa hh:mm o aaaa-mm-dd hh:mm"),
    ("resultado", "Sí", "pasa / falla (también acepta OK/NG, conforme/rechazo, 0/1)"),
    ("lote", "No", "Identificador del lote"),
    ("(variables)", "Sí, al menos 1", "Una columna numérica por variable de proceso medida ANTES de conocer el resultado: "
     "temperaturas, presiones, tiempos de ciclo, lecturas de sensores"),
    ("", "", "No incluyas variables medidas después de la inspección final: harían trampa. Mínimo 200 lotes y 20 fallas; ideal "
     "1,000 lotes. El archivo se procesa en tu navegador y no se envía a ningún servidor."),
]


class DataError(ValueError):
    """El archivo no se puede evaluar."""


@dataclass
class Issue:
    nivel: str
    fila: int | None
    mensaje: str


def _plain(text) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return "_".join(text.strip().lower().replace("-", " ").split())


def _key(text) -> str:
    key = _plain(text)
    return ALIASES.get(key, key)


def read_table(source, name: str = "") -> pd.DataFrame:
    raw = source.getvalue() if hasattr(source, "getvalue") else source
    name = name or getattr(source, "name", "")
    try:
        if isinstance(raw, (str, Path)):
            raw = Path(raw).read_bytes()
        if str(name).lower().endswith(".csv") or raw[:2] != b"PK":
            text = raw.decode("utf-8-sig")
            head = text.split("\n", 1)[0]
            sep = ";" if head.count(";") > head.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        else:
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, dtype=str)
            df = next((d for n, d in sheets.items() if _key(n) == "lotes"), next(iter(sheets.values())))
    except Exception as exc:
        raise DataError(f"No se pudo leer el archivo como Excel (.xlsx) o CSV: {exc}") from exc
    return df.rename(columns=_key).dropna(how="all")


def _dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    serial = pd.to_numeric(s, errors="coerce")
    ok = serial.between(20000, 80000) & out.isna()
    if ok.any():
        out[ok] = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial[ok], unit="D")
    return out


def validate(df: pd.DataFrame):
    """Devuelve (lotes ordenados por fecha con columnas fecha, y y variables; lista de variables; avisos)."""
    missing = [c for c in ("fecha", "resultado") if c not in df.columns]
    if missing:
        raise DataError("Faltan columnas obligatorias: " + ", ".join(missing) + ". Usa la plantilla descargable.")
    issues: list[Issue] = []
    rows = pd.Series(np.arange(len(df)) + 2, index=df.index)
    fecha = _dates(df["fecha"])
    res = df["resultado"].map(_plain)
    y = pd.Series(np.where(res.isin(FAIL), 1, np.where(res.isin(PASS), 0, -1)), index=df.index)
    bad = fecha.isna() | (y < 0)
    for r in df.index[fecha.isna()][:30]:
        issues.append(Issue("error", int(rows[r]), f"«fecha» no reconocida (valor: {df.at[r, 'fecha']!r})"))
    for r in df.index[(y < 0) & fecha.notna()][:30]:
        issues.append(Issue("error", int(rows[r]), f"«resultado» debe ser pasa o falla (valor: {df.at[r, 'resultado']!r})"))
    if bad.sum() > 30:
        issues.append(Issue("error", None, f"{int(bad.sum())} filas excluidas en total por fecha o resultado no reconocidos."))

    candidates = [c for c in df.columns if c not in ("fecha", "resultado", "lote")]
    num = df[candidates].apply(lambda s: pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False)
                                                        .replace({"": np.nan, "nan": np.nan, "NA": np.nan}), errors="coerce"))
    given = df[candidates].apply(lambda s: ~s.astype(str).str.strip().isin(["", "nan", "NA", "None"]))
    text_cols = [c for c in candidates if given[c].sum() and num[c].notna().sum() < 0.9 * given[c].sum()]
    if text_cols:
        issues.append(Issue("aviso", None, "Columnas no numéricas que no se usan: " + ", ".join(text_cols[:10])
                            + ("…" if len(text_cols) > 10 else "")))
    num = num.drop(columns=text_cols)
    keep = num.loc[~bad]
    const = [c for c in keep.columns if keep[c].nunique(dropna=True) < 2]
    sparse = [c for c in keep.columns if c not in const and keep[c].isna().mean() > MAX_MISSING]
    if const:
        issues.append(Issue("aviso", None, f"{len(const)} variables constantes se descartan."))
    if sparse:
        issues.append(Issue("aviso", None, f"{len(sparse)} variables con más de {MAX_MISSING:.0%} de vacíos se descartan."))
    variables = [c for c in keep.columns if c not in const and c not in sparse]
    if not variables:
        raise DataError("No hay variables de proceso numéricas utilizables.")

    out = pd.concat([fecha.rename("fecha"), y.rename("y"), num[variables]], axis=1).loc[~bad]
    if "lote" in df.columns:
        out.insert(0, "lote", df.loc[~bad, "lote"].astype(str))
    out = out.sort_values("fecha", kind="stable").reset_index(drop=True)
    if len(out) < MIN_LOTS:
        raise DataError(f"Se necesitan al menos {MIN_LOTS} lotes válidos; hay {len(out)}.")
    if out["y"].sum() < MIN_FAILURES:
        issues.append(Issue("aviso", None, f"Solo {int(out['y'].sum())} fallas: con menos de {MIN_FAILURES} no hay nada que evaluar."))
    return out, variables, issues


# ---------------------------------------------------------------- ejemplos
def example_data(kind: str = "señal", n: int = 1500, seed: int = 3) -> pd.DataFrame:
    """Lotes simulados de inyección de plástico.

    kind="señal": 14 variables, 4 de ellas con efecto real (temperatura de molde, presión de sostenimiento,
    humedad de la resina y tiempo de ciclo). kind="débil": 120 variables de ruido y un efecto pequeño, como SECOM."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-07-01 06:00")
    fecha = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 300, n)), unit="D")
    if kind in ("señal", "moderada"):
        d = pd.DataFrame({
            "temp_molde_c": rng.normal(62, 2.5, n), "presion_sostenimiento_bar": rng.normal(480, 25, n),
            "humedad_resina_pct": np.clip(rng.gamma(4, 0.012, n), 0, None), "tiempo_ciclo_s": rng.normal(28, 1.2, n),
            "temp_barril_z1_c": rng.normal(225, 3, n), "temp_barril_z2_c": rng.normal(232, 3, n),
            "velocidad_inyeccion_mm_s": rng.normal(85, 6, n), "contrapresion_bar": rng.normal(12, 1.5, n),
            "temp_ambiente_c": 22 + 4 * np.sin(np.arange(n) / n * 6) + rng.normal(0, 1, n),
            "cojin_mm": rng.normal(6, 0.6, n), "par_husillo_nm": rng.normal(310, 20, n),
            "lote_resina": rng.integers(1, 9, n).astype(float), "operador_turno": rng.integers(1, 4, n).astype(float),
            "presion_pico_bar": rng.normal(1150, 40, n)})
        k = 1.0 if kind == "señal" else 0.6
        logit = (-5.2 + 1.5 * (1 - k) + k * (0.9 * np.abs(d["temp_molde_c"] - 62) - 0.06 * (d["presion_sostenimiento_bar"] - 480)
                 + 45 * (d["humedad_resina_pct"] - 0.048) + 0.6 * (d["tiempo_ciclo_s"] - 28)))
        d.loc[rng.random(n) < 0.03, "humedad_resina_pct"] = np.nan
    else:
        d = pd.DataFrame(rng.normal(0, 1, (n, 120)), columns=[f"sensor_{i:03d}" for i in range(120)])
        d.loc[:, d.columns[::7]] = d.loc[:, d.columns[::7]].mask(rng.random((n, len(d.columns[::7]))) < 0.15)
        logit = -3.0 + 0.30 * d["sensor_004"].fillna(0) + 0.20 * d["sensor_051"].fillna(0)
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    out = pd.DataFrame({"lote": [f"L{25000 + i}" for i in range(n)], "fecha": fecha.strftime("%d/%m/%Y %H:%M"),
                        "resultado": np.where(y, "falla", "pasa")})
    return pd.concat([out, d.round(4)], axis=1)


EXAMPLES = {  # nombre en pantalla: (tipo, lotes, semilla) — uno por cada dictamen posible
    "Inyección de plástico · señal clara": ("señal", 1500, 3),
    "Inyección de plástico · señal moderada, 10 meses de datos": ("moderada", 1000, 5),
    "Proceso con 120 sensores y señal débil (como SECOM)": ("débil", 1500, 3),
}


def example(name: str) -> pd.DataFrame:
    kind, n, seed = EXAMPLES[name]
    return example_data(kind, n, seed)


def example_with_errors() -> pd.DataFrame:
    df = example_data().astype(str)
    df.loc[7, "fecha"] = "32/13/2025"
    df.loc[30, "resultado"] = "pendiente"
    df["comentario"] = "sin observaciones"
    return df


def template_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        example_data().to_excel(xl, sheet_name="lotes", index=False)
        pd.DataFrame(INSTRUCTIONS[1:], columns=INSTRUCTIONS[0]).to_excel(xl, sheet_name="instrucciones", index=False)
    return buf.getvalue()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
