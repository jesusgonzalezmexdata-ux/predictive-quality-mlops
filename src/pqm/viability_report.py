"""Dictamen de una página en lenguaje de auditoría: criterio, evidencia, resultado y riesgo por hallazgo."""
from __future__ import annotations

import datetime as dt
import html

import pandas as pd

from pqm.viability import DESPLEGAR, NO_DESPLEGAR

SIGUIENTES = {
    DESPLEGAR: ["Operar 4 semanas en modo sombra: el modelo marca lotes y nadie actúa; comparar contra la inspección real.",
                "Después, retener solo los lotes marcados en una línea y dejar otra como control durante 8 semanas.",
                "Monitorear deriva de variables y tasa de falla mensual; reentrenar si más del 20 % de las variables deriva."],
    NO_DESPLEGAR: ["No invertir en el despliegue con estos datos y costos.",
                   "Revisar si faltan variables del proceso medidas antes del resultado (material, molde, mantenimiento).",
                   "Mejorar el registro del resultado por lote: sin etiquetas confiables ningún modelo se puede demostrar."],
}
SIGUIENTE_RECOLECTAR = ["Seguir registrando variables y resultado por lote sin cambiar el proceso de inspección.",
                        "Repetir este dictamen cuando se alcance el número de fallas indicado.",
                        "Mientras tanto, usar el modelo solo como apoyo de análisis, no para retener ni liberar lotes."]
COLOR = {DESPLEGAR: ("#1F7A3D", "#FFFFFF"), NO_DESPLEGAR: ("#D7261E", "#FFFFFF")}


def _e(x) -> str:
    return html.escape(str(x))


def _table(df: pd.DataFrame, cls: str = "") -> str:
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f"<table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def report_html(r, issues=None, proceso: str = "Proceso", data_rows=None) -> str:
    aj = r["ajustes"]
    bg, fg = COLOR.get(r["dictamen"], ("#FFD21F", "#0E0F11"))
    parts = [f"<h1>Dictamen de viabilidad de un modelo de calidad · {_e(proceso)}</h1>",
             f"<p class='meta'>Datos del {r['desde']:%d/%m/%Y} al {r['hasta']:%d/%m/%Y} · emitido el {dt.date.today():%d/%m/%Y} · "
             "Elaboró: Jesús R. L. González</p>",
             f"<div class='verdict' style='background:{bg};color:{fg}'><span>Dictamen</span><b>{_e(r['dictamen'])}</b>"
             f"<p>{_e(r['motivo'])}</p></div>"]
    err = sum(i.nivel == "error" for i in (issues or []))
    leidos = f"{data_rows[1]:,} de {data_rows[0]:,} lotes válidos ({err} excluidos por error) · " if data_rows else ""
    parts.append(f"<h2>1 · Alcance</h2><p>{leidos}{r['lotes']:,} lotes, {r['fallas']} fallas, {r['variables']} variables de proceso. "
                 f"Entrenamiento hasta el {r['corte']:%d/%m/%Y}; evaluación única en los {r['lotes_eval']} lotes posteriores "
                 f"({r['fallas_eval']} fallas). Costos supuestos por lote: scrap no detectado {aj.scrap_no_detectado:,.0f} MXN, inspección "
                 f"{aj.inspeccion:,.0f} MXN, retrabajo de una falla detectada {aj.retrabajo_detectado:,.0f} MXN. Los datos se procesaron en "
                 "el equipo del usuario.</p>")
    if r["hallazgos"]:
        h = pd.DataFrame(r["hallazgos"])
        parts.append("<h2>2 · Hallazgos</h2>" + _table(pd.DataFrame({
            "Hallazgo": h["titulo"], "Criterio": h["criterio"], "Evidencia": h["evidencia"], "Resultado": h["resultado"],
            "Riesgo": h["riesgo"].replace("", "—")}), "findings"))
    if r.get("evaluable"):
        m = r["mon"]
        parts.append("<h2>3 · Comparación de estrategias en el periodo evaluado</h2>" + _table(pd.DataFrame({
            "Estrategia": ["Sin modelo (no inspeccionar)", "Inspeccionar todo", f"Con modelo ({r['modelo']})"],
            "Costo MXN": [f"{m['costo_sin_modelo']:,.0f}", f"{m['costo_inspeccionar_todo']:,.0f}", f"{m['costo_con_modelo']:,.0f}"]})))
        imp = r["importancia"][r["importancia"]["importancia"] > 0]
        parts.append("<h2>4 · Variables que más pesan</h2>" + _table(pd.DataFrame({
            "Variable": imp["variable"], "Pérdida de PR-AUC si se revuelve": imp["importancia"].map("{:.3f}".format),
            "Deriva (PSI)": imp["psi"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")}))
            + "<p class='note'>Importancia por permutación en el periodo evaluado. Es una pista para ingeniería, no una causa raíz.</p>")
    steps = SIGUIENTES.get(r["dictamen"], SIGUIENTE_RECOLECTAR)
    parts.append("<h2>5 · Condiciones y siguientes pasos</h2><ol>" + "".join(f"<li>{_e(s)}</li>" for s in steps) + "</ol>")
    if r.get("riesgos"):
        parts.append("<h2>6 · Riesgos y supuestos</h2><ul>" + "".join(f"<li>{_e(s)}</li>" for s in r["riesgos"]) + "</ul>")
    parts.append("<p class='note'>Referencias: ISO 9001:2015 · 6.1 (acciones para abordar riesgos), 7.1.5 (recursos de seguimiento y "
                 "medición), 8.7 (control de salidas no conformes), 9.1.3 (análisis y evaluación). Protocolo: partición cronológica, "
                 "ventanas crecientes, umbral por costo elegido sin ver el periodo evaluado, intervalos bootstrap.</p>")
    css = """body{font-family:Barlow,system-ui,-apple-system,'Segoe UI',sans-serif;color:#0E0F11;max-width:960px;margin:24px auto;
padding:0 16px;font-size:14px;line-height:1.5}h1{font:700 26px/1.1 'Barlow Condensed','Arial Narrow',sans-serif;margin-bottom:4px}
h2{font:700 18px/1.2 'Barlow Condensed','Arial Narrow',sans-serif;margin:22px 0 6px;border-bottom:3px solid #0E0F11;padding-bottom:2px}
.meta{color:#4A4E54;margin-top:0}.verdict{padding:12px 16px;border-radius:4px;margin:12px 0}.verdict span{display:block;font-size:12px;
text-transform:uppercase;letter-spacing:.06em;opacity:.85}.verdict b{font:700 30px/1.1 'Barlow Condensed','Arial Narrow',sans-serif}
.verdict p{margin:6px 0 0}table{border-collapse:collapse;margin:6px 0;font-size:13px;width:100%}th,td{border:1px solid #C9CDD1;
padding:4px 8px;text-align:left;vertical-align:top}th{background:#F6F7F5}.findings td:nth-child(4){font-weight:600;white-space:nowrap}
.note{color:#4A4E54;font-size:12.5px}@media print{body{margin:0}h2{break-after:avoid}}"""
    return ("<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Dictamen de viabilidad · "
            f"{_e(proceso)}</title><style>{css}</style></head><body>" + "".join(parts)
            + "<p class='note'>© 2026 Jesús R. L. González · Código bajo licencia MIT</p></body></html>")


def main(argv=None) -> None:
    """python -m pqm.viability_report lotes.csv [dictamen.html] [proceso]: el dictamen sin abrir el tablero."""
    import sys

    from pqm import viability as vb, viability_intake as vi

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("Uso: python -m pqm.viability_report lotes.csv [dictamen.html] [nombre del proceso]")
    raw = vi.read_table(args[0], args[0])
    df, variables, issues = vi.validate(raw)
    r = vb.evaluate(df, variables)
    out = args[1] if len(args) > 1 else "dictamen_viabilidad.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report_html(r, issues, args[2] if len(args) > 2 else "Proceso", (len(raw), len(df))))
    print(f"{r['dictamen']}: {r['motivo']}\nDictamen escrito en {out}")


if __name__ == "__main__":
    main()
