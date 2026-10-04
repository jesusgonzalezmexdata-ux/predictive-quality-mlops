"""Captura del tablero para el README (requiere Streamlit corriendo en :8766)."""
import sys

from playwright.sync_api import sync_playwright

out, tab = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1360, "height": 1500}, device_scale_factor=2)
    pg.goto("http://localhost:8766", wait_until="networkidle")
    pg.wait_for_selector("h1", timeout=60000)
    pg.add_style_tag(content='header[data-testid="stHeader"]{display:none!important}')
    pg.get_by_role("tab", name=tab).click()
    pg.wait_for_timeout(6000)
    blk = pg.locator("[data-testid='stMainBlockContainer']").bounding_box()
    charts = pg.locator("[data-testid='stPlotlyChart'] >> visible=true")
    chart = charts.last.bounding_box() if charts.count() else {"y": blk["y"] + blk["height"] - 10, "height": 0}
    top = pg.locator("h1").first.bounding_box()["y"] - 20
    pg.screenshot(path=out, clip={"x": blk["x"], "y": top, "width": blk["width"],
                                  "height": chart["y"] + chart["height"] + 10 - top})
    b.close()
