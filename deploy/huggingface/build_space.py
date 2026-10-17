"""Builds a static Hugging Face Space that runs the Streamlit app in the browser with stlite.

Usage from the repository root: python3 deploy/huggingface/build_space.py _space

Reads deploy/huggingface/space.json, copies the listed files and writes index.html and the
Space card. No server is needed: Python runs in the visitor's browser (Pyodide).
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

STLITE = "1.11.0"
ROOT = Path(__file__).resolve().parents[2]
CONF = json.loads((Path(__file__).parent / "space.json").read_text(encoding="utf-8"))


def files_to_mount() -> list[str]:
    out = []
    for item in CONF["include"]:
        p = ROOT / item
        if p.is_dir():
            out += [str(f.relative_to(ROOT)).replace("\\", "/") for f in sorted(p.rglob("*"))
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in {".pyc", ".map"}]
        elif p.is_file():
            out.append(item)
        else:
            raise FileNotFoundError(item)
    return out


def index_html(files: list[str]) -> str:
    mount = {
        "entrypoint": CONF["entrypoint"],
        "requirements": CONF["requirements"],
        "files": {f: {"url": f"./{f}"} for f in files},
        "streamlitConfig": {"theme.base": "light", "client.toolbarMode": "viewer", **CONF.get("streamlitConfig", {})},
    }
    title = html.escape(CONF["title"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.css">
<style>
  body{{margin:0;font-family:system-ui,sans-serif}}
  #boot{{max-width:36rem;margin:18vh auto 0;padding:0 1.25rem;color:#33414e;line-height:1.6}}
  #boot h1{{font-size:1.4rem;margin:0 0 .5rem}}
</style>
</head>
<body>
<div id="root"><div id="boot"><h1>{title}</h1><p>Starting Python in your browser. The first visit takes 20 to 40 seconds while the libraries download; later visits are faster.</p></div></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.js";
mount({json.dumps(mount)}, document.getElementById("root"));
</script>
</body>
</html>
"""


def card() -> str:
    c = CONF
    return f"""---
title: {c['title']}
emoji: {c['emoji']}
colorFrom: {c['colorFrom']}
colorTo: {c['colorTo']}
sdk: static
app_file: index.html
pinned: false
license: {c['license']}
short_description: {c['short_description'][:60]}
---

# {c['title']}

{c['summary']}

The app runs entirely in your browser with [stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide); the first load downloads the Python libraries.

- Source code and evidence: [{c['repo']}](https://github.com/jesusgonzalezmexdata-ux/{c['repo']})
- Project page: [jesusgonzalezmexdata-ux.github.io/{c['repo']}](https://jesusgonzalezmexdata-ux.github.io/{c['repo']}/)

This Space is synchronised automatically from the GitHub repository; edit the code there.
"""


def main(out: str) -> None:
    dest = Path(out)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    files = files_to_mount()
    for f in files:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    (dest / "index.html").write_text(index_html(files), encoding="utf-8")
    (dest / "README.md").write_text(card(), encoding="utf-8")
    print(f"{len(files)} files, entrypoint {CONF['entrypoint']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_space")
