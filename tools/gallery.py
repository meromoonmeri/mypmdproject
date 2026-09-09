#!/usr/bin/env python3
"""
Petit serveur d'aperçu : galerie HTML des sprites Dynamax générés.

    .venv/bin/python tools/gallery.py [--port 3000] [--out output]

Affiche pour chaque Pokémon : la comparaison avant/après, les animations
en GIF (à la vitesse du jeu), la planche FX sur fond magenta et le rapport
de validation.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmdmax import config as C          # noqa: E402
from pmdmax import sheet as SH          # noqa: E402
from pmdmax import verify as VF         # noqa: E402

ROOT = C.REPO


def scan(out_root: str):
    mons = []
    if not os.path.isdir(out_root):
        return mons
    for name in sorted(os.listdir(out_root)):
        folder = os.path.join(out_root, name)
        if not os.path.isfile(os.path.join(folder, C.MULTI_SHEET_XML)):
            continue
        data = SH.read_anim_data(os.path.join(folder, C.MULTI_SHEET_XML))
        rep = VF.verify_folder(folder)
        anims = []
        for a in data.sorted_anims():
            if a.is_ref:
                continue
            gif = os.path.join(folder, "preview", f"{a.name}.gif")
            anims.append({
                "name": a.name,
                "index": a.index,
                "size": f"{a.frame_width}x{a.frame_height}",
                "frames": len(a.durations),
                "gif": os.path.relpath(gif, ROOT) if os.path.isfile(gif) else None,
            })
        cmp_gif = os.path.join(folder, "preview", "compare_Walk.gif")
        fx = os.path.join(folder, "fx", "Dynamax-FX-magenta-x4.png")
        contact = os.path.join(folder, "preview", "contact.png")
        mons.append({
            "name": name,
            "folder": os.path.relpath(folder, ROOT),
            "ok": rep.ok,
            "errors": rep.errors[:6],
            "warnings": rep.warnings[:4],
            "palette": len(rep.palette),
            "shadow": data.shadow_size,
            "anims": anims,
            "compare": os.path.relpath(cmp_gif, ROOT) if os.path.isfile(cmp_gif) else None,
            "fx": os.path.relpath(fx, ROOT) if os.path.isfile(fx) else None,
            "contact": os.path.relpath(contact, ROOT) if os.path.isfile(contact) else None,
        })
    return mons


PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>pmdmax — sprites Dynamax</title>
<style>
 :root{--bg:#12101a;--card:#1e1b2b;--line:#332c47;--txt:#ece9f5;--dim:#9d93b8;
       --red:#e92d60;--pink:#fc6ea0;--ok:#3ddc97;--bad:#ff5d5d}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);
   font:15px/1.55 ui-sans-serif,system-ui,'Segoe UI',sans-serif}
 header{padding:26px 30px;border-bottom:1px solid var(--line);
   background:linear-gradient(120deg,#2a0f22,#12101a 60%)}
 h1{margin:0;font-size:23px;letter-spacing:.3px}
 h1 span{color:var(--pink)}
 .sub{color:var(--dim);margin-top:6px;font-size:13.5px}
 .wrap{padding:24px 30px;max-width:1500px;margin:0 auto}
 .mon{background:var(--card);border:1px solid var(--line);border-radius:14px;
   padding:20px;margin-bottom:24px}
 .mhead{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}
 .mhead h2{margin:0;font-size:19px}
 .badge{font-size:12px;padding:3px 10px;border-radius:20px;font-weight:600}
 .ok{background:rgba(61,220,151,.14);color:var(--ok)}
 .bad{background:rgba(255,93,93,.14);color:var(--bad)}
 .meta{color:var(--dim);font-size:13px}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 .panel{background:#171426;border:1px solid var(--line);border-radius:10px;padding:12px}
 .panel h3{margin:0 0 9px;font-size:12.5px;text-transform:uppercase;
   letter-spacing:.8px;color:var(--pink)}
 img{image-rendering:pixelated;display:block;max-width:100%}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:12px}
 .cell{background:#171426;border:1px solid var(--line);border-radius:9px;
   padding:9px;text-align:center}
 .cell img{margin:0 auto;max-height:150px;width:auto}
 .cell .n{font-size:12.5px;margin-top:7px;font-weight:600}
 .cell .d{font-size:11px;color:var(--dim)}
 .fx{overflow-x:auto;background:#0d0b14;border-radius:8px;padding:8px}
 .fx img{max-width:none;height:120px}
 .err{color:var(--bad);font-size:12.5px;margin:3px 0}
 .warn{color:#ffc46b;font-size:12.5px;margin:3px 0}
 code{background:#0d0b14;padding:2px 6px;border-radius:5px;font-size:12.5px;
   color:var(--pink)}
 .empty{padding:60px;text-align:center;color:var(--dim)}
</style></head><body>
<header>
  <h1>pmdmax — sprites <span>Dynamax</span></h1>
  <div class="sub">Format strict PMDCollab/SpriteCollab · nuages tournoyants ·
  transformation multi-frames · validation SpriteBot</div>
</header>
<div class="wrap" id="app"></div>
<script>
const DATA = __DATA__;
const app = document.getElementById('app');
if (!DATA.length) {
  app.innerHTML = '<div class="empty">Aucun sprite dans <code>output/</code>.<br>' +
    'Lance <code>python -m pmdmax build 0025</code>.</div>';
}
for (const m of DATA) {
  const d = document.createElement('div');
  d.className = 'mon';
  let h = `<div class="mhead"><h2>${m.name}</h2>` +
    `<span class="badge ${m.ok?'ok':'bad'}">${m.ok?'VALIDE':'REFUSE'}</span>` +
    `<span class="meta">${m.anims.length} animations · ${m.palette} couleurs · ShadowSize ${m.shadow}</span></div>`;
  for (const e of m.errors) h += `<div class="err">✖ ${e}</div>`;
  for (const w of m.warnings) h += `<div class="warn">▲ ${w}</div>`;

  h += '<div class="row">';
  if (m.compare) h += `<div class="panel"><h3>Avant / Après</h3>
      <img src="/${m.compare}" alt="comparaison"></div>`;
  if (m.contact) h += `<div class="panel"><h3>Planche de contact</h3>
      <img src="/${m.contact}" style="max-height:260px" alt="contact"></div>`;
  h += '</div>';

  if (m.fx) h += `<div class="panel" style="margin-top:16px"><h3>FX de transformation — fond magenta</h3>
      <div class="fx"><img src="/${m.fx}" alt="fx"></div></div>`;

  const withGif = m.anims.filter(a=>a.gif);
  if (withGif.length) {
    h += '<div class="panel" style="margin-top:16px"><h3>Animations</h3><div class="grid">';
    for (const a of withGif) {
      h += `<div class="cell"><img src="/${a.gif}" alt="${a.name}">
        <div class="n">${a.name}</div>
        <div class="d">${a.size} · ${a.frames}f</div></div>`;
    }
    h += '</div></div>';
  }
  d.innerHTML = h;
  app.appendChild(d);
}
</script></body></html>
"""


class Handler(SimpleHTTPRequestHandler):
    out_root = "output"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            body = PAGE.replace("__DATA__", json.dumps(scan(self.out_root)))
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    Handler.out_root = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"Galerie sur http://0.0.0.0:{a.port}  (dossier : {Handler.out_root})")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
