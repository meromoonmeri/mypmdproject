"""Manifeste `kit.json` + visionneuse HTML autonome, méthode guilde.

Le dépôt guilde embarque tous ses aperçus en base64 dans un seul fichier
HTML : il s'ouvre sans serveur, sans dépendance, et se transmet tel quel.
On reprend le procédé, avec les calques cochables et le sélecteur
d'ambiance.
"""
from __future__ import annotations

import base64
import io
import json
import os

from PIL import Image

from .zonecompose import (AMBIANCES, CALQUES, FX_FRAMES, OUT, SCENE_H,
                          SCENE_W)


def _b64(path):
    """Encode en WEBP sans perte si c'est plus léger, sinon en PNG."""
    im = Image.open(path).convert("RGBA")
    p = io.BytesIO()
    im.save(p, "PNG", optimize=True)
    w = io.BytesIO()
    im.save(w, "WEBP", lossless=True, quality=100)
    if w.tell() < p.tell():
        return "data:image/webp;base64," + \
            base64.b64encode(w.getvalue()).decode()
    return "data:image/png;base64," + base64.b64encode(p.getvalue()).decode()


def ecris_kit(zones_faites, racine=OUT):
    src = json.load(open(os.path.join(racine, "zones.json"), encoding="utf-8"))
    par_nom = {z["legendaire"]: z for z in src["zones"]}

    kit = {
        "projet": "Zones de boss - Pokemon Mystery Dungeon Rescue Team DX",
        "methode": ("Un PNG transparent par calque, aux dimensions de la "
                    "scene. Les ambiances sont des variantes du meme jeu de "
                    "calques. Tout est decrit ici pour Tiled / Aseprite / "
                    "PMDO."),
        "moteur": {
            "scene": [SCENE_W, SCENE_H],
            "tuile": 24,
            "agrandissement": "entier, plus proche voisin",
            "note": src.get("moteur", ""),
        },
        "calques": [
            {"id": c, "nom": n, "parallaxe": p,
             "anime": c == "07_fx",
             "frames": FX_FRAMES if c == "07_fx" else 1}
            for c, n, p in CALQUES
        ],
        "ambiances": {k: v["nom"] for k, v in AMBIANCES.items()},
        "zones": {},
    }

    for nom in zones_faites:
        z = par_nom[nom]
        dz = {
            "legendaire": nom,
            "dex": z["dex"],
            "type": z["type"],
            "donjon": z["donjon_fr"],
            "donjon_en": z["donjon_en"],
            "etage": z["etage"],
            "niveau": z["niveau"],
            "arc": z["arc"],
            "biome": z["biome"],
            "palette": z["palette"],
            "sources": sorted(
                os.path.basename(f)
                for f in os.listdir(os.path.join(racine, nom, "source"))
                if f.endswith(".png")
            ) if os.path.isdir(os.path.join(racine, nom, "source")) else [],
            "ambiances": {},
        }
        for amb in AMBIANCES:
            d = os.path.join(racine, "calques", nom, amb)
            if not os.path.isdir(d):
                continue
            dz["ambiances"][amb] = {
                "calques": sorted(f for f in os.listdir(d)
                                  if f.endswith(".png")),
                "plat": f"zones/{nom}/{amb}.png",
            }
        kit["zones"][nom] = dz

    p = os.path.join(racine, "kit.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(kit, f, indent=2, ensure_ascii=False)
    return p


HTML = """<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Zones de boss - Rescue Team DX</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#12121a;color:#e8e8f0;
 font:13px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:16px 20px;background:#1b1b26;border-bottom:1px solid #2c2c3a}
h1{margin:0;font-size:17px;letter-spacing:.4px}
.sub{color:#8f8fa6;font-size:12px;margin-top:3px}
.wrap{display:flex;gap:18px;padding:18px;align-items:flex-start;
 flex-wrap:wrap}
.panel{background:#1b1b26;border:1px solid #2c2c3a;border-radius:8px;
 padding:14px}
.stage{position:relative;image-rendering:pixelated;background:#000;
 border:1px solid #33334a;border-radius:4px;overflow:hidden}
.stage img{position:absolute;inset:0;width:100%;height:100%;
 image-rendering:pixelated}
label{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer}
label:hover{color:#fff}
input[type=checkbox]{accent-color:#e08a4a}
select,button{background:#26263a;color:#e8e8f0;border:1px solid #3a3a52;
 border-radius:5px;padding:6px 10px;font-size:13px;cursor:pointer}
button.on{background:#e08a4a;color:#171720;border-color:#e08a4a}
.row{display:flex;gap:8px;align-items:center;margin-bottom:10px;
 flex-wrap:wrap}
.par{color:#6f6f88;font-size:11px;margin-left:auto}
h2{font-size:13px;margin:0 0 8px;color:#b8b8cc;text-transform:uppercase;
 letter-spacing:.7px}
.meta{font-size:12px;color:#9a9ab2}
.meta b{color:#e8e8f0;font-weight:600}
.pal{display:flex;gap:3px;margin-top:8px}
.pal i{width:22px;height:22px;border-radius:3px;border:1px solid #00000060}
</style>
<header>
<h1>Zones de boss &mdash; Pokemon Mystery Dungeon : Rescue Team DX</h1>
<div class="sub">Calques separes, ambiances, tuile 24&times;24, scene
320&times;240. Decochez un calque : rien d'autre ne bouge.</div>
</header>
<div class="wrap">
 <div class="panel">
  <div class="row">
   <select id="zone"></select>
   <select id="amb"></select>
   <select id="zoom">
    <option value="2">x2</option><option value="3" selected>x3</option>
    <option value="4">x4</option>
   </select>
   <button id="play" class="on">Animation</button>
  </div>
  <div class="stage" id="stage"></div>
 </div>
 <div class="panel" style="min-width:290px">
  <h2>Calques</h2><div id="layers"></div>
  <h2 style="margin-top:16px">Zone</h2><div class="meta" id="meta"></div>
  <div class="pal" id="pal"></div>
 </div>
</div>
<script>
const KIT = __KIT__, IMG = __IMG__;
const st=document.getElementById('stage'), zs=document.getElementById('zone'),
      as=document.getElementById('amb'), ls=document.getElementById('layers'),
      zo=document.getElementById('zoom'), pb=document.getElementById('play');
let onoff={}, frame=0, playing=true;
KIT.calques.forEach(c=>onoff[c.id]=true);
Object.keys(KIT.zones).forEach(n=>zs.add(new Option(n,n)));
function ambs(){const z=KIT.zones[zs.value];as.innerHTML='';
 Object.keys(z.ambiances).forEach(a=>as.add(new Option(
  KIT.ambiances[a]||a,a)));}
function layers(){ls.innerHTML='';KIT.calques.forEach(c=>{
 const l=document.createElement('label');
 l.innerHTML='<input type=checkbox '+(onoff[c.id]?'checked':'')+'>'+
  '<span>'+c.id.replace(/^\\d+_/,'')+(c.anime?' &#9673;':'')+'</span>'+
  '<span class=par>'+c.parallaxe.toFixed(2)+'</span>';
 l.querySelector('input').onchange=e=>{onoff[c.id]=e.target.checked;draw();};
 ls.appendChild(l);});}
function draw(){
 const n=zs.value,a=as.value,z=KIT.zones[n],k=parseInt(zo.value);
 st.style.width=(KIT.moteur.scene[0]*k)+'px';
 st.style.height=(KIT.moteur.scene[1]*k)+'px';
 st.innerHTML='';
 KIT.calques.forEach(c=>{
  if(!onoff[c.id])return;
  let f=c.anime?(c.id+'_'+String(frame%c.frames).padStart(2,'0')):c.id;
  const src=IMG[n+'/'+a+'/'+f];
  if(!src)return;
  const im=new Image();im.src=src;st.appendChild(im);});
 document.getElementById('meta').innerHTML=
  '<b>'+z.donjon+'</b> &mdash; '+z.etage+'<br>Niveau '+z.niveau+
  ' &middot; '+z.type.join(' / ')+'<br>'+z.biome+
  '<br><span style="color:#6f6f88">'+z.sources.length+
  ' planches generees</span>';
 document.getElementById('pal').innerHTML=
  z.palette.map(c=>'<i style="background:'+c+'"></i>').join('');}
zs.onchange=()=>{ambs();draw();};as.onchange=draw;zo.onchange=draw;
pb.onclick=()=>{playing=!playing;pb.className=playing?'on':'';};
setInterval(()=>{if(playing){frame++;draw();}},130);
ambs();layers();draw();
</script></html>"""


def ecris_apercu(zones_faites, racine=OUT):
    kit = json.load(open(os.path.join(racine, "kit.json"), encoding="utf-8"))
    img = {}
    for nom in zones_faites:
        for amb, d in kit["zones"][nom]["ambiances"].items():
            for f in d["calques"]:
                p = os.path.join(racine, "calques", nom, amb, f)
                img[f"{nom}/{amb}/{f[:-4]}"] = _b64(p)
    html = HTML.replace("__KIT__", json.dumps(kit, ensure_ascii=False)) \
               .replace("__IMG__", json.dumps(img))
    p = os.path.join(racine, "apercu_zones.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return p, len(img)
