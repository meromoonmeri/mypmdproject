"""Assemblage des zones de boss à partir des planches GÉNÉRÉES.

Division du travail, et c'est le point important :

    generate_image  ->  zone_boss/<Legendaire>/source/*.png   LA MATIÈRE
    ce module       ->  zone_boss/calques/<zone>/<amb>/*.png  LE MONTAGE

Aucune texture n'est peinte ici. Le module découpe, détoure, pave et
compose des planches produites par le générateur d'images. Il fabrique
la structure — pas les pixels.

Le format de sortie est celui du dépôt guilde : un PNG transparent par
calque, tous aux dimensions de la scène, numérotés pour que l'ordre
alphabétique soit l'ordre de rendu. N'importe quel calque se décoche
sans rien recalculer.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from . import zoneassets as ZA

OUT = "zone_boss"
SCENE_W, SCENE_H = 320, 240
TILE = 24
HORIZON = 88          # ligne de partage ciel / sol, en pixels

CALQUES = [
    ("00_ciel", "Ciel et fond lointain", 0.15),
    ("01_lointain", "Silhouettes de decor", 0.40),
    ("02_sol", "Sol jouable (tileset)", 1.00),
    ("03_murs", "Murs et bordures", 1.00),
    ("04_props", "Rochers, cristaux, elements poses", 1.00),
    ("05_liquide", "Lave, eau, glace au sol", 1.00),
    ("06_arene", "Marquage de l'arene de boss", 1.00),
    ("07_fx", "Particules et lueurs (anime)", 1.10),
    ("08_eclairage", "Eclairage d'ambiance", 1.00),
    ("09_bordure", "Vignette de premier plan", 1.00),
]

AMBIANCES = {
    "normal": {"nom": "Normale", "gain": 1.00, "teinte": None, "melange": 0.0},
    "combat": {"nom": "Combat de boss", "gain": 1.14,
               "teinte": (1.00, 0.86, 0.72), "melange": 0.18},
    "sombre": {"nom": "Approche silencieuse", "gain": 0.62,
               "teinte": (0.55, 0.62, 0.90), "melange": 0.22},
    "cataclysme": {"nom": "Cataclysme", "gain": 1.22,
                   "teinte": (1.00, 0.55, 0.30), "melange": 0.30},
}

FX_FRAMES = 8         # longueur de la boucle de particules


# --------------------------------------------------------------------------
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def vide():
    return np.zeros((SCENE_H, SCENE_W, 4), np.float32)


def _resize(rgba, w, h):
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8),
                         "RGBA")
    return np.asarray(im.resize((max(1, int(w)), max(1, int(h))),
                                Image.NEAREST), np.float32) / 255.0


def _src(zone_dir, nom):
    p = os.path.join(zone_dir, "source", nom + ".png")
    return p if os.path.exists(p) else None


# --------------------------------------------------------------------------
# les calques
# --------------------------------------------------------------------------
def calque_ciel(pal):
    """Dégradé tramé entre les deux teintes sombres de la palette.

    Seul calque sans source générée : c'est un aplat de fond, pas une
    matière. Le tramage ordonné évite les bandes franches d'un dégradé
    sur palette réduite.
    """
    out = vide()
    haut, bas = np.array(pal[0]), np.array(pal[1])
    t = np.linspace(0, 1, SCENE_H, dtype=np.float32)[:, None]
    col = haut[None, :] * (1 - t) + bas[None, :] * t
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                      [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - .5
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W]
    tram = bayer[yy % 4, xx % 4] * (1.0 / 14.0)
    out[..., :3] = np.clip(col[:, None, :] + tram[..., None], 0, 1)
    out[..., 3] = 1.0
    return out


def calque_lointain(zone_dir):
    p = _src(zone_dir, "01_lointain")
    if not p:
        return vide()
    a = ZA.prepare(p, mode="key", bloc=4, couleurs=12, seuil=0.34, ronge=1)
    # on ne garde que la bande utile : les lignes qui contiennent du sujet
    lignes = np.where(a[..., 3].max(1) > 0.5)[0]
    if len(lignes) == 0:
        return vide()
    a = a[lignes[0]:lignes[-1] + 1]
    h = int(round(a.shape[0] * (SCENE_W / a.shape[1])))
    h = max(24, min(h, 120))
    a = _resize(a, SCENE_W, h)
    out = vide()
    ZA.blit(out, a, 0, HORIZON + 6 - h)
    return out


def sol_texture(zone_dir):
    """Prépare la texture de sol et la ramène à un multiple de la tuile."""
    p = _src(zone_dir, "02_sol_tileset")
    if not p:
        return None
    a = ZA.prepare(p, mode="opaque", bloc=4, couleurs=16)
    n = 10 * TILE                      # 240x240 = 10x10 tuiles distinctes
    return _resize(a, n, n)


def calque_sol(tex):
    if tex is None:
        return vide()
    out = vide()
    h = SCENE_H - HORIZON
    ZA.blit(out, ZA.tile_fill(tex, SCENE_W, h), 0, HORIZON)
    return out


def calque_murs(zone_dir, pal):
    """Bande de murs le long de l'horizon.

    Si une planche `03_murs` a été générée on l'utilise, sinon on pose un
    bandeau tiré de la palette : le sol ne doit jamais toucher le ciel
    directement, c'est ce qui créait les bandes noires à l'horizon.
    """
    out = vide()
    p = _src(zone_dir, "03_murs")
    if p:
        a = ZA.prepare(p, mode="key", bloc=4, couleurs=14, seuil=0.34)
        blocs = ZA.extraire(a, ZA.decoupe(a, min_px=60))
        if blocs:
            x = 0
            i = 0
            while x < SCENE_W:
                b = blocs[i % len(blocs)]
                bh = 30
                bw = max(8, int(round(b.shape[1] * bh / b.shape[0])))
                ZA.blit(out, _resize(b, bw, bh), x, HORIZON - bh + 8)
                x += bw
                i += 1
            return out
    bande = np.zeros((14, SCENE_W, 4), np.float32)
    bande[..., :3] = np.array(pal[0]) * 0.7
    bande[:2, :, :3] = np.array(pal[2]) * 0.8
    bande[..., 3] = 1.0
    ZA.blit(out, bande, 0, HORIZON - 6)
    return out


def calque_props(zone_dir, seed):
    """Sème les objets découpés sur le sol, triés par profondeur."""
    p = _src(zone_dir, "04_props")
    if not p:
        return vide()
    a = ZA.prepare(p, mode="key", bloc=4, couleurs=16, seuil=0.34, ronge=1)
    objets = ZA.extraire(a, ZA.decoupe(a, min_px=40))
    if not objets:
        return vide()
    rng = np.random.default_rng(seed)
    out = vide()
    poses = []
    for i in range(11):
        o = objets[rng.integers(0, len(objets))]
        y = int(HORIZON + 10 + rng.random() ** 1.4 * (SCENE_H - HORIZON - 40))
        # plus l'objet est bas, plus il est proche : il grandit
        prof = (y - HORIZON) / max(1, SCENE_H - HORIZON)
        ht = int(round(TILE * (0.9 + 1.5 * prof)))
        wd = max(6, int(round(o.shape[1] * ht / o.shape[0])))
        x = int(rng.integers(-4, SCENE_W - wd + 4))
        poses.append((y, x, _resize(o, wd, ht)))
    for y, x, o in sorted(poses, key=lambda t: t[0]):
        ZA.blit(out, o, x, y - o.shape[0])
    return out


def calque_arene(pal):
    """Anneau au sol qui délimite l'arène, en projection écrasée."""
    out = vide()
    cx, cy = SCENE_W // 2, HORIZON + (SCENE_H - HORIZON) * 3 // 5
    rx, ry = 104.0, 40.0
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    anneau = (np.abs(d - 1.0) < 0.055) & (yy > HORIZON)
    tirets = ((xx.astype(int) + yy.astype(int) * 2) % 10) < 6
    m = anneau & tirets
    out[..., :3] = np.array(pal[3])
    out[..., 3] = m.astype(np.float32) * 0.75
    return out


def calque_fx(zone_dir, seed, phase):
    """Particules additives, boucle exacte.

    Les positions dérivent d'un modulo sur la hauteur : à `phase` = 1 la
    couche est identique à `phase` = 0, au pixel près.
    """
    p = _src(zone_dir, "07_fx")
    if not p:
        return vide()
    a = ZA.prepare(p, mode="black", bloc=4, couleurs=16)
    sprites = ZA.extraire(a, ZA.decoupe(a, min_px=30))
    if not sprites:
        return vide()
    rng = np.random.default_rng(seed + 777)
    out = vide()
    for i in range(14):
        s = sprites[rng.integers(0, len(sprites))]
        t = int(rng.integers(8, 22))
        s = _resize(s, t, max(1, int(round(s.shape[0] * t / s.shape[1]))))
        x = int(rng.integers(0, SCENE_W))
        y0 = float(rng.random())
        monte = int(rng.integers(1, 3))       # tours entiers par boucle
        span = SCENE_H - HORIZON + 40
        y = SCENE_H - int(((y0 + monte * phase) % 1.0) * span)
        ZA.blit(out, s, x - t // 2, y)
    return out


def calque_eclairage(pal):
    """Halo chaud centré sur l'arène, en mode additif doux."""
    out = vide()
    cx, cy = SCENE_W // 2, HORIZON + (SCENE_H - HORIZON) * 3 // 5
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    d = np.sqrt(((xx - cx) / 150.0) ** 2 + ((yy - cy) / 110.0) ** 2)
    g = np.clip(1.0 - d, 0, 1) ** 2
    g = np.round(g * 5) / 5.0                 # paliers : pas de dégradé lisse
    out[..., :3] = np.array(pal[4])
    out[..., 3] = g * 0.30
    return out


def calque_bordure():
    out = vide()
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    dx = np.abs(xx - SCENE_W / 2) / (SCENE_W / 2)
    dy = np.abs(yy - SCENE_H / 2) / (SCENE_H / 2)
    v = np.clip(np.maximum(dx, dy) * 1.25 - 0.55, 0, 1)
    out[..., 3] = (np.round(v * 4) / 4.0) * 0.55
    return out


# --------------------------------------------------------------------------
def applique_ambiance(rgba, amb):
    """Une ambiance est un réglage colorimétrique, pas une scène refaite."""
    a = AMBIANCES[amb]
    out = rgba.copy()
    c = out[..., :3] * a["gain"]
    if a["teinte"] is not None:
        t = np.array(a["teinte"], np.float32)
        lum = c.mean(2, keepdims=True)
        c = c * (1 - a["melange"]) + (lum * t[None, None, :]) * a["melange"]
    out[..., :3] = np.clip(c, 0, 1)
    return out


def construis_zone(zone, racine=OUT, ambiances=("normal", "combat")):
    """Fabrique tous les calques d'une zone, pour chaque ambiance."""
    nom = zone["legendaire"]
    zone_dir = os.path.join(racine, nom)
    pal = [hex_rgb(h) for h in zone["palette"]]
    seed = abs(hash(nom)) % 100000

    tex = sol_texture(zone_dir)
    base = {
        "00_ciel": calque_ciel(pal),
        "01_lointain": calque_lointain(zone_dir),
        "02_sol": calque_sol(tex),
        "03_murs": calque_murs(zone_dir, pal),
        "04_props": calque_props(zone_dir, seed),
        "05_liquide": vide(),
        "06_arene": calque_arene(pal),
        "08_eclairage": calque_eclairage(pal),
        "09_bordure": calque_bordure(),
    }

    ecrits = []
    for amb in ambiances:
        d = os.path.join(racine, "calques", nom, amb)
        for cid, _, _ in CALQUES:
            if cid == "07_fx":
                for f in range(FX_FRAMES):
                    im = applique_ambiance(
                        calque_fx(zone_dir, seed, f / FX_FRAMES), amb)
                    p = os.path.join(d, f"07_fx_{f:02d}.png")
                    ZA.save_rgba(im, p)
                    ecrits.append(p)
                continue
            im = applique_ambiance(base[cid], amb)
            p = os.path.join(d, cid + ".png")
            ZA.save_rgba(im, p)
            ecrits.append(p)

        plat = vide()
        for cid, _, _ in CALQUES:
            src = base[cid] if cid != "07_fx" else calque_fx(zone_dir, seed, 0)
            ZA.blit(plat, applique_ambiance(src, amb), 0, 0)
        p = os.path.join(racine, "zones", nom, f"{amb}.png")
        ZA.save_rgba(plat, p)
        ecrits.append(p)

    if tex is not None:
        p = os.path.join(racine, "tuiles", nom, "tileset.png")
        ZA.save_rgba(tex, p)
        ecrits.append(p)

    return ecrits
