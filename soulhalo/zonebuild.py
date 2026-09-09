"""Génération des zones de boss en CALQUES séparés, méthode du dépôt guilde.

Reprend la structure éprouvée de `meromoonmeri/guilde-treehouse-pmd` :

    kit.json                    manifeste : calques, zones, ambiances
    calques/<zone>/<ambiance>/  un PNG TRANSPARENT par calque, numéroté
    zones/<zone>/               la composition à plat, par ambiance
    tuiles/<zone>/              le tileset : bloc autotile 3x3 + tuiles
    apercu_zones.html           visionneuse : calques cochables, ambiances

Trois principes repris tels quels :

1. UN PNG PAR CALQUE, transparent, aux mêmes dimensions que la scène. On
   peut donc en désactiver n'importe lequel sans rien recalculer.
2. LES AMBIANCES SONT DES VARIANTES du même jeu de calques, pas des
   scènes séparées.
3. UN MANIFESTE JSON décrit tout, pour qu'un moteur ou un éditeur (Tiled,
   Aseprite) puisse s'en servir sans deviner.

Ce qui change par rapport à la guilde : les zones sont MODULAIRES et
paramétrables — chaque calque se règle par `ZoneConfig`, et l'ensemble
dérive de la palette de six teintes déclarée dans `zone_boss/zones.json`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from . import pixelart as PX
from .tileset import (TILE, Autotile, TileParams, ZoneLayers, load_zones,
                      palette_rgb, value_noise)

OUT = "zone_boss"
SCENE_W, SCENE_H = 320, 240

# Les calques, dans l'ordre de composition. Numérotés comme chez la guilde
# pour que le tri alphabétique soit l'ordre de rendu.
CALQUES = [
    {"id": "00_ciel", "nom": "Ciel et fond lointain", "parallaxe": 0.15},
    {"id": "01_lointain", "nom": "Silhouettes de decor", "parallaxe": 0.40},
    {"id": "02_sol", "nom": "Sol jouable (tileset)", "parallaxe": 1.00},
    {"id": "03_murs", "nom": "Murs et bordures", "parallaxe": 1.00},
    {"id": "04_props", "nom": "Rochers, cristaux, elements poses", "parallaxe": 1.00},
    {"id": "05_liquide", "nom": "Lave, eau, glace au sol", "parallaxe": 1.00},
    {"id": "06_arene", "nom": "Marquage de l'arene de boss", "parallaxe": 1.00},
    {"id": "07_fx", "nom": "Particules et lueurs (anime)", "parallaxe": 1.10},
    {"id": "08_eclairage", "nom": "Eclairage d'ambiance", "parallaxe": 1.00},
    {"id": "09_bordure", "nom": "Vignette de premier plan", "parallaxe": 1.00},
]

# Ambiances : des VARIANTES du meme jeu de calques (comme jour/nuit chez la
# guilde). Chacune est un reglage colorimetrique, pas une scene refaite.
AMBIANCES = {
    "normal": {"nom": "Normale", "gain": 1.00, "teinte": None, "melange": 0.0},
    "combat": {"nom": "Combat de boss", "gain": 1.14,
               "teinte": (1.00, 0.86, 0.72), "melange": 0.18},
    "sombre": {"nom": "Sombre", "gain": 0.62,
               "teinte": (0.55, 0.62, 0.90), "melange": 0.22},
    "cataclysme": {"nom": "Cataclysme", "gain": 1.22,
                   "teinte": (1.00, 0.55, 0.30), "melange": 0.30},
}


@dataclass
class ZoneConfig:
    """Réglages modulaires d'une zone. Tout est surchargeable."""
    seed: int = 0
    horizon: float = 0.46           # hauteur de l'horizon, en fraction
    crete: float = 0.26             # amplitude des silhouettes
    props_n: int = 16               # nombre d'elements poses
    props_taille: tuple = (3, 9)
    liquide: bool = False           # lave / eau au sol
    liquide_niveau: float = 0.86
    fx_n: int = 26                  # particules
    fx_montee: float = 0.6          # sens : >0 monte, <0 descend
    arene: bool = True              # cercle d'arene marque au sol
    arene_r: float = 0.30
    crack: float = 0.0              # fissures dans le sol
    vignette: float = 0.55


# Réglages par zone : ce qui distingue une caverne de magma d'un sommet.
PROFILS = {
    "Groudon": ZoneConfig(liquide=True, crack=0.45, fx_montee=0.8, crete=0.34,
                          horizon=0.40),
    "Moltres": ZoneConfig(crack=0.30, fx_montee=0.5, crete=0.40),
    "Entei": ZoneConfig(crack=0.40, fx_montee=0.6, horizon=0.52),
    "Kyogre": ZoneConfig(liquide=True, liquide_niveau=0.55, fx_montee=0.9,
                         horizon=0.38),
    "Lugia": ZoneConfig(liquide=True, liquide_niveau=0.40, fx_montee=0.7,
                        vignette=0.85, horizon=0.30),
    "Suicune": ZoneConfig(fx_montee=-0.4, horizon=0.50, crete=0.20),
    "Articuno": ZoneConfig(fx_montee=-0.7, crack=0.22, crete=0.30),
    "Regice": ZoneConfig(fx_montee=-0.5, crack=0.28),
    "Zapdos": ZoneConfig(fx_montee=-0.9, crete=0.42, horizon=0.40),
    "Raikou": ZoneConfig(fx_montee=0.4, crete=0.18),
    "Rayquaza": ZoneConfig(crete=0.16, horizon=0.34, fx_montee=0.3),
    "Latios": ZoneConfig(crete=0.44, horizon=0.44),
    "Latias": ZoneConfig(crete=0.50, horizon=0.30, vignette=0.80),
    "Ho-Oh": ZoneConfig(crete=0.22, horizon=0.42, fx_montee=-0.3),
    "Jirachi": ZoneConfig(fx_n=40, fx_montee=-0.25, crete=0.24),
    "Mewtwo": ZoneConfig(vignette=0.90, fx_n=14, crete=0.20, horizon=0.52),
    "Deoxys": ZoneConfig(fx_n=32, crete=0.28, vignette=0.70),
    "Regirock": ZoneConfig(crack=0.35, crete=0.22),
    "Registeel": ZoneConfig(crack=0.15, crete=0.20),
}


def _rgba(h=SCENE_H, w=SCENE_W):
    return np.zeros((h, w, 4), np.float32)


def _save(arr, path):
    """Écrit un PNG RGBA. Les calques restent TRANSPARENTS."""
    a = np.clip(arr, 0.0, 1.0)
    if a.shape[2] == 3:
        a = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


class ZoneBuilder:
    """Construit les calques d'une zone, un par un, indépendamment."""

    def __init__(self, zone, cfg: ZoneConfig | None = None,
                 w=SCENE_W, h=SCENE_H):
        self.z = zone
        self.w, self.h = int(w), int(h)
        self.cfg = cfg or PROFILS.get(zone["legendaire"], ZoneConfig())
        self.pal = palette_rgb(zone)
        self.seed = self.cfg.seed or (abs(hash(zone["legendaire"])) % 9973)
        self.tiles = Autotile(zone, TileParams(crack=self.cfg.crack),
                              seed=self.seed)
        self.hz = int(self.h * self.cfg.horizon)

    # -- 00 ciel -----------------------------------------------------------
    def ciel(self, phase=0.0):
        h, w = self.h, self.w
        g = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        col = self.pal[0][None, None, :] * (1 - g) + \
            self.pal[1][None, None, :] * g
        col = np.repeat(col, w, axis=1)
        n = value_noise(h, w, 2, seed=self.seed + 3)
        # décalage ENTIER de pixels : garantit le raccord de la boucle
        n = np.roll(n, int(round(phase * w)) % w, axis=1)
        col = col * (0.86 + 0.28 * n[..., None])
        out = _rgba(h, w)
        out[..., :3] = PX.dither(np.clip(col, 0, 1), levels=12)
        out[..., 3] = 1.0
        return out

    # -- 01 lointain -------------------------------------------------------
    def lointain(self, phase=0.0):
        h, w = self.h, self.w
        out = _rgba(h, w)
        p1 = value_noise(1, w, 3, seed=self.seed + 7)[0]
        p2 = value_noise(1, w, 7, seed=self.seed + 9)[0]
        crete = (self.cfg.horizon + self.cfg.crete * (p1 - 0.5) +
                 0.10 * (p2 - 0.5)) * h
        yy = np.arange(h, dtype=np.float32)[:, None]
        m = (yy > crete[None, :]).astype(np.float32)
        m[self.hz:] = 0.0                    # s'arrête à l'horizon
        out[..., :3] = (self.pal[1] * 0.66)[None, None, :]
        out[..., 3] = m
        return out

    # -- 02 sol ------------------------------------------------------------
    def sol(self):
        """Sol pavé avec PLUSIEURS variantes de tuile.

        Une tuile unique répétée se repère immédiatement : l'œil voit le
        motif revenir tous les 24 px. Quatre variantes alternées de façon
        déterministe suffisent à casser la grille.
        """
        h, w = self.h, self.w
        out = _rgba(h, w)
        out[..., :3] = self.tiles.pave(h, w, n_var=4)
        # bord d'horizon irrégulier : une ligne parfaitement droite trahit
        # le calque, alors qu'un donjon a des contours dentelés
        bord = value_noise(1, w, 5, seed=self.seed + 41)[0]
        lim = (self.hz + (bord - 0.5) * TILE * 0.7).astype(np.int32)
        yy = np.arange(h)[:, None]
        out[..., 3] = (yy >= lim[None, :]).astype(np.float32)
        return out

    # -- 03 murs -----------------------------------------------------------
    def murs(self):
        """Paroi qui ferme l'arène au fond, avec crête irrégulière."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        haut = max(0, self.hz - TILE)
        band = self.tiles.pave(TILE * 2, w, n_var=4, seed=self.seed + 7) * 0.70
        out[haut:haut + TILE, :, :3] = band[:TILE]
        bord = value_noise(1, w, 5, seed=self.seed + 41)[0]
        lim = (self.hz + (bord - 0.5) * TILE * 0.7).astype(np.int32)
        yy = np.arange(h)[:, None]
        dedans = ((yy >= haut) & (yy < lim[None, :])).astype(np.float32)
        out[..., 3] = dedans
        # arête éclairée en haut de paroi : elle accroche la lumière
        for x in range(w):
            y = int(np.clip(haut, 0, h - 1))
            if dedans[y, x] > 0:
                out[y, x, :3] = np.clip(out[y, x, :3] * 1.6, 0, 1)
        return out

    # -- 04 props ----------------------------------------------------------
    def props(self):
        h, w = self.h, self.w
        out = _rgba(h, w)
        rng = np.random.default_rng(self.seed + 100)   # graine FIXE
        lo, hi = self.cfg.props_taille
        yy, xx = np.indices((h, w))
        for _ in range(self.cfg.props_n):
            cx = int(rng.integers(6, w - 6))
            cy = int(rng.integers(self.hz + 4, h - 4))
            r = int(rng.integers(lo, hi))
            d = np.hypot(xx - cx, (yy - cy) * 1.7)
            m = (d < r).astype(np.float32)
            if m.sum() == 0:
                continue
            col = self.pal[2] * (0.78 + 0.44 * float(rng.random()))
            sub = np.zeros((h, w, 3), np.float32) + col[None, None, :]
            sub = PX.emboss(sub, m, light=0.34, shade=0.30)
            out[..., :3] = out[..., :3] * (1 - m[..., None]) + sub * m[..., None]
            out[..., 3] = np.maximum(out[..., 3], m)
        out[..., :3] = PX.dither(out[..., :3], levels=10)
        return out

    # -- 05 liquide --------------------------------------------------------
    def liquide(self, phase=0.0):
        """Lave ou eau. Nappe qui ondule, en boucle exacte."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        if not self.cfg.liquide:
            return out
        y0 = int(h * self.cfg.liquide_niveau)
        n = value_noise(h, w, 4, seed=self.seed + 21)
        n = np.roll(n, int(round(phase * w)) % w, axis=1)
        col = self.pal[3][None, None, :] * (0.7 + 0.6 * n[..., None])
        out[y0:, :, :3] = PX.dither(np.clip(col[y0:], 0, 1), levels=8)
        out[y0:, :, 3] = 0.92
        return out

    # -- 06 arène ----------------------------------------------------------
    def arene(self):
        """Cercle qui marque le terrain du combat."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        if not self.cfg.arene:
            return out
        cy = int(self.hz + (h - self.hz) * 0.55)
        cx = w // 2
        yy, xx = np.indices((h, w)).astype(np.float32)
        d = np.hypot(xx - cx, (yy - cy) * 1.9) / (w * self.cfg.arene_r)
        anneau = ((d > 0.94) & (d < 1.0)).astype(np.float32)
        out[..., :3] = self.pal[4][None, None, :]
        out[..., 3] = anneau * 0.55
        return out

    # -- 07 fx -------------------------------------------------------------
    def fx(self, phase=0.0):
        """Particules. BOUCLE exactement : cos de période entière."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        yy, xx = np.indices((h, w)).astype(np.float32)
        add = np.zeros((h, w, 3), np.float32)
        rng = np.random.default_rng(self.seed + 200)   # graine FIXE
        n = self.cfg.fx_n
        px, py, ph = rng.random(n), rng.random(n), rng.random(n)
        col = self.pal[4]
        for i in range(n):
            t = (phase + ph[i]) % 1.0
            cx = px[i] * w
            cy = ((py[i] - t * self.cfg.fx_montee) % 1.0) * h
            tw = 0.5 + 0.5 * np.cos(2 * np.pi * t)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            add += np.exp(-d2 / 5.0)[..., None] * col[None, None, :] * \
                np.float32(tw * 0.95)
        a = np.clip(add.max(2), 0.0, 1.0)
        out[..., :3] = np.clip(add, 0, 1)
        out[..., 3] = a
        return out

    # -- 08 éclairage ------------------------------------------------------
    def eclairage(self, phase=0.0):
        """Halo doux au centre de l'arène. Respire en boucle."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        cy = int(self.hz + (h - self.hz) * 0.5)
        yy, xx = np.indices((h, w)).astype(np.float32)
        d = np.hypot(xx - w / 2, yy - cy) / (w * 0.45)
        pulse = 0.92 + 0.08 * float(np.cos(2 * np.pi * phase))
        glow = np.exp(-d * d * 2.2) * pulse
        out[..., :3] = self.pal[4][None, None, :]
        out[..., 3] = np.clip(glow * 0.30, 0, 1)
        return out

    # -- 09 bordure --------------------------------------------------------
    def bordure(self):
        """Vignette de premier plan : referme le cadre."""
        h, w = self.h, self.w
        out = _rgba(h, w)
        yy, xx = np.indices((h, w)).astype(np.float32)
        d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        v = np.clip((d - 0.72) / 0.75, 0, 1) ** 1.6
        out[..., :3] = self.pal[0][None, None, :] * 0.5
        out[..., 3] = v * self.cfg.vignette
        return out

    # -- tout --------------------------------------------------------------
    def calques(self, phase=0.0):
        """Les dix calques, dans l'ordre. Chacun est indépendant."""
        return {
            "00_ciel": self.ciel(phase),
            "01_lointain": self.lointain(phase),
            "02_sol": self.sol(),
            "03_murs": self.murs(),
            "04_props": self.props(),
            "05_liquide": self.liquide(phase),
            "06_arene": self.arene(),
            "07_fx": self.fx(phase),
            "08_eclairage": self.eclairage(phase),
            "09_bordure": self.bordure(),
        }

    def compose(self, calques=None, phase=0.0, ambiance="normal"):
        """Empile les calques. `calques` permet d'en désactiver."""
        cs = calques if calques is not None else self.calques(phase)
        out = np.zeros((self.h, self.w, 3), np.float32)
        for c in CALQUES:
            lay = cs.get(c["id"])
            if lay is None:
                continue
            a = lay[..., 3:]
            out = out * (1 - a) + lay[..., :3] * a
        return applique_ambiance(out, ambiance)


def applique_ambiance(rgb, ambiance="normal"):
    """Variante colorimétrique, comme le jour/nuit de la guilde.

    C'est un RÉGLAGE du même rendu, pas une scène refaite : les calques
    restent identiques, seule leur colorimétrie change.
    """
    a = AMBIANCES.get(ambiance, AMBIANCES["normal"])
    out = np.clip(rgb, 0, 1) * a["gain"]
    if a["teinte"] is not None and a["melange"] > 0:
        t = np.asarray(a["teinte"], np.float32)
        lum = out.mean(2, keepdims=True)
        out = out * (1 - a["melange"]) + lum * t[None, None, :] * a["melange"]
    return PX.dither(np.clip(out, 0, 1), levels=14)
