"""Tilesets de zones de boss : 24x24, autotile 3x3, couches séparées.

Un tileset de donjon n'est pas une illustration : c'est une grille de
tuiles qui doivent se raccorder dans TOUS les sens. Le format retenu est
celui de PMDO / RogueEssence :

    tuile      24x24 pixels
    autotile   bloc 3x3 (coin HG, bord H, coin HD / bord G, plein, bord D /
               coin BG, bord B, coin BD) — de quoi habiller n'importe
               quelle forme de salle
    écran      320x240 logique, agrandissement ENTIER

Chaque zone sort en COUCHES séparées, pour la parallaxe :

    0 ciel      ne défile presque pas
    1 lointain  silhouettes
    2 sol       le tileset jouable
    3 props     posés sur le sol
    4 fx        additive : lueur, particules

La règle du projet s'applique ici aussi : tout ce qui bouge BOUCLE. Les
couches animées (fx) sont rendues sur un cycle entier, `render(0) ==
render(1)`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import field as F
from . import pixelart as PX

TAU = F.TAU
TILE = 24                       # taille de tuile PMDO
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def load_zones(path="zone_boss/zones.json"):
    with open(_p(path), encoding="utf-8") as f:
        return json.load(f)


def hex_rgb(h):
    """« #1b2f5e » -> (0.106, 0.184, 0.369)."""
    h = str(h).lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def palette_rgb(zone):
    return np.array([hex_rgb(c) for c in zone["palette"]], np.float32)


# --------------------------------------------------------------------------
# bruit cohérent, périodique — indispensable pour que les tuiles se raccordent
# --------------------------------------------------------------------------
def _hash_grid(p, seed):
    """Table (p, p) de valeurs pseudo-aléatoires stables, vectorisée.

    Pas de RNG appelé par pixel : on veut la même valeur à chaque appel
    pour une même cellule, et on la veut vite.
    """
    iy, ix = np.indices((p, p)).astype(np.int64)
    n = (ix * 374761393 + iy * 668265263 + int(seed) * 2654435761) % (2 ** 31)
    n = (n ^ (n >> 13)) * 1274126177 % (2 ** 31)
    return (((n ^ (n >> 16)) % 100003) / 100003.0).astype(np.float32)


def value_noise(h, w, period, seed=0, octaves=3):
    """Bruit de valeur PÉRIODIQUE sur `period` cellules.

    La périodicité est le point clé d'un tileset : un bruit quelconque
    laisse une couture visible tous les 24 pixels. Ici la grille de
    valeurs se referme sur elle-même (indices pris modulo `p`) ET
    l'échantillonnage tombe pile sur la période, donc le bord droit d'une
    tuile prolonge exactement son bord gauche.
    """
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        p = max(1, int(period * (2 ** o)))
        g = _hash_grid(p, seed + o * 17)
        # échantillonnage périodique : x va de 0 à p SANS atteindre p
        gx = np.arange(w, dtype=np.float32) * (p / w)
        gy = np.arange(h, dtype=np.float32) * (p / h)
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx = (gx - x0)[None, :]
        fy = (gy - y0)[:, None]
        # lissage cubique : le bruit linéaire laisse des arêtes visibles
        sx = fx * fx * (3 - 2 * fx)
        sy = fy * fy * (3 - 2 * fy)
        x1 = (x0 + 1) % p
        y1 = (y0 + 1) % p
        x0 = x0 % p
        y0 = y0 % p
        c00 = g[np.ix_(y0, x0)]
        c10 = g[np.ix_(y0, x1)]
        c01 = g[np.ix_(y1, x0)]
        c11 = g[np.ix_(y1, x1)]
        top = c00 * (1 - sx) + c10 * sx
        bot = c01 * (1 - sx) + c11 * sx
        acc += (top * (1 - sy) + bot * sy) * amp
        tot += amp
        amp *= 0.5
    return (acc / max(tot, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------
# autotile
# --------------------------------------------------------------------------
@dataclass
class TileParams:
    tile: int = TILE
    grain: float = 0.18          # force du grain de matière
    edge_light: float = 0.30     # arête éclairée du bloc
    edge_shade: float = 0.26
    dither_levels: int = 10      # palette bornée : c'est du pixel art
    crack: float = 0.0           # fissures (roche, glace)


class Autotile:
    """Bloc 3x3 de tuiles qui se raccordent dans tous les sens.

    Disposition, comme dans les tilesets de PMDO :

        HG  H   HD
        G   .   D
        BG  B   BD

    Le centre est le remplissage plein ; les huit autres portent les
    bordures. Un moteur choisit la bonne tuile selon les voisins.
    """

    def __init__(self, zone, params: TileParams | None = None, seed=0):
        self.zone = zone
        self.p = params or TileParams()
        self.pal = palette_rgb(zone)
        self.seed = int(seed)

    # -- matière -----------------------------------------------------------
    def _matiere(self, h, w, idx_lo=1, idx_hi=3, variante=0):
        """Champ de matière : un mélange entre deux teintes de la palette.

        On ne prend PAS de couleurs arbitraires : les six teintes du JSON
        sont la palette de la zone, et tout en dérive.
        """
        v = int(variante) * 101
        n = value_noise(h, w, 3, seed=self.seed + v)
        lo = self.pal[idx_lo]
        hi = self.pal[idx_hi]
        col = lo[None, None, :] * (1 - n[..., None]) + hi[None, None, :] * n[..., None]
        grain = value_noise(h, w, 8, seed=self.seed + 5 + v)
        col = col * (1.0 + self.p.grain * (grain[..., None] - 0.5))
        return np.clip(col, 0.0, 1.0)

    def fill(self, variante=0):
        """Tuile de remplissage : le sol plein.

        `variante` donne une tuile DIFFÉRENTE mais raccordable. Un tileset
        à tuile unique se trahit immédiatement : l'œil repère le motif
        répété tous les 24 px. Les vrais tilesets posent plusieurs
        variantes du même sol et les alternent.
        """
        t = self.p.tile
        col = self._matiere(t, t, variante=variante)
        if self.p.crack > 0:
            cr = value_noise(t, t, 4, seed=self.seed + 11 + variante * 31)
            veine = (np.abs(cr - 0.5) < 0.030).astype(np.float32)
            col = col * (1.0 - veine[..., None] * self.p.crack)
        return PX.dither(col, levels=self.p.dither_levels)

    def variants(self, n=4):
        """`n` tuiles de sol interchangeables."""
        return [self.fill(v) for v in range(n)]

    def pave(self, h, w, n_var=4, seed=None):
        """Pave (h, w) en alternant les variantes de façon déterministe.

        Le choix de variante vient d'un hachage de la position : la même
        carte redonne toujours le même pavage, mais l'œil n'y voit plus de
        répétition régulière.
        """
        t = self.p.tile
        vs = self.variants(n_var)
        ny, nx = h // t + 2, w // t + 2
        s = self.seed if seed is None else int(seed)
        out = np.zeros((ny * t, nx * t, 3), np.float32)
        for gy in range(ny):
            for gx in range(nx):
                k = int((gx * 73856093) ^ (gy * 19349663) ^ (s * 83492791))
                out[gy * t:(gy + 1) * t, gx * t:(gx + 1) * t] = vs[k % n_var]
        return out[:h, :w]

    def block(self):
        """Le bloc 3x3 complet, prêt à découper."""
        t = self.p.tile
        out = np.zeros((t * 3, t * 3, 3), np.float32)
        base = self._matiere(t * 3, t * 3)
        for ry in range(3):
            for rx in range(3):
                sub = base[ry * t:(ry + 1) * t, rx * t:(rx + 1) * t].copy()
                # masque du plein : les bords sont ronges du cote exterieur
                m = np.ones((t, t), np.float32)
                if ry == 0:
                    m[:2] = 0.0
                if ry == 2:
                    m[-2:] = 0.0
                if rx == 0:
                    m[:, :2] = 0.0
                if rx == 2:
                    m[:, -2:] = 0.0
                sub = PX.emboss(sub, m, light=self.p.edge_light,
                                shade=self.p.edge_shade)
                sub = sub * m[..., None] + \
                    self.pal[0][None, None, :] * (1.0 - m[..., None])
                out[ry * t:(ry + 1) * t, rx * t:(rx + 1) * t] = sub
        return PX.dither(out, levels=self.p.dither_levels)

    def raccorde(self):
        """Vérifie que la tuile de remplissage boucle dans les deux sens.

        C'est LA propriété qui rend un tileset utilisable : sans elle, une
        couture apparaît tous les 24 pixels.
        """
        f = self.fill()
        dv = float(np.abs(f[0] - f[-1]).mean())
        dh = float(np.abs(f[:, 0] - f[:, -1]).mean())
        return dv, dh


# --------------------------------------------------------------------------
# couches de zone
# --------------------------------------------------------------------------
class ZoneLayers:
    """Les cinq couches d'une zone, rendues séparément pour la parallaxe."""

    def __init__(self, zone, w=320, h=240, seed=0):
        self.z = zone
        self.w, self.h = int(w), int(h)
        self.pal = palette_rgb(zone)
        self.seed = int(seed)
        self.tiles = Autotile(zone, seed=seed)

    # -- 0 : ciel ----------------------------------------------------------
    def sky(self, phase=0.0):
        """Dégradé de fond + nappe lente. Boucle sur un tour entier."""
        h, w = self.h, self.w
        g = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        col = self.pal[0][None, None, :] * (1 - g) + \
            self.pal[1][None, None, :] * g
        col = np.repeat(col, w, axis=1)
        # nappe qui derive : le decalage est un ENTIER de pixels par boucle,
        # sinon le raccord ne serait pas exact
        n = value_noise(h, w, 2, seed=self.seed + 3)
        shift = int(round(phase * w)) % w
        n = np.roll(n, shift, axis=1)
        col = col * (0.85 + 0.30 * n[..., None])
        return PX.dither(np.clip(col, 0, 1), levels=12)

    # -- 1 : lointain ------------------------------------------------------
    def far(self, phase=0.0):
        """Silhouettes de décor, en aplat sombre — la profondeur vient du
        contraste, pas du détail."""
        h, w = self.h, self.w
        out = np.zeros((h, w, 4), np.float32)
        prof = value_noise(1, w, 3, seed=self.seed + 7)[0]
        prof2 = value_noise(1, w, 6, seed=self.seed + 9)[0]
        crete = (0.42 + 0.26 * prof + 0.12 * prof2) * h
        yy = np.arange(h, dtype=np.float32)[:, None]
        m = (yy > crete[None, :]).astype(np.float32)
        col = self.pal[1] * 0.72
        out[..., :3] = col[None, None, :]
        out[..., 3] = m
        return out

    # -- 2 : sol -----------------------------------------------------------
    def ground(self):
        """La couche jouable, pavée avec la tuile de remplissage."""
        t = self.tiles.p.tile
        f = self.tiles.fill()
        ny = self.h // t + 1
        nx = self.w // t + 1
        return np.tile(f, (ny, nx, 1))[:self.h, :self.w]

    # -- 3 : props ---------------------------------------------------------
    def props(self):
        """Éléments posés : rochers, cristaux. Semis DÉTERMINISTE."""
        h, w = self.h, self.w
        out = np.zeros((h, w, 4), np.float32)
        rng = np.random.default_rng(self.seed + 100)   # graine FIXE
        for i in range(14):
            cx = int(rng.integers(8, w - 8))
            cy = int(rng.integers(h // 2, h - 6))
            r = int(rng.integers(3, 8))
            yy, xx = np.indices((h, w))
            d = np.hypot(xx - cx, (yy - cy) * 1.6)
            m = (d < r).astype(np.float32)
            if m.sum() == 0:
                continue
            col = self.pal[2] * (0.8 + 0.4 * float(rng.random()))
            sub = np.zeros((h, w, 3), np.float32) + col[None, None, :]
            sub = PX.emboss(sub, m, light=0.34, shade=0.28)
            out[..., :3] = out[..., :3] * (1 - m[..., None]) + sub * m[..., None]
            out[..., 3] = np.maximum(out[..., 3], m)
        out[..., :3] = PX.dither(out[..., :3], levels=10)
        return out

    # -- 4 : fx ------------------------------------------------------------
    def fx(self, phase=0.0):
        """Couche additive animée. BOUCLE exactement sur un cycle."""
        h, w = self.h, self.w
        yy, xx = np.indices((h, w)).astype(np.float32)
        add = np.zeros((h, w, 3), np.float32)
        rng = np.random.default_rng(self.seed + 200)   # graine FIXE
        n = 26
        px = rng.random(n)
        py = rng.random(n)
        ph = rng.random(n)
        col = self.pal[4]
        for i in range(n):
            t = (phase + ph[i]) % 1.0
            cx = px[i] * w
            cy = ((py[i] - t * 0.6) % 1.0) * h        # montee, boucle exacte
            tw = 0.5 + 0.5 * np.cos(TAU * t)          # periode ENTIERE
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            add += np.exp(-d2 / 5.0)[..., None] * col[None, None, :] * \
                np.float32(tw * 0.9)
        return np.clip(add, 0.0, 1.0)

    # -- composition -------------------------------------------------------
    def compose(self, phase=0.0):
        """Les cinq couches empilées, pour l'aperçu."""
        out = self.sky(phase)
        far = self.far(phase)
        out = out * (1 - far[..., 3:]) + far[..., :3] * far[..., 3:]
        g = self.ground()
        hz = int(self.h * 0.52)
        out[hz:] = g[hz:]
        pr = self.props()
        pr[:hz, 3] = 0.0
        out = out * (1 - pr[..., 3:]) + pr[..., :3] * pr[..., 3:]
        out = np.clip(out + self.fx(phase) * 0.85, 0.0, 1.0)
        return PX.dither(out, levels=14)
