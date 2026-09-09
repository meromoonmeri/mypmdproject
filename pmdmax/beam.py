"""
FX de transformation Dynamax : la colonne d'énergie et les éclairs spiralés.

Découpage voulu :
  1. une COLONNE d'énergie rouge OPAQUE tombe du ciel sur le Pokémon ;
  2. elle frappe le sol (flash + onde de choc) ;
  3. plusieurs ÉCLAIRS rouges remontent en SPIRALE et condensent les nuages
     Dynamax au-dessus de la tête ;
  4. la colonne se dissipe, les nuages restent et se mettent à tourner.

Les planches sont exportables sur fond MAGENTA (#FF00FF) — le magenta pur
n'appartient pas à la palette Dynamax, il sert donc de fond de chroma-key
propre pour retoucher les frames dans Aseprite ou les monter dans un moteur.
La version à alpha transparent est conservée en parallèle pour la composition
sur les spritesheets SpriteCollab (qui, elles, interdisent tout fond).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import clouds as CL
from . import config as C
from . import pixels as P

#: Couleur de chroma-key des planches FX (hors palette Dynamax).
MAGENTA = (255, 0, 255, 255)


# --------------------------------------------------------------------------
# Éléments
# --------------------------------------------------------------------------

def draw_column(idx: np.ndarray, cx: float, top: float, bottom: float,
                width: float, rng: np.random.Generator,
                opaque: bool = True, taper: float = 0.0) -> None:
    """
    Colonne d'énergie verticale, en bandes concentriques :
        contour sombre | mid | brillant | cœur blanc-rose | brillant | mid
    `taper` (0..1) rétrécit progressivement le bas de la colonne.
    """
    if bottom <= top or width <= 0:
        return
    h, w = idx.shape
    y0, y1 = max(0, int(top)), min(h, int(math.ceil(bottom)))

    for y in range(y0, y1):
        t = (y - top) / max(1.0, bottom - top)
        # la colonne ondule légèrement et s'affine vers le bas
        wob = math.sin(t * 7.0 + rng.random() * 0.35) * (width * 0.06)
        half = width * 0.5 * (1.0 - taper * t) + wob
        if half < 0.5:
            continue
        cxx = cx + math.sin(t * 4.2) * (width * 0.05)

        # Bandes calées pour que la colonne lise ROUGE : cœur blanc très
        # étroit, fin liseré rose, puis l'essentiel de la largeur en rouge
        # vif -> rouge sombre, bord quasi noir.
        core = half * 0.14 if half >= 4.5 else 0.0
        rim = half * 0.30
        bright = half * 0.58
        mid = half * 0.86
        outer = half

        xa, xb = int(math.floor(cxx - outer)), int(math.ceil(cxx + outer))
        for x in range(max(0, xa), min(w, xb + 1)):
            d = abs(x + 0.5 - cxx)
            if d > outer:
                continue
            if d <= core:
                v = C.IDX_CORE
            elif d <= rim:
                v = C.IDX_RIM
            elif d <= bright:
                v = C.IDX_BRIGHT
            elif d <= mid:
                v = C.IDX_MID
            else:
                v = C.IDX_DARK if opaque else C.IDX_MID
            idx[y, x] = v

    # arcs qui crépitent le long de la colonne
    n_arcs = int(rng.integers(2, 5))
    for _ in range(n_arcs):
        ay = float(rng.uniform(top, bottom))
        side = 1 if rng.random() < 0.5 else -1
        length = float(rng.uniform(2.0, 5.0))
        P.draw_bolt(idx,
                    (cx + side * width * 0.35, ay),
                    (cx + side * (width * 0.5 + length), ay + rng.uniform(-3, 3)),
                    rng, segments=3, amplitude=1.4,
                    core=C.IDX_RIM, halo=C.IDX_MID)


def draw_impact(idx: np.ndarray, cx: float, cy: float, radius: float,
                rng: np.random.Generator, ring_only: bool = False) -> None:
    """Onde de choc au sol : ellipse écrasée, éclatée en pointes."""
    if radius <= 0:
        return
    h, w = idx.shape
    outer = np.zeros((h, w), dtype=bool)
    inner = np.zeros((h, w), dtype=bool)
    P.ellipse_mask(outer, cx, cy, radius, max(1.0, radius * 0.34))
    P.ellipse_mask(inner, cx, cy, radius * 0.62, max(0.6, radius * 0.21))

    ring = outer & ~inner
    idx[ring] = C.IDX_RIM
    edge = P.outline_of(outer) & (idx == 0)
    idx[edge] = C.IDX_DARK
    if not ring_only:
        idx[inner] = C.IDX_BRIGHT
        core = np.zeros((h, w), dtype=bool)
        P.ellipse_mask(core, cx, cy, radius * 0.26, max(0.6, radius * 0.1))
        idx[core] = C.IDX_CORE

    # pointes d'énergie qui giclent sur les côtés
    for _ in range(int(rng.integers(3, 7))):
        a = float(rng.uniform(0, 2 * math.pi))
        ln = radius * float(rng.uniform(0.5, 1.0))
        P.draw_line(idx, (cx + math.cos(a) * radius * 0.8,
                          cy + math.sin(a) * radius * 0.28),
                    (cx + math.cos(a) * (radius * 0.8 + ln),
                     cy + math.sin(a) * (radius * 0.28 + ln * 0.34)),
                    C.IDX_RIM, thick=1)


def draw_spiral_bolts(idx: np.ndarray, cx: float, cy: float, progress: float,
                      rng: np.random.Generator, arms: int = 3,
                      radius: float = 11.0, height: float = 16.0,
                      turns: float = 1.35) -> None:
    """
    Les éclairs qui montent en spirale et « tricotent » les nuages.

    progress 0..1 : 0 = les éclairs partent du sol, 1 = ils ont atteint le
    sommet, là où la couronne de nuages va se former.
    """
    progress = max(0.0, min(1.0, progress))
    for arm in range(arms):
        base_a = 2 * math.pi * arm / arms + progress * 2.2
        pts: List[Tuple[float, float]] = []
        steps = 13
        # Fenêtre glissante : seule la portion haute de chaque bras est
        # tracée. Sans ça, les spirales s'accumulent et enterrent le Pokémon.
        t_start = max(0.0, progress - 0.55)
        for i in range(steps + 1):
            t = t_start + (progress - t_start) * (i / steps)
            a = base_a + t * turns * 2 * math.pi
            # la spirale se resserre en montant
            r = radius * (1.0 - 0.45 * t)
            jitter = float(rng.uniform(-0.8, 0.8)) * math.sin(math.pi * min(1.0, t + 0.05))
            pts.append((cx + math.cos(a) * r + jitter,
                        cy - t * height + math.sin(a) * r * 0.34))
        if len(pts) < 2:
            continue
        # trait fin : halo cramoisi 2 px + cœur clair 1 px
        P.draw_polyline(idx, pts, C.IDX_MID, thick=2)
        P.draw_polyline(idx, pts, C.IDX_RIM, thick=1)
        # étincelle en tête de spirale
        hx, hy = pts[-1]
        P.draw_bolt(idx, (hx, hy),
                    (hx + rng.uniform(-3, 3), hy - rng.uniform(1, 4)),
                    rng, segments=3, amplitude=1.2)


def draw_falling_sparks(idx: np.ndarray, cx: float, w_spread: float,
                        top: float, bottom: float, count: int,
                        rng: np.random.Generator) -> None:
    """Petites traînées d'énergie qui tombent autour de la colonne."""
    h, w = idx.shape
    for _ in range(count):
        x = cx + float(rng.uniform(-w_spread, w_spread))
        y = float(rng.uniform(top, bottom))
        ln = float(rng.uniform(1.5, 4.0))
        P.draw_line(idx, (x, y), (x, y + ln),
                    C.IDX_RIM if rng.random() < 0.6 else C.IDX_CORE, thick=1)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

@dataclass
class BeamFrame:
    """Une frame de FX + l'info dont le pipeline a besoin pour composer."""
    idx: np.ndarray          # grille d'index (0 = vide)
    behind: np.ndarray       # partie à dessiner DERRIÈRE le Pokémon
    scale_hint: float        # facteur de taille du Pokémon à cette frame
    flash: bool = False      # frame de flash (le sprite est blanchi)


#: Scénario par défaut : 16 frames, ~2,5 s à 60 fps avec les durées de
#: pmdmax.transform. Chaque tuple décrit une étape du storyboard.
DEFAULT_STORY = [
    # (nom,            t_debut, t_fin, n_frames)
    ("gather",   0.00, 0.14, 2),   # le ciel s'assombrit, étincelles montantes
    ("descend",  0.14, 0.38, 4),   # la colonne tombe
    ("impact",   0.38, 0.50, 2),   # elle frappe : flash
    ("spiral",   0.50, 0.76, 4),   # les éclairs montent en spirale
    ("condense", 0.76, 0.92, 3),   # les nuages se forment
    ("settle",   0.92, 1.00, 1),   # tout se stabilise
]


def render_beam_frame(w: int, h: int, cx: float, ground_y: float,
                      head_y: float, t: float, seed: int = 0,
                      column_width: float = 13.0) -> BeamFrame:
    """
    Rend une frame du FX complet à l'instant normalisé `t` (0..1).

    cx, ground_y : point d'impact au sol (= centre/pied du Pokémon).
    head_y       : altitude où la couronne de nuages doit se former.
    """
    rng = np.random.default_rng(seed * 104729 + int(t * 100000))
    fx = P.new_idx(w, h)
    behind = P.new_idx(w, h)
    flash = False
    scale_hint = 1.0

    # ---- 1. rassemblement : étincelles + halo au sol -------------------
    if t < 0.14:
        u = t / 0.14
        draw_falling_sparks(fx, cx, column_width * 0.9, 0, ground_y,
                            int(2 + 6 * u), rng)
        draw_impact(behind, cx, ground_y, 3.0 + 6.0 * u, rng, ring_only=True)

    # ---- 2. descente de la colonne -------------------------------------
    elif t < 0.38:
        u = (t - 0.14) / 0.24
        # front de la colonne : accélération quadratique
        front = u * u * ground_y
        draw_column(fx, cx, 0.0, max(2.0, front), column_width, rng,
                    opaque=True, taper=0.12)
        draw_falling_sparks(fx, cx, column_width * 1.1, 0, front, 5, rng)
        draw_impact(behind, cx, ground_y, 6.0 + 5.0 * u, rng, ring_only=True)
        scale_hint = 1.0 + 0.03 * u

    # ---- 3. impact : flash + onde de choc ------------------------------
    elif t < 0.50:
        u = (t - 0.38) / 0.12
        flash = u < 0.55
        draw_column(fx, cx, 0.0, ground_y, column_width * (1.0 + 0.35 * (1 - u)),
                    rng, opaque=True, taper=0.0)
        draw_impact(fx, cx, ground_y, 9.0 + 13.0 * u, rng)
        draw_impact(behind, cx, ground_y, 12.0 + 16.0 * u, rng, ring_only=True)
        scale_hint = 1.03 + 0.22 * u

    # ---- 4. spirale ascendante -----------------------------------------
    elif t < 0.76:
        u = (t - 0.50) / 0.26
        # la colonne s'estompe en remontant
        draw_column(fx, cx, ground_y * u * 0.9, ground_y,
                    column_width * (1.0 - 0.55 * u), rng,
                    opaque=False, taper=0.45)
        draw_spiral_bolts(fx, cx, ground_y, u, rng, arms=3,
                          radius=11.0, height=max(4.0, ground_y - head_y))
        draw_impact(behind, cx, ground_y, 20.0 * (1.0 - u) + 6.0, rng, ring_only=True)
        scale_hint = 1.25 + 0.45 * u

    # ---- 5. condensation des nuages ------------------------------------
    elif t < 0.92:
        u = (t - 0.76) / 0.16
        draw_spiral_bolts(fx, cx, ground_y, 1.0 - 0.7 * u, rng, arms=3,
                          radius=11.0 * (1.0 - 0.35 * u),
                          height=max(4.0, ground_y - head_y))
        cb, cf = CL.render_cloud_layer(
            w, h, cx, head_y, phase=0.15 + u * 0.35,
            count=3, rx=9.0, ry=5.0,
            base_scale=1.6 + 2.6 * u, bolts=True, seed=seed)
        P.blit_idx(behind, cb, 0, 0)
        P.blit_idx(fx, cf, 0, 0)
        scale_hint = 1.70 + 0.22 * u

    # ---- 6. stabilisation ----------------------------------------------
    else:
        u = (t - 0.92) / 0.08
        cb, cf = CL.render_cloud_layer(
            w, h, cx, head_y, phase=0.5 + u * 0.12,
            count=3, rx=9.0, ry=5.0, base_scale=4.2,
            bolts=True, seed=seed)
        P.blit_idx(behind, cb, 0, 0)
        P.blit_idx(fx, cf, 0, 0)
        scale_hint = 1.92 + 0.08 * u

    return BeamFrame(idx=fx, behind=behind, scale_hint=scale_hint, flash=flash)


def render_beam_frames(w: int, h: int, cx: float, ground_y: float,
                       head_y: float, n_frames: int = 16, seed: int = 0,
                       column_width: float = 13.0) -> List[BeamFrame]:
    """Rend la séquence complète du FX."""
    out = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        out.append(render_beam_frame(w, h, cx, ground_y, head_y, t,
                                     seed=seed, column_width=column_width))
    return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def frames_to_sheet(frames: List[BeamFrame], w: int, h: int,
                    background: Optional[Tuple[int, int, int, int]] = MAGENTA,
                    include_behind: bool = True) -> np.ndarray:
    """
    Assemble les frames en une bande horizontale.

    background=MAGENTA -> planche de chroma-key (ce que l'on veut pour
    retoucher/monter le FX). background=None -> fond transparent, utilisé
    pour composer sur les spritesheets SpriteCollab.
    """
    n = len(frames)
    sheet = np.zeros((h, w * n, 4), dtype=np.uint8)
    if background is not None:
        sheet[:, :, :] = np.array(background, dtype=np.uint8)

    for i, fr in enumerate(frames):
        merged = fr.idx.copy()
        if include_behind:
            m = (merged == 0) & (fr.behind > 0)
            merged[m] = fr.behind[m]
        rgba = P.to_rgba(merged)
        opaque = rgba[:, :, 3] > 0
        region = sheet[:, i * w:(i + 1) * w]
        region[opaque] = rgba[opaque]

    return sheet
