"""
Primitives de dessin pixel-art.

Tout est fait sur des grilles d'INDEX (uint8, HxW) où 0 = transparent et
1..6 = les tons de C.DYNA_RAMP. On ne convertit en RGBA qu'à la toute fin
(`to_rgba`), ce qui garantit deux choses exigées par SpriteBot :
  - aucune couleur hors palette n'apparaît jamais ;
  - aucun pixel semi-transparent (l'alpha vaut 0 ou 255, jamais entre).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from . import config as C

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Grilles
# --------------------------------------------------------------------------

def new_idx(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def to_rgba(idx: np.ndarray) -> np.ndarray:
    """Grille d'index -> image RGBA (HxWx4 uint8)."""
    ramp = np.array(C.DYNA_RAMP, dtype=np.uint8)
    return ramp[np.clip(idx, 0, len(C.DYNA_RAMP) - 1)]


def blit_idx(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Colle `src` sur `dst` en (x, y) ; l'index 0 est transparent."""
    sh, sw = src.shape
    dh, dw = dst.shape
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    w = min(sw - sx0, dw - dx0)
    h = min(sh - sy0, dh - dy0)
    if w <= 0 or h <= 0:
        return
    region = src[sy0:sy0 + h, sx0:sx0 + w]
    m = region > 0
    dst[dy0:dy0 + h, dx0:dx0 + w][m] = region[m]


# --------------------------------------------------------------------------
# Masques booléens
# --------------------------------------------------------------------------

def disc_mask(mask: np.ndarray, cx: float, cy: float, r: float) -> None:
    """Ajoute un disque plein au masque booléen (en place)."""
    if r <= 0:
        return
    h, w = mask.shape
    y0, y1 = max(0, int(cy - r - 1)), min(h - 1, int(cy + r + 1))
    x0, x1 = max(0, int(cx - r - 1)), min(w - 1, int(cx + r + 1))
    if y1 < y0 or x1 < x0:
        return
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    xs = np.arange(x0, x1 + 1)[None, :] + 0.5
    mask[y0:y1 + 1, x0:x1 + 1] |= ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r


def ellipse_mask(mask: np.ndarray, cx: float, cy: float, rx: float, ry: float) -> None:
    if rx <= 0 or ry <= 0:
        return
    h, w = mask.shape
    y0, y1 = max(0, int(cy - ry - 1)), min(h - 1, int(cy + ry + 1))
    x0, x1 = max(0, int(cx - rx - 1)), min(w - 1, int(cx + rx + 1))
    if y1 < y0 or x1 < x0:
        return
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    xs = np.arange(x0, x1 + 1)[None, :] + 0.5
    mask[y0:y1 + 1, x0:x1 + 1] |= (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0


def rect_mask(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    h, w = mask.shape
    xa, xb = max(0, int(round(x0))), min(w, int(round(x1)))
    ya, yb = max(0, int(round(y0))), min(h, int(round(y1)))
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = True


def dilate(mask: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """Dilatation 1 px (sans effet de bord : on passe par un padding)."""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(p)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in offsets:
        out |= np.roll(np.roll(p, dy, axis=0), dx, axis=1)
    return out[1:-1, 1:-1]


def outline_of(mask: np.ndarray, diagonal: bool = True) -> np.ndarray:
    """Couronne de 1 px autour du masque (hors masque)."""
    return dilate(mask, diagonal) & ~mask


def run_depth(mask: np.ndarray) -> np.ndarray:
    """
    Pour chaque pixel actif, distance verticale depuis le haut de SON segment
    plein (le "run" de la colonne). Sert à ombrer une forme volumétrique :
    0 = surface éclairée, grand = profondeur / dessous.
    Renvoie -1 pour les pixels vides.
    """
    h, w = mask.shape
    depth = np.full((h, w), -1, dtype=np.int32)
    for x in range(w):
        d = 0
        col = mask[:, x]
        for y in range(h):
            if col[y]:
                depth[y, x] = d
                d += 1
            else:
                d = 0
    return depth


def bottom_run(mask: np.ndarray) -> np.ndarray:
    """Distance depuis le BAS du segment plein (0 = pixel le plus bas)."""
    return run_depth(mask[::-1])[::-1]


# --------------------------------------------------------------------------
# Traits
# --------------------------------------------------------------------------

def draw_line(idx: np.ndarray, p0: Point, p1: Point, val: int, thick: int = 1) -> None:
    """Bresenham, épaisseur en pixels carrés."""
    h, w = idx.shape
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    guard = 0
    while guard < 10000:
        guard += 1
        if thick <= 1:
            if 0 <= x0 < w and 0 <= y0 < h:
                idx[y0, x0] = val
        else:
            r = thick // 2
            xa, xb = max(0, x0 - r), min(w, x0 + thick - r)
            ya, yb = max(0, y0 - r), min(h, y0 + thick - r)
            if xb > xa and yb > ya:
                idx[ya:yb, xa:xb] = val
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_polyline(idx: np.ndarray, pts: Sequence[Point], val: int, thick: int = 1) -> None:
    for a, b in zip(pts, pts[1:]):
        draw_line(idx, a, b, val, thick)


def jagged_path(p0: Point, p1: Point, segments: int, amplitude: float,
                rng: np.random.Generator) -> List[Point]:
    """
    Chemin en zigzag entre deux points : l'ossature d'un éclair.
    Le décalage est perpendiculaire au trajet et s'annule aux extrémités.
    """
    segments = max(2, segments)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    pts: List[Point] = []
    for i in range(segments + 1):
        t = i / segments
        # fenêtre : 0 aux bouts, max au milieu
        window = math.sin(math.pi * t)
        off = float(rng.uniform(-amplitude, amplitude)) * window
        pts.append((p0[0] + dx * t + nx * off, p0[1] + dy * t + ny * off))
    return pts


def draw_bolt(idx: np.ndarray, p0: Point, p1: Point, rng: np.random.Generator,
              segments: int = 6, amplitude: float = 2.5,
              core: int = C.IDX_RIM, halo: int = C.IDX_MID,
              branches: int = 0) -> None:
    """
    Éclair : un halo épais (RIM) puis un cœur fin (CORE) par-dessus.
    `branches` ajoute des ramifications courtes partant du tronc.
    """
    pts = jagged_path(p0, p1, segments, amplitude, rng)
    if halo:
        draw_polyline(idx, pts, halo, thick=2)
    draw_polyline(idx, pts, core, thick=1)

    for _ in range(branches):
        i = int(rng.integers(1, max(2, len(pts) - 1)))
        base = pts[i]
        ang = float(rng.uniform(0, 2 * math.pi))
        ln = float(rng.uniform(2.0, 5.0))
        tip = (base[0] + math.cos(ang) * ln, base[1] + math.sin(ang) * ln)
        sub = jagged_path(base, tip, 3, 1.2, rng)
        draw_polyline(idx, sub, core, thick=1)


def draw_arc_bolt(idx: np.ndarray, cx: float, cy: float, rx: float, ry: float,
                  a0: float, a1: float, rng: np.random.Generator,
                  steps: int = 10, amplitude: float = 1.6,
                  core: int = C.IDX_CORE, halo: int = C.IDX_RIM) -> None:
    """Éclair suivant un arc d'ellipse (utilisé pour les spirales)."""
    pts: List[Point] = []
    for i in range(steps + 1):
        t = i / steps
        a = a0 + (a1 - a0) * t
        window = math.sin(math.pi * t)
        jitter = float(rng.uniform(-amplitude, amplitude)) * window
        pts.append((cx + math.cos(a) * (rx + jitter),
                    cy + math.sin(a) * (ry + jitter * 0.5)))
    if halo:
        draw_polyline(idx, pts, halo, thick=2)
    draw_polyline(idx, pts, core, thick=1)


def sprinkle(idx: np.ndarray, mask: np.ndarray, count: int, val: int,
             rng: np.random.Generator) -> None:
    """Sème `count` pixels de valeur `val` dans la zone autorisée `mask`."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or count <= 0:
        return
    pick = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    idx[ys[pick], xs[pick]] = val


# --------------------------------------------------------------------------
# Ombrage volumétrique
# --------------------------------------------------------------------------

#: Rampes d'ombrage selon l'éloignement (0 = premier plan lumineux).
#: 4 bandes, du haut éclairé vers le bas dans l'ombre.
#: Le corps du nuage doit rester ROUGE : le rose clair (RIM) n'est qu'un
#: liseré et le blanc (CORE) qu'un minuscule reflet spéculaire.
_TONE_RAMPS = {
    0: (C.IDX_RIM, C.IDX_BRIGHT, C.IDX_MID, C.IDX_DARK),    # avant, éclairé
    1: (C.IDX_BRIGHT, C.IDX_MID, C.IDX_MID, C.IDX_DARK),    # milieu
    2: (C.IDX_MID, C.IDX_DARK, C.IDX_DARK, C.IDX_OUTLINE),  # arrière, sombre
}

#: Bornes des bandes, en FRACTION de la hauteur du segment plein.
#: En relatif (et non en pixels absolus), une grosse touffe reste aussi
#: lisible qu'une petite : sinon tout ce qui dépasse 4 px vire au noir.
_BANDS = (0.20, 0.46, 0.76)


def shade_volume(mask: np.ndarray, tone: int = 0, ribs: bool = True,
                 outline: bool = True) -> np.ndarray:
    """
    Transforme un masque en volume ombré facon pixel-art PMD :
    liseré clair sur le dessus, dégradé vers le bas, contour sombre 1 px,
    et nervures horizontales optionnelles (le "côtelé" des nuages Dynamax).
    """
    ramp = _TONE_RAMPS[tone]
    depth = run_depth(mask)
    total = depth + bottom_run(mask) + 1          # hauteur du run par pixel
    frac = np.where(mask, depth / np.maximum(1, total), 1.0)

    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask & (frac < _BANDS[0])] = ramp[0]
    out[mask & (frac >= _BANDS[0]) & (frac < _BANDS[1])] = ramp[1]
    out[mask & (frac >= _BANDS[1]) & (frac < _BANDS[2])] = ramp[2]
    out[mask & (frac >= _BANDS[2])] = ramp[3]

    if ribs:
        # nervures : une ligne sur trois passe d'un cran plus sombre,
        # uniquement sous la zone de pleine lumière
        rib = mask & (depth >= 2) & (((depth - 2) % 3) == 0) & (frac >= _BANDS[0])
        darker = {C.IDX_CORE: C.IDX_RIM, C.IDX_RIM: C.IDX_BRIGHT,
                  C.IDX_BRIGHT: C.IDX_MID, C.IDX_MID: C.IDX_DARK,
                  C.IDX_DARK: C.IDX_DARK}
        for src, dst in darker.items():
            out[rib & (out == src)] = dst

    # la dernière ligne de chaque colonne passe en ombre franche
    bot = bottom_run(mask)
    out[mask & (bot == 0) & (frac >= _BANDS[1])] = ramp[3]

    if outline:
        out[outline_of(mask)] = C.IDX_OUTLINE

    return out


def shade_sphere(mask: np.ndarray, cx: float, cy: float, radius: float,
                 tone: int = 0, light: Tuple[float, float] = (-0.45, -0.62),
                 ribs: bool = True, outline: bool = True) -> np.ndarray:
    """
    Ombrage sphérique : les bandes suivent la distance à un point de lumière
    placé en haut-à-gauche, comme sur les sprites PMD officiels. Donne un
    volume rond là où `shade_volume` (purement vertical) aplatit la forme.
    """
    ramp = _TONE_RAMPS[tone]
    h, w = mask.shape
    r = max(1.0, radius)
    lx = cx + light[0] * r
    ly = cy + light[1] * r

    ys = np.arange(h)[:, None] + 0.5
    xs = np.arange(w)[None, :] + 0.5
    # distance au point de lumière, normalisée par le diamètre utile
    d = np.sqrt(((xs - lx) / (r * 1.9)) ** 2 + ((ys - ly) / (r * 1.7)) ** 2)

    out = np.zeros(mask.shape, dtype=np.uint8)
    # bandes resserrées sur la lumière : l'essentiel de la sphère reste
    # dans les deux tons les plus sombres de la rampe.
    out[mask & (d < 0.34)] = ramp[0]
    out[mask & (d >= 0.34) & (d < 0.58)] = ramp[1]
    out[mask & (d >= 0.58) & (d < 0.86)] = ramp[2]
    out[mask & (d >= 0.86)] = ramp[3]

    # minuscule reflet spéculaire au point le plus éclairé (grosses touffes)
    if r >= 3.5:
        spec = mask & (d < 0.15)
        if spec.any():
            out[spec] = C.IDX_CORE

    if ribs and r >= 3.0:
        depth = run_depth(mask)
        rib = mask & (depth >= 2) & (((depth - 1) % 3) == 0) & (d >= 0.5)
        darker = {C.IDX_CORE: C.IDX_RIM, C.IDX_RIM: C.IDX_BRIGHT,
                  C.IDX_BRIGHT: C.IDX_MID, C.IDX_MID: C.IDX_DARK,
                  C.IDX_DARK: C.IDX_DARK}
        for src, dst in darker.items():
            out[rib & (out == src)] = dst

    # base : la dernière ligne de chaque colonne reste dans l'ombre
    bot = bottom_run(mask)
    out[mask & (bot == 0) & (d >= 0.6)] = ramp[3]

    if outline:
        out[outline_of(mask)] = C.IDX_OUTLINE

    return out
