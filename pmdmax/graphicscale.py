"""
Graphic scale : agrandissement de pixel-art **sans introduire de couleur**.

Pourquoi c'est important ici : SpriteBot refuse les pixels semi-transparents
et râle au-delà de 15 couleurs. Tout filtre bilinéaire/lanczos est donc
interdit. On utilise les familles EPX/Scale2x/Scale3x, qui ne font que
*recopier* des pixels existants : la palette de sortie est un sous-ensemble
exact de la palette d'entrée, et l'alpha reste binaire.

Facteurs non entiers : on monte en EPX jusqu'à un entier composable
(2, 3, 4, 6, 8, 9, 12...) puis on redescend en "nearest" — qui, lui aussi,
se contente de choisir un pixel existant.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Égalité RGBA pixel à pixel -> masque booléen HxW."""
    return np.all(a == b, axis=-1)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Décalage avec réplication des bords (clamp), comme les impls EPX."""
    out = a
    if dy:
        out = np.roll(out, dy, axis=0)
        if dy > 0:
            out[:dy] = a[:1]
        else:
            out[dy:] = a[-1:]
    if dx:
        prev = out
        out = np.roll(out, dx, axis=1)
        if dx > 0:
            out[:, :dx] = prev[:, :1]
        else:
            out[:, dx:] = prev[:, -1:]
    return out


def scale2x(img: np.ndarray) -> np.ndarray:
    """EPX / Scale2x. img : HxWx4 uint8 -> 2Hx2Wx4."""
    e = img
    b = _shift(img, 1, 0)    # au-dessus
    h = _shift(img, -1, 0)   # en-dessous
    d = _shift(img, 0, 1)    # à gauche
    f = _shift(img, 0, -1)   # à droite

    out = np.repeat(np.repeat(e, 2, axis=0), 2, axis=1)

    m0 = _eq(d, b) & ~_eq(d, h) & ~_eq(b, f)
    m1 = _eq(b, f) & ~_eq(b, d) & ~_eq(f, h)
    m2 = _eq(h, d) & ~_eq(h, f) & ~_eq(d, b)
    m3 = _eq(f, h) & ~_eq(f, b) & ~_eq(h, d)

    out[0::2, 0::2][m0] = d[m0]
    out[0::2, 1::2][m1] = f[m1]
    out[1::2, 0::2][m2] = d[m2]
    out[1::2, 1::2][m3] = f[m3]
    return out


def scale3x(img: np.ndarray) -> np.ndarray:
    """Scale3x. img : HxWx4 uint8 -> 3Hx3Wx4."""
    e = img
    a = _shift(img, 1, 1)
    b = _shift(img, 1, 0)
    c = _shift(img, 1, -1)
    d = _shift(img, 0, 1)
    f = _shift(img, 0, -1)
    g = _shift(img, -1, 1)
    h = _shift(img, -1, 0)
    i = _shift(img, -1, -1)

    out = np.repeat(np.repeat(e, 3, axis=0), 3, axis=1)

    db = _eq(d, b) & ~_eq(d, h) & ~_eq(b, f)
    bf = _eq(b, f) & ~_eq(b, d) & ~_eq(f, h)
    hd = _eq(h, d) & ~_eq(h, f) & ~_eq(d, b)
    fh = _eq(f, h) & ~_eq(f, b) & ~_eq(h, d)

    m = db
    out[0::3, 0::3][m] = d[m]
    m = (db & ~_eq(e, c)) | (bf & ~_eq(e, a))
    out[0::3, 1::3][m] = b[m]
    m = bf
    out[0::3, 2::3][m] = f[m]
    m = (hd & ~_eq(e, a)) | (db & ~_eq(e, g))
    out[1::3, 0::3][m] = d[m]
    # centre : inchangé
    m = (bf & ~_eq(e, i)) | (fh & ~_eq(e, c))
    out[1::3, 2::3][m] = f[m]
    m = hd
    out[2::3, 0::3][m] = d[m]
    m = (fh & ~_eq(e, g)) | (hd & ~_eq(e, i))
    out[2::3, 1::3][m] = h[m]
    m = fh
    out[2::3, 2::3][m] = f[m]
    return out


def nearest(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Rééchantillonnage plus-proche-voisin (aucune couleur inventée)."""
    h, w = img.shape[:2]
    ys = np.minimum((np.arange(out_h) * h) // out_h, h - 1)
    xs = np.minimum((np.arange(out_w) * w) // out_w, w - 1)
    return img[ys][:, xs]


#: Facteurs entiers atteignables par composition d'EPX.
_COMPOSABLE = {
    1: [],
    2: [2],
    3: [3],
    4: [2, 2],
    6: [2, 3],
    8: [2, 2, 2],
    9: [3, 3],
    12: [2, 2, 3],
    16: [2, 2, 2, 2],
    18: [2, 3, 3],
}


def _epx_chain(img: np.ndarray, factor: int) -> np.ndarray:
    for step in _COMPOSABLE[factor]:
        img = scale2x(img) if step == 2 else scale3x(img)
    return img


def graphic_scale(img: np.ndarray, factor: float, mode: str = "epx") -> np.ndarray:
    """
    Agrandit `img` (HxWx4 uint8) d'un facteur quelconque, palette préservée.

    mode="epx"     : EPX puis nearest si le facteur n'est pas composable.
    mode="nearest" : nearest pur (rendu très cubique, "gros pixels").

    Le facteur peut être fractionnaire (1.5, 1.75, 2.25...).
    """
    if abs(factor - 1.0) < 1e-9:
        return img.copy()
    if factor <= 0:
        raise ValueError("factor doit être > 0")

    h, w = img.shape[:2]
    out_w = max(1, int(round(w * factor)))
    out_h = max(1, int(round(h * factor)))

    if mode == "nearest":
        return nearest(img, out_w, out_h)
    if mode != "epx":
        raise ValueError(f"mode inconnu: {mode}")

    # Plus petit entier composable >= factor
    target = None
    for cand in sorted(_COMPOSABLE):
        if cand >= factor - 1e-9:
            target = cand
            break
    if target is None:
        target = max(_COMPOSABLE)

    big = _epx_chain(img, target)
    if big.shape[0] == out_h and big.shape[1] == out_w:
        return big
    return nearest(big, out_w, out_h)


def scale_point(p: Tuple[int, int], anchor: Tuple[int, int], factor: float) -> Tuple[int, int]:
    """
    Position d'un point après `graphic_scale`, exprimée en delta par rapport
    à l'ancre. Cohérent avec le rééchantillonnage : le centre du pixel p
    (p + 0.5) est mis à l'échelle puis re-discrétisé.
    """
    dx = math.floor((p[0] + 0.5) * factor) - math.floor((anchor[0] + 0.5) * factor)
    dy = math.floor((p[1] + 0.5) * factor) - math.floor((anchor[1] + 0.5) * factor)
    return dx, dy


def palette_of(img: np.ndarray) -> set:
    """Ensemble des couleurs RGBA opaques présentes."""
    flat = img.reshape(-1, 4)
    flat = flat[flat[:, 3] > 0]
    return set(map(tuple, np.unique(flat, axis=0).tolist())) if len(flat) else set()
