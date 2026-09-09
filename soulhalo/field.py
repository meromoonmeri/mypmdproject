"""
Outils de champ pour la parallaxe circulaire.

Tout ce qui dépend du temps doit être **périodique en phase** : chaque terme
s'exprime en cos/sin de 2*pi*k*phase avec k entier, si bien que la dernière
frame se raccorde exactement à la première. Aucun tirage aléatoire par frame.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

TAU = 2.0 * np.pi


# --------------------------------------------------------------------------
# Géométrie
# --------------------------------------------------------------------------
def polar(h: int, w: int, cx: float | None = None, cy: float | None = None):
    """Grille polaire : rayon normalisé (1.0 = demi-petit-côté) et angle."""
    cx = (w - 1) / 2.0 if cx is None else cx
    cy = (h - 1) / 2.0 if cy is None else cy
    yy, xx = np.indices((h, w)).astype(np.float32)
    dx = xx - np.float32(cx)
    dy = yy - np.float32(cy)
    r = np.hypot(dx, dy)
    rn = r / np.float32(min(h, w) / 2.0)
    th = np.arctan2(dy, dx)
    return rn.astype(np.float32), th.astype(np.float32)


def smoothstep(e0: float, e1: float, x):
    """
    Transition douce de 0 à 1 entre e0 et e1.

    Les bornes peuvent être **décroissantes** (e1 < e0) : c'est le cas usuel
    pour une vignette ou un fondu vers l'extérieur. Le garde-fou porte donc
    sur la valeur absolue de l'écart, jamais sur son signe.
    """
    d = e1 - e0
    d = d if abs(d) > 1e-6 else (1e-6 if d >= 0 else -1e-6)
    t = np.clip((x - e0) / d, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def ring(rn, center: float, width: float):
    """Anneau doux centré sur un rayon (profil en cloche, jamais négatif)."""
    d = (rn - np.float32(center)) / np.float32(max(1e-6, width))
    return np.exp(-(d * d)).astype(np.float32)


def angdiff(th, a: float):
    """Écart angulaire signé le plus court entre le champ th et l'angle a."""
    d = th - np.float32(a)
    return ((d + np.float32(np.pi)) % np.float32(TAU) - np.float32(np.pi)).astype(np.float32)


# --------------------------------------------------------------------------
# Couleur
# --------------------------------------------------------------------------
def ramp(anchors, t):
    """Rampe de couleur par interpolation linéaire. anchors = [(pos, (r,g,b))]."""
    pos = np.array([a[0] for a in anchors], dtype=np.float32)
    cols = np.array([a[1] for a in anchors], dtype=np.float32)
    out = np.empty(t.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c]).astype(np.float32)
    return out


def cos_palette(t, a=(0.5, 0.5, 0.5), b=(0.5, 0.5, 0.5),
                c=(1.0, 1.0, 1.0), d=(0.00, 0.33, 0.67)):
    """Palette arc-en-ciel continue et périodique (utile pour la teinte en theta)."""
    a = np.array(a, np.float32); b = np.array(b, np.float32)
    c = np.array(c, np.float32); d = np.array(d, np.float32)
    out = np.empty(t.shape + (3,), dtype=np.float32)
    for i in range(3):
        out[..., i] = a[i] + b[i] * np.cos(TAU * (c[i] * t + d[i]))
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------
# Flou séparable (box x3 ~ gaussien) — pur numpy, déterministe
# --------------------------------------------------------------------------
def _box1(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(ap, axis=axis, dtype=np.float32)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    lo[axis] = slice(0, n)
    return ((cs[tuple(hi)] - cs[tuple(lo)]) / np.float32(2 * r + 1)).astype(np.float32)


def blur(a: np.ndarray, radius: float, passes: int = 3) -> np.ndarray:
    r = int(max(0, round(radius)))
    if r == 0:
        return a.astype(np.float32)
    out = a.astype(np.float32)
    for _ in range(passes):
        out = _box1(out, r, 0)
        out = _box1(out, r, 1)
    return out


# --------------------------------------------------------------------------
# Plaque peinte -> bande polaire
# --------------------------------------------------------------------------
def polar_strip(path: str, n_theta: int = 1024, n_rad: int = 384,
                reach: float = 0.98) -> np.ndarray:
    """
    Convertit une plaque carrée (peinte par le générateur d'images) en bande
    polaire `(n_rad, n_theta, 3)` : une ligne = un cercle concentrique.

    L'échantillonnage sur des cercles complets rend la bande **naturellement
    cyclique en theta** : la rotation ne peut pas produire de couture.
    """
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    th = np.linspace(0.0, TAU, n_theta, endpoint=False, dtype=np.float32)
    rr = np.linspace(0.0, reach, n_rad, dtype=np.float32)[:, None]
    xs = cx + np.cos(th)[None, :] * rr * cx
    ys = cy + np.sin(th)[None, :] * rr * cy

    x0 = np.clip(xs.astype(np.int32), 0, w - 1)
    y0 = np.clip(ys.astype(np.int32), 0, h - 1)
    return a[y0, x0]


def sample_strip(strip: np.ndarray, rn, th, rot: float, rad_scale: float = 1.0):
    """
    Échantillonne la bande polaire pour chaque pixel.

    `rot` est en tours (1.0 = un tour complet) : un entier par boucle garantit
    le raccord exact de l'animation.
    """
    n_rad, n_theta = strip.shape[:2]
    # Le modulo est applique APRES conversion entiere : en float32, une
    # valeur juste sous n_theta peut s'arrondir a n_theta et deborder.
    ti = ((th / np.float32(TAU) + np.float32(rot)) * n_theta)
    ti = np.mod(ti.astype(np.int64), n_theta).astype(np.int32)
    ri = np.clip((rn * np.float32(rad_scale)) * (n_rad - 1), 0, n_rad - 1).astype(np.int32)
    return strip[ri, ti]
