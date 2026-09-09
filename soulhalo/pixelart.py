"""Rendu pixel art travaillé : tramage ordonné, volume 2.5D, palette bornée.

Trois outils qui donnent au rendu la facture d'un vrai pixel art plutôt
que celle d'un dégradé lissé mis à l'échelle :

`dither`      remplace les dégradés continus par une trame de Bayer. Une
              rampe lisse agrandie au plus proche voisin donne des bandes
              franches ; la trame casse ces bandes en motif régulier, ce
              que fait tout pixel art travaillé.

`quantize`    borne le nombre de couleurs. SpriteCollab impose 15 couleurs
              par sprite ; un panneau d'interface qui en compte 4000 jure
              à côté.

`emboss`      donne le relief 2.5D : lumière en haut-gauche, ombre en
              bas-droite, ombre portée décalée. C'est ce qui fait qu'un
              panneau semble posé SUR l'écran au lieu d'être peint dedans.

Tout travaille sur la grille logique, avant l'agrandissement entier — un
tramage appliqué après agrandissement produirait des points de 3x3 pixels
au lieu d'une trame fine.
"""
from __future__ import annotations

import numpy as np

# Matrice de Bayer 4x4, normalisée sur [0,1). L'ordre est celui du seuil
# classique : chaque valeur dit « à partir de quelle fraction ce pixel
# s'allume ». C'est la trame la plus lisible à petite taille.
BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], np.float32) / 16.0

BAYER2 = np.array([[0, 2], [3, 1]], np.float32) / 4.0


def bayer(h, w, size=4):
    """Trame de Bayer répétée sur (h, w)."""
    m = BAYER4 if size == 4 else BAYER2
    n = m.shape[0]
    return np.tile(m, (h // n + 1, w // n + 1))[:h, :w]


def dither(buf, levels=12, strength=1.0, size=4):
    """Quantifie chaque canal sur `levels` paliers, avec tramage ordonné.

    Sans tramage, réduire à 12 paliers crée des bandes visibles dans les
    dégradés. Le seuil de Bayer décale chaque pixel juste avant l'arrondi,
    ce qui dissout la bande en motif — la technique du pixel art depuis
    toujours.
    """
    a = np.clip(np.asarray(buf, np.float32), 0.0, 1.0)
    if levels < 2:
        return a
    h, w = a.shape[:2]
    t = (bayer(h, w, size) - 0.5)[..., None] * (strength / (levels - 1))
    return np.clip(np.round((a + t) * (levels - 1)) / (levels - 1), 0.0, 1.0)


def quantize(buf, palette):
    """Ramène chaque pixel à la couleur la plus proche de `palette`."""
    a = np.clip(np.asarray(buf, np.float32), 0.0, 1.0)
    p = np.asarray(palette, np.float32).reshape(-1, 3)
    flat = a.reshape(-1, 3)
    # distance au carré à chaque entrée de palette, en une passe
    d = ((flat[:, None, :] - p[None, :, :]) ** 2).sum(2)
    return p[d.argmin(1)].reshape(a.shape)


def ramp(base, n=5, lo=0.55, hi=1.35):
    """Rampe de teintes d'une même couleur, du plus sombre au plus clair.

    Un pixel art propre n'utilise pas n'importe quelles nuances : il
    décline une couleur en quelques paliers réguliers.
    """
    b = np.asarray(base, np.float32)
    ks = np.linspace(lo, hi, n, dtype=np.float32)[:, None]
    return np.clip(b[None, :] * ks, 0.0, 1.0)


def emboss(buf, mask, light=0.35, shade=0.30, depth=1):
    """Relief 2.5D : arête claire en haut-gauche, sombre en bas-droite.

    `mask` vaut 1 sur la forme à mettre en relief. On compare le masque à
    lui-même décalé : là où la forme commence en venant du haut-gauche,
    c'est une arête éclairée ; là où elle finit, une arête d'ombre.
    """
    a = np.asarray(buf, np.float32).copy()
    m = np.asarray(mask, np.float32)
    d = max(1, int(depth))

    up = np.zeros_like(m)
    up[d:, :] = m[:-d, :]
    left = np.zeros_like(m)
    left[:, d:] = m[:, :-d]
    down = np.zeros_like(m)
    down[:-d, :] = m[d:, :]
    right = np.zeros_like(m)
    right[:, :-d] = m[:, d:]

    hi = np.clip(m - np.minimum(up, left), 0.0, 1.0)      # bord haut-gauche
    lo = np.clip(m - np.minimum(down, right), 0.0, 1.0)   # bord bas-droite
    a += hi[..., None] * light
    a -= lo[..., None] * shade
    return np.clip(a, 0.0, 1.0)


def drop_shadow(buf, mask, dx=2, dy=2, opacity=0.45):
    """Ombre portée dure, décalée — pas de flou.

    Une ombre floue trahit le rendu vectoriel. Le pixel art pose une ombre
    nette, décalée d'un ou deux pixels, souvent tramée.
    """
    a = np.asarray(buf, np.float32).copy()
    m = np.asarray(mask, np.float32)
    h, w = m.shape[:2]
    sh = np.zeros_like(m)
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    sh[y0:y1, x0:x1] = m[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    sh = np.clip(sh - m, 0.0, 1.0)          # jamais sous la forme elle-même
    # l'ombre est tramée : une ombre pleine paraît lourde à cette échelle
    sh = sh * (bayer(h, w, 4) > 0.35).astype(np.float32)
    return np.clip(a * (1.0 - sh[..., None] * opacity), 0.0, 1.0)


def outline(buf, mask, color=(0.16, 0.09, 0.04)):
    """Cerne la forme d'un trait 1 px. Le pixel art cerne presque tout."""
    a = np.asarray(buf, np.float32).copy()
    m = (np.asarray(mask, np.float32) > 0.5).astype(np.float32)
    g = np.zeros_like(m)
    g[1:] = np.maximum(g[1:], m[:-1])
    g[:-1] = np.maximum(g[:-1], m[1:])
    g[:, 1:] = np.maximum(g[:, 1:], m[:, :-1])
    g[:, :-1] = np.maximum(g[:, :-1], m[:, 1:])
    edge = np.clip(g - m, 0.0, 1.0)[..., None]
    return a * (1.0 - edge) + np.asarray(color, np.float32) * edge


def scanline(buf, strength=0.05, period=3):
    """Assombrit une ligne sur `period`. Donne le grain d'un écran.

    À utiliser avec parcimonie : trop marqué, on ne voit plus que ça.
    """
    a = np.asarray(buf, np.float32).copy()
    h = a.shape[0]
    k = np.ones(h, np.float32)
    k[::period] = 1.0 - strength
    return np.clip(a * k[:, None, None], 0.0, 1.0)
