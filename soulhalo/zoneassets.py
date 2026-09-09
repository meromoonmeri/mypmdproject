"""Traitement des calques GÉNÉRÉS : détourage, nettoyage, assemblage.

Le générateur d'images sort de belles planches, mais pas des assets de
jeu. Trois défauts systématiques, et leur correction :

1. LE FOND N'EST PAS VRAIMENT UNIFORME. Le magenta demandé arrive avec
   des milliers de nuances proches (compression, dégradés parasites). Un
   détourage sur l'égalité exacte laisserait un liseré magenta autour de
   chaque objet. On travaille donc par DISTANCE à la couleur-clé, avec
   une passe qui ronge le halo résiduel.

2. « PIXEL ART » NE VEUT PAS DIRE PALETTE BORNÉE. Une planche annoncée en
   12 couleurs en contient 145 000. On quantifie explicitement, sinon le
   résultat jure à côté des sprites SpriteCollab (15 couleurs).

3. LA GRILLE DE PIXELS EST FLOUE. Le modèle dessine « à la manière du
   pixel art » sans respecter une grille : les blocs font 3,7 px de côté.
   On rééchantillonne sur une grille franche.

Le format de sortie suit celui du dépôt guilde : un PNG RGBA transparent
par calque, plus un manifeste JSON.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

MAGENTA = (1.0, 0.0, 1.0)


def _f(im):
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


# --------------------------------------------------------------------------
# détourage
# --------------------------------------------------------------------------
def key_out(img, cle=MAGENTA, seuil=0.34, ronge=1):
    """Détoure une couleur-clé et renvoie du RGBA.

    `seuil` est une distance dans l'espace RGB, pas une égalité : le fond
    généré n'est jamais parfaitement uniforme.

    `ronge` retire N pixels sur le pourtour du sujet. Sans ça, il reste un
    liseré de pixels à moitié teintés par la clé — le fameux halo magenta
    qui trahit un détourage bâclé.
    """
    a = _f(img)
    k = np.asarray(cle, np.float32)
    d = np.sqrt(((a - k[None, None, :]) ** 2).sum(2))
    alpha = (d > seuil).astype(np.float32)

    for _ in range(max(0, int(ronge))):
        e = alpha.copy()
        e[1:] = np.minimum(e[1:], alpha[:-1])
        e[:-1] = np.minimum(e[:-1], alpha[1:])
        e[:, 1:] = np.minimum(e[:, 1:], alpha[:, :-1])
        e[:, :-1] = np.minimum(e[:, :-1], alpha[:, 1:])
        alpha = e

    # les pixels gardés qui tirent encore vers la clé sont désaturés
    proche = (d < seuil * 1.9) & (alpha > 0)
    if proche.any():
        lum = a[proche].mean(1, keepdims=True)
        a[proche] = a[proche] * 0.35 + lum * 0.65

    out = np.zeros(a.shape[:2] + (4,), np.float32)
    out[..., :3] = a
    out[..., 3] = alpha
    return out


def key_out_black(img, seuil=0.10):
    """Pour les calques ADDITIFS (particules) : le noir devient l'alpha.

    Une particule lumineuse n'a pas de contour net — son alpha, c'est sa
    luminosité. Un détourage binaire lui couperait son dégradé.
    """
    a = _f(img)
    lum = a.max(2)
    alpha = np.clip((lum - seuil) / max(1e-6, 1.0 - seuil), 0.0, 1.0)
    out = np.zeros(a.shape[:2] + (4,), np.float32)
    out[..., :3] = a
    out[..., 3] = alpha
    return out


# --------------------------------------------------------------------------
# nettoyage pixel art
# --------------------------------------------------------------------------
def snap_grid(rgba, bloc):
    """Rééchantillonne sur une grille de `bloc` px, au plus proche voisin.

    Le générateur dessine des « pixels » de taille irrégulière. En
    réduisant puis en réagrandissant, on impose une grille franche — c'est
    la même idée que le downscale-then-upscale des pixel artistes.
    """
    b = max(1, int(bloc))
    h, w = rgba.shape[:2]
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8),
                         "RGBA")
    petit = im.resize((max(1, w // b), max(1, h // b)), Image.NEAREST)
    return np.asarray(petit, np.float32) / 255.0


def quantize_palette(rgba, n=16):
    """Borne le nombre de couleurs, alpha préservé.

    Les sprites SpriteCollab tiennent en 15 couleurs. Un décor à 145 000
    couleurs posé à côté ne ressemble plus à du pixel art.
    """
    a = np.clip(rgba, 0, 1)
    rgb = a[..., :3]
    alpha = a[..., 3] if a.shape[2] == 4 else np.ones(a.shape[:2], np.float32)
    vis = alpha > 0.5
    if vis.sum() < 4:
        return a
    im = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")
    q = im.quantize(colors=int(n), method=Image.MEDIANCUT, dither=Image.NONE)
    out = np.asarray(q.convert("RGB"), np.float32) / 255.0
    res = np.zeros(a.shape[:2] + (4,), np.float32)
    res[..., :3] = out
    res[..., 3] = alpha
    return res


def prepare(path, mode="key", bloc=4, couleurs=16, seuil=0.34, ronge=1):
    """Chaîne complète : charger -> détourer -> grille -> palette."""
    im = Image.open(path)
    if mode == "black":
        r = key_out_black(im)
    elif mode == "opaque":
        a = _f(im)
        r = np.concatenate([a, np.ones(a.shape[:2] + (1,), np.float32)], 2)
    else:
        r = key_out(im, seuil=seuil, ronge=ronge)
    r = snap_grid(r, bloc)
    return quantize_palette(r, couleurs)


# --------------------------------------------------------------------------
# découpe des planches d'objets
# --------------------------------------------------------------------------
def decoupe(rgba, min_px=40):
    """Isole chaque objet d'une planche détourée.

    Étiquetage par propagation : on parcourt les pixels opaques et on
    agrège les voisins. Les objets sont censés ne pas se toucher — la
    consigne de génération l'exigeait — donc un simple remplissage suffit.
    """
    alpha = rgba[..., 3] > 0.5
    h, w = alpha.shape
    vu = np.zeros((h, w), bool)
    boites = []
    for y in range(h):
        for x in range(w):
            if not alpha[y, x] or vu[y, x]:
                continue
            pile = [(y, x)]
            vu[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            n = 0
            while pile:
                cy, cx = pile.pop()
                n += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and \
                                alpha[ny, nx] and not vu[ny, nx]:
                            vu[ny, nx] = True
                            pile.append((ny, nx))
            if n >= min_px:
                boites.append((x0, y0, x1 + 1, y1 + 1))
    return boites


def extraire(rgba, boites):
    return [rgba[b[1]:b[3], b[0]:b[2]].copy() for b in boites]


# --------------------------------------------------------------------------
# entrées / sorties
# --------------------------------------------------------------------------
def save_rgba(rgba, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    a = np.clip(rgba, 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8), "RGBA").save(path)


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0


def blit(dst, src, x, y):
    """Compose `src` RGBA sur `dst` RGBA à la position (x, y)."""
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(dw, x + sw), min(dh, y + sh)
    if x2 <= x1 or y2 <= y1:
        return dst
    s = src[y1 - y:y2 - y, x1 - x:x2 - x]
    d = dst[y1:y2, x1:x2]
    sa = s[..., 3:]
    da = d[..., 3:]
    out_a = sa + da * (1 - sa)
    safe = np.maximum(out_a, 1e-6)
    d[..., :3] = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / safe
    d[..., 3:] = out_a
    return dst


def tile_fill(tuile, w, h):
    """Pave une surface avec une tuile."""
    th, tw = tuile.shape[:2]
    ny, nx = h // th + 2, w // tw + 2
    return np.tile(tuile, (ny, nx, 1))[:h, :w]
