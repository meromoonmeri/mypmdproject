#!/usr/bin/env python3
"""Finit une planche generee au format des planches `structures_pmd/`.

    python3 scripts/finition_planche.py planche_generee.png structures_pmd/XX_nom.png

Trois etapes, dans l'ordre :

1. detourage du fond uni (couleur mediane des bords) par inondation depuis la
   frontiere -> alpha dur 0/255, sans halo ni pixel sombre colle aux bordures ;
2. reechantillonnage NEAREST du CADRE COMPLET sur le canevas 338 x 267 : comme la
   planche generee a le meme rapport que le canevas, les marges des modules tombent
   d'elles-memes sur celles de la planche Arcanin ;
3. palette ramenee a 32 couleurs (median-cut, sans trame) puis ecriture en PNG RGBA.

Un apercu x3 sur fond sombre est ecrit a cote de la sortie (<sortie>_3x.png) pour
controler a l'oeil, notamment la lisibilite des yeux apres reduction.
"""
import sys
from collections import deque

import numpy as np
from PIL import Image

CANVAS = (338, 267)
NCOLORS = 32
BG_TOL = 70
PREV = (30, 30, 36, 255)


def keyout(rgb):
    """Alpha dur : tout ce qui est joignable au bord et proche de la couleur de fond devient transparent."""
    h, w = rgb.shape[:2]
    bg = np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]), 0)
    near_bg = np.abs(rgb.astype(int) - bg.astype(int)).sum(-1) <= BG_TOL
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_bg[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_bg[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and near_bg[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return np.where(seen, 0, 255).astype(np.uint8)


def finish(src, dst):
    rgb = np.array(Image.open(src).convert('RGB'))
    img = Image.fromarray(np.dstack([rgb, keyout(rgb)]), 'RGBA')
    img = img.resize(CANVAS, Image.NEAREST)
    pal = img.convert('RGB').quantize(colors=NCOLORS, method=Image.MEDIANCUT,
                                      dither=Image.NONE).convert('RGB')
    out = np.dstack([np.array(pal), np.array(img)[:, :, 3]])
    Image.fromarray(out, 'RGBA').save(dst)

    ys, xs = np.where(out[..., 3] > 0)
    ncol = len(np.unique(out[..., :3][out[..., 3] > 0].reshape(-1, 3), axis=0))
    print(f'{dst}: {CANVAS[0]}x{CANVAS[1]}  contenu y {ys.min()}-{ys.max()} x {xs.min()}-{xs.max()}  '
          f'{ncol} couleurs  {int((out[..., 3] > 0).sum())} px opaques')
    print(f'  (reperes planche Arcanin : y 3-253, x 30-327)')

    zoom = Image.fromarray(out, 'RGBA').resize((CANVAS[0] * 3, CANVAS[1] * 3), Image.NEAREST)
    prev = Image.new('RGBA', zoom.size, PREV)
    prev.alpha_composite(zoom)
    preview = dst[:-4] + '_3x.png' if dst.lower().endswith('.png') else dst + '_3x.png'
    prev.save(preview)
    print(f'  apercu x3 : {preview}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    finish(sys.argv[1], sys.argv[2])
