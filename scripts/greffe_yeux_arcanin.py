"""Greffe les yeux « fenetre a meneaux » de la hutte Arcanin sur la planche Evoli.

1. releve du masque des yeux sur la planche Arcanin (variante de droite), decompose
   en 3 encres : prune (vitrage), caramel (barreaux + cadre), brun (contour) ;
2. reechantillonnage a la taille de chaque tete Evoli par max-pool, priorite
   prune > caramel > contour : les barreaux survivent a la reduction ;
3. effacement de l'ancien oeil du generateur par inpainting (plus proche voisin non
   efface, ligne puis colonne) -> la fourrure et son ombrage sont conserves ;
4. tamponnage du masque.

    python3 scripts/greffe_yeux_arcanin.py [planche_arcane.png] [planche_evoli.png] [sortie.png]
Les zones de greffe (ZONES) sont calibrees pour la planche Evoli en 338 x 267.
"""
import os
import sys

import numpy as np
from PIL import Image

ARC = os.path.join(os.path.dirname(__file__), '..', 'arcanine_hut_tileset_by_geoisevil-d5pujgf (2).png')
TGT = os.path.join(os.path.dirname(__file__), '..', 'structures_pmd', '12_eevee_shop.png')

SRC_ZONE = (46, 166, 64, 218)          # y0, x0, y1, x1 des deux yeux, planche Arcanin
PLUM = {(0x50, 0x40, 0x50), (0x58, 0x38, 0x50), (0x60, 0x50, 0x50), (0x39, 0x3F, 0x46),
        (0x42, 0x47, 0x4F), (0x35, 0x35, 0x33)}
CARAMEL = {(0xEF, 0x9F, 0x62), (0xCA, 0x80, 0x4F), (0xD9, 0x91, 0x5B), (0xF6, 0x94, 0x65),
           (0xE7, 0xA7, 0x79), (0xFC, 0xB6, 0x85), (0xF5, 0x7D, 0x43), (0xE6, 0x76, 0x56),
           (0xF0, 0xA8, 0x58), (0xF8, 0x70, 0x50), (0xD0, 0x78, 0x18), (0xA8, 0x58, 0x18)}
OUTLINE = {(0xA3, 0x69, 0x44), (0x65, 0x56, 0x41), (0x8B, 0x69, 0x4E), (0xA0, 0x7A, 0x5A),
           (0x80, 0x68, 0x48), (0x88, 0x68, 0x50), (0xA6, 0x81, 0x62)}
INK = {'plum': (0x4A, 0x36, 0x44), 'caramel': (0xF0, 0xA6, 0x63), 'outline': (0x3A, 0x1E, 0x12)}
# encres de l'oeil peint par le generateur, a effacer avant greffe
ERASE = {(0x39, 0x19, 0x0D), (0x3C, 0x25, 0x1C), (0x3A, 0x31, 0x33), (0x3C, 0x2E, 0x2C),
         (0x77, 0x4A, 0x2A), (0x2E, 0x2A, 0x30), (0x36, 0x2C, 0x2A), (0x30, 0x2A, 0x28)}

# tete A, tete B, grosse tete de la hutte, petite tete du support
ZONES = {'A': (42, 61, 56, 99), 'B': (42, 177, 56, 215),
         'hut': (152, 61, 166, 99), 'stand': (106, 266, 112, 279)}


def source_mask():
    a = np.array(Image.open(ARC).convert('RGBA'))
    y0, x0, y1, x1 = SRC_ZONE
    sub = a[y0:y1 + 1, x0:x1 + 1]
    cls = np.zeros(sub.shape[:2], dtype='<U8')
    for yy in range(sub.shape[0]):
        for xx in range(sub.shape[1]):
            r, g, b, al = sub[yy, xx]
            if al == 0:
                continue
            t = (int(r), int(g), int(b))
            for name, group in (('plum', PLUM), ('caramel', CARAMEL), ('outline', OUTLINE)):
                if t in group:
                    cls[yy, xx] = name
                    break
    return (cls != '').astype(np.uint8), cls


def resample(mask, cls, th, tw):
    sh, sw = mask.shape
    out = np.zeros((th, tw), dtype='<U8')
    for ty in range(th):
        a0, a1 = ty * sh // th, max(ty * sh // th + 1, (ty + 1) * sh // th)
        for tx in range(tw):
            b0, b1 = tx * sw // tw, max(tx * sw // tw + 1, (tx + 1) * sw // tw)
            blk = cls[a0:a1, b0:b1][mask[a0:a1, b0:b1] > 0]
            if blk.size == 0:
                continue
            for pref in ('plum', 'caramel', 'outline'):
                if (blk == pref).any():
                    out[ty, tx] = pref
                    break
    return out


def dilate(out):
    o, hit = out.copy(), out != ''
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.roll(np.roll(out, dy, 0), dx, 1)
            o[(s != '') & ~hit] = 'outline'
    return o


def inpaint(tgt, y0, x0, y1, x1):
    """Remplace les pixels d'encre du generateur par la couleur voisine la plus proche."""
    zone = tgt[y0:y1 + 1, x0:x1 + 1]
    col = zone[..., :3].astype(int)
    dead = np.zeros(col.shape[:2], bool)
    for c in ERASE:
        dead |= (col == np.array(c)).all(-1)
    dead |= (col.sum(-1) < 260)                       # toutes les encre tres sombres
    good = ~dead & (zone[..., 3] > 0)
    n = int(dead.sum())
    for _ in range(2):                                # 2 passes suffisent a 15 px de haut
            # ligne d'abord, puis colonne
            for yy in range(col.shape[0]):
                cand = np.where(good[yy])[0]
                for xx in np.where(dead[yy])[0]:
                    if not len(cand):
                        continue
                    src = cand[np.argmin(abs(cand - xx))]
                    zone[yy, xx, :3] = col[yy, src]
                    col[yy, xx] = col[yy, src]; good[yy, xx] = True; dead[yy, xx] = False
            for xx in range(col.shape[1]):
                cand = np.where(good[:, xx])[0]
                for yy in np.where(dead[:, xx])[0]:
                    if not len(cand) or dead[yy, xx]:
                        continue
                    src = cand[np.argmin(abs(cand - yy))]
                    zone[yy, xx, :3] = col[src, xx]
                    col[yy, xx] = col[src, xx]; good[yy, xx] = True; dead[yy, xx] = False
    return n


def main():
    mask, cls = source_mask()
    tgt = np.array(Image.open(TGT).convert('RGBA'))
    for name, (y0, x0, y1, x1) in ZONES.items():
        th, tw = y1 - y0 + 1, x1 - x0 + 1
        er = inpaint(tgt, y0, x0 - 3, y1, x1 + 3)
        out = dilate(resample(mask, cls, th, tw))
        keep = out != ''
        zone = tgt[y0:y1 + 1, x0:x1 + 1]
        zone[..., :3][keep] = np.array([INK[p] for p in out[keep]])
        zone[..., 3][keep] = 255
        cnt = {k: int((out == k).sum()) for k in INK}
        print(f'{name}: greffe {tw}x{th} sur {er} px effaces, encres {cnt}')
    if len(sys.argv) > 3:                                  # sortie explicite possible
        Image.fromarray(tgt, 'RGBA').save(sys.argv[3])
    preview = os.path.splitext(sys.argv[3] if len(sys.argv) > 3 else TGT)[0]

    z = Image.fromarray(tgt, 'RGBA').resize((338 * 3, 267 * 3), Image.NEAREST)
    bg = Image.new('RGBA', z.size, (30, 30, 36, 255)); bg.alpha_composite(z)
    bg.save(preview + '_apercu3x.png')
    for name, (y0, x0, y1, x1) in ZONES.items():
        c = tgt[max(0, y0 - 15):y1 + 16, max(0, x0 - 22):x1 + 23]
        im = Image.fromarray(c, 'RGBA').resize((c.shape[1] * 12, c.shape[0] * 12), Image.NEAREST)
        b = Image.new('RGBA', im.size, (30, 30, 36, 255)); b.alpha_composite(im)
        b.save(f'{preview}_zoom_{name}.png')
    # zoom specifique sur la petite tete du support
    print('ok')


if __name__ == '__main__':
    main()
