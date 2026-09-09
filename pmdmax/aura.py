"""
L'aura de fluide rouge ondulant, **adaptée à la silhouette de chaque Pokémon**.

Principe : l'aura n'est pas un décor plaqué, elle est **dérivée du sprite
lui-même**. Pour chaque frame et chaque direction :

  1. on prend le masque opaque de la tuile (la silhouette réelle) ;
  2. on calcule la **distance euclidienne** de chaque pixel extérieur à cette
     silhouette ;
  3. on module la **portée** de l'aura par un champ d'ondes qui remonte le
     long du corps — ce sont les langues de fluide ;
  4. on colore par bandes selon la distance relative : liseré rose collé au
     contour, puis cramoisi, puis bordeaux **tramé** qui se dissout.

Comme tout part du masque, un Ectoplasma rond, un Dracaufeu avec ses ailes ou
les oreilles de Pikachu produisent automatiquement une aura qui épouse leur
contour, sans réglage manuel.

L'animation est **cyclique** : toutes les ondes sont des sinusoïdes de
fréquence entière en `phase`, donc la frame N se raccorde exactement à la
frame 0 — pas de saut dans la boucle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import config as C
from . import pixels as P


@dataclass
class AuraParams:
    """Réglages de l'aura."""
    reach: float = 9.0          # portée max en pixels (avant modulation)
    strength: float = 1.0       # 0 = pas d'aura, 1 = pleine
    tongues: float = 1.0        # amplitude des langues de fluide
    rise: float = 1.0           # vitesse de remontée des ondulations
    up_bias: float = 1.45       # allongement vers le haut (le feu monte)
    dither: bool = True         # tramage du bord externe
    sparks: bool = True         # gouttelettes détachées
    seed: int = 0

    def scaled(self, factor: float) -> "AuraParams":
        """Adapte la portée à un sprite agrandi."""
        out = AuraParams(**self.__dict__)
        out.reach = self.reach * max(1.0, factor)
        return out


# --------------------------------------------------------------------------
# Distance à la silhouette
# --------------------------------------------------------------------------

def edt_outside(mask: np.ndarray, max_r: float) -> np.ndarray:
    """
    Distance euclidienne exacte de chaque pixel EXTÉRIEUR au masque, plafonnée
    à `max_r`. Les pixels du masque valent 0, les pixels au-delà valent `inf`.

    Implémentation : minimum sur une fenêtre d'offsets. `max_r` étant petit
    (quelques pixels), c'est à la fois exact et rapide — et surtout sans
    dépendance à scipy.
    """
    R = int(math.ceil(max_r))
    h, w = mask.shape
    best = np.full((h, w), np.inf, dtype=np.float32)
    best[mask] = 0.0

    pad = np.zeros((h + 2 * R, w + 2 * R), dtype=bool)
    pad[R:R + h, R:R + w] = mask

    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            d2 = dx * dx + dy * dy
            if d2 == 0 or d2 > max_r * max_r:
                continue
            d = math.sqrt(d2)
            # pixels dont le voisin (dx, dy) appartient au masque
            shifted = pad[R - dy:R - dy + h, R - dx:R - dx + w]
            np.minimum(best, np.where(shifted, d, np.inf), out=best)

    return best


def _outward_normal(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direction approximative « vers l'extérieur » en chaque pixel, obtenue par
    gradient du masque lissé. Sert à faire onduler l'aura le long du contour
    plutôt que dans un repère global.
    """
    m = mask.astype(np.float32)
    # petit flou séparable 1-2-1 (reste entier, pas de dépendance)
    for _ in range(2):
        m = (np.pad(m, ((0, 0), (1, 0)))[:, :-1]
             + 2 * m
             + np.pad(m, ((0, 0), (0, 1)))[:, 1:]) / 4.0
        m = (np.pad(m, ((1, 0), (0, 0)))[:-1, :]
             + 2 * m
             + np.pad(m, ((0, 1), (0, 0)))[1:, :]) / 4.0
    gy, gx = np.gradient(m)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-6
    return -gx / norm, -gy / norm


# --------------------------------------------------------------------------
# Champ d'ondulation
# --------------------------------------------------------------------------

def _wave_field(h: int, w: int, phase: float, p: AuraParams,
                cx: float, cy: float) -> np.ndarray:
    """
    Champ scalaire dans [0, 1] qui module la portée de l'aura.

    Il est construit dans le repère **angulaire** de la silhouette : chaque
    pixel est repéré par l'angle θ vu depuis le centre de masse. Les langues
    de fluide sont alors des lobes régulièrement répartis autour du contour
    (et non des vagues en coordonnées écran), ce qui les fait « lécher » la
    forme quelle qu'elle soit.

    Les fréquences sont **entières en θ et en phase** : le motif tourne et
    retrouve exactement son état initial après un tour — la boucle est donc
    parfaitement continue.
    """
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    two_pi = 2.0 * math.pi
    s = p.seed * 0.9

    dx = xs - cx
    dy = ys - cy
    theta = np.arctan2(dy, dx)                       # -pi..pi
    radius = np.sqrt(dx * dx + dy * dy)

    # Langues principales : lobes répartis autour du contour, qui tournent
    # lentement. `nlobes` entier => continuité en theta.
    # Peu de lobes = langues larges et bien séparées (façon flammes) plutôt
    # qu'une frange de petites dents.
    nlobes = 5.0
    t1 = np.sin(nlobes * theta - two_pi * p.rise * phase + s)
    # Harmonique : casse la régularité pour un rendu organique
    t2 = np.sin(9.0 * theta + two_pi * 2.0 * p.rise * phase + s * 1.6)
    # Ondulation radiale : le fluide « respire » en s'éloignant
    t3 = np.sin(radius * 0.85 - two_pi * 2.0 * p.rise * phase + s * 2.1)

    field = 0.70 * t1 + 0.19 * t2 + 0.11 * t3
    field = (field + 1.0) * 0.5                      # -> [0, 1]

    # Contraste : on creuse les creux pour séparer nettement les langues,
    # sinon l'aura redevient un halo uniforme. Seuil haut = grandes langues
    # espacées, avec du vide entre elles.
    field = np.clip((field - 0.42) / 0.50, 0.0, 1.0)
    field = np.power(field, 0.72)                    # pointes plus effilées

    # Le feu monte : les langues portent plus haut au-dessus du centre.
    vert = np.clip((cy - ys) / max(1.0, cy), 0.0, 1.0)
    return np.clip(field * (1.0 + (p.up_bias - 1.0) * vert), 0.0, p.up_bias)


# --------------------------------------------------------------------------
# Rendu
# --------------------------------------------------------------------------

def render_aura(mask: np.ndarray, phase: float,
                p: Optional[AuraParams] = None) -> np.ndarray:
    """
    Rend l'aura autour d'une silhouette, pour une phase donnée (0..1).

    `mask` : masque booléen HxW de la silhouette (pixels opaques du sprite).
    Renvoie une grille d'index (0 = vide), à composer **sous** le sprite.
    """
    p = p or AuraParams()
    h, w = mask.shape
    out = P.new_idx(w, h)
    if p.strength <= 0.0 or not mask.any():
        return out

    reach_max = p.reach * p.up_bias + 2.0
    dist = edt_outside(mask, reach_max)

    ys, xs = np.where(mask)
    cx, cy = float(xs.mean()), float(ys.mean())

    field = _wave_field(h, w, phase, p, cx, cy)

    outside = (~mask) & np.isfinite(dist)

    # --- 1. liseré : une gaine PLEINE collée au contour ------------------
    # Épaisseur constante, indépendante des ondulations : c'est ce qui donne
    # le trait rose net de la référence, et ce qui rend la silhouette lisible.
    # Gaine fine et constante (1 à 2 px) : elle souligne le contour sans
    # l'épaissir. C'est le trait rose net de la référence.
    rim_w = 1.0 if p.reach < 7.0 else 2.0
    rim = outside & (dist <= rim_w)
    out[rim] = C.IDX_RIM

    glow_w = rim_w + 1.0
    glow = outside & (dist > rim_w) & (dist <= glow_w)
    out[glow] = C.IDX_BRIGHT

    # --- 2. langues de fluide, au-delà de la gaine -----------------------
    # Leur portée dépend du champ angulaire : là où le lobe est haut, la
    # langue s'étire loin ; ailleurs, elle s'arrête juste après la gaine.
    span = max(1.0, p.reach * p.strength - glow_w)
    reach = glow_w + span * p.tongues * field

    tongue = outside & (dist > glow_w) & (dist <= reach)
    # position dans la langue : 0 = base (près du corps), 1 = pointe
    u = np.zeros((h, w), dtype=np.float32)
    denom = np.maximum(reach - glow_w, 1e-6)
    np.divide(dist - glow_w, denom, out=u, where=tongue)

    out[tongue & (u < 0.34)] = C.IDX_BRIGHT
    out[tongue & (u >= 0.34) & (u < 0.68)] = C.IDX_MID
    out[tongue & (u >= 0.68)] = C.IDX_DARK

    # --- 3. tramage : la pointe des langues se dissout -------------------
    if p.dither:
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        checker = ((xx + yy) % 2) == 0
        sparse = ((xx * 2 + yy) % 4) == 0
        out[tongue & (u >= 0.55) & (u < 0.80) & ~checker] = C.IDX_EMPTY
        out[tongue & (u >= 0.80) & ~sparse] = C.IDX_EMPTY

    # --- 4. crêtes lumineuses sur les langues les plus hautes ------------
    out[rim & (field > 0.86)] = C.IDX_CORE

    # --- 5. gouttelettes détachées ---------------------------------------
    if p.sparks:
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        halo = outside & (dist > reach) & (dist <= reach + 2.0) & (field > 0.90)
        # le décalage doit être PÉRIODIQUE en phase, sinon la frame de fin
        # ne se raccorde pas à la frame de début (`% 8`).
        step = int(phase * 8) % 8
        drops = halo & (((xx * 3 + yy * 5 + step) % 13) == 0)
        out[drops] = C.IDX_RIM

    return out


def render_aura_rgba(mask: np.ndarray, phase: float,
                     p: Optional[AuraParams] = None) -> np.ndarray:
    """Idem `render_aura`, mais directement en RGBA."""
    return P.to_rgba(render_aura(mask, phase, p))


def aura_margin(p: AuraParams) -> int:
    """Débordement maximal de l'aura, pour dimensionner les tuiles."""
    return int(math.ceil(p.reach * p.up_bias * max(1.0, p.strength) + 2.0))


def silhouette(tile_rgba: np.ndarray) -> np.ndarray:
    """Masque booléen des pixels opaques d'une tuile RGBA."""
    return tile_rgba[:, :, 3] > 0
