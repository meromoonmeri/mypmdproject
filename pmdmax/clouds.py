"""
Les nuages Dynamax qui tournoient au-dessus de la tête.

Référence visuelle (Sword/Shield, Pokémon GO) : ce ne sont PAS une grosse
nébuleuse unique, mais quelques touffes rouges compactes, bien séparées, qui
orbitent lentement autour d'un point situé juste au-dessus du crâne. Vues de
3/4, elles décrivent une ellipse très écrasée (perspective) ; celles qui
passent DERRIÈRE la tête sont plus petites, plus sombres et masquées par le
Pokémon, celles qui passent DEVANT sont plus grosses et plus claires.

Chaque touffe est une grappe de disques (metaballs pauvres) ombrée par
`pixels.shade_volume`, ce qui donne le côté « bourgeonnant côtelé » du modèle
officiel sans jamais sortir de la palette 6 tons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from . import config as C
from . import pixels as P


@dataclass
class Puff:
    """Une touffe de nuage positionnée sur l'orbite."""
    x: float          # centre en coordonnées tuile
    y: float
    scale: float      # rayon de base en pixels
    depth: float      # -1 = tout au fond, +1 = tout devant
    seed: int

    @property
    def behind(self) -> bool:
        return self.depth < 0.0

    @property
    def tone(self) -> int:
        """Ton d'ombrage : les nuages du fond sont plus sombres."""
        if self.depth > 0.35:
            return 0
        if self.depth > -0.35:
            return 1
        return 2


def orbit_puffs(cx: float, cy: float, rx: float, ry: float, count: int,
                phase: float, base_scale: float,
                bob: float = 0.0) -> List[Puff]:
    """
    Répartit `count` touffes sur une ellipse vue en perspective.

    phase : 0..1, avance de la rotation (1 = un tour complet).
    bob   : amplitude du flottement vertical global (respiration).
    """
    puffs: List[Puff] = []
    for i in range(count):
        a = 2 * math.pi * (phase + i / count)
        # sin(a) < 0 => moitié haute de l'ellipse => derrière la tête
        depth = -math.sin(a)
        x = cx + math.cos(a) * rx
        y = cy + math.sin(a) * ry * 0.5 + math.sin(2 * math.pi * phase) * bob
        # perspective : plus gros devant
        scale = base_scale * (0.72 + 0.28 * (depth + 1.0) / 2.0 * 1.6)
        puffs.append(Puff(x=x, y=y, scale=scale, depth=depth, seed=i * 977))
    # dessin du fond vers l'avant
    puffs.sort(key=lambda p: p.depth)
    return puffs


def _puff_mask(shape: Tuple[int, int], puff: Puff, wobble: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Masque d'une touffe : un disque central + 4-6 lobes satellites,
    ce qui donne la silhouette « chou-fleur » caractéristique.
    """
    mask = np.zeros(shape, dtype=bool)
    r = puff.scale
    # noyau légèrement ovale, plus large que haut
    P.ellipse_mask(mask, puff.x, puff.y, r * 0.98, r * 0.70)

    # bourgeons sur la moitié SUPÉRIEURE seulement : c'est ce qui donne la
    # silhouette « chou-fleur » (dessus bosselé, dessous plat).
    lobes = 4 if r >= 3.0 else 3
    for k in range(lobes):
        a = math.pi * (0.12 + 0.76 * (k / max(1, lobes - 1)))  # de ~20° à ~160°
        a_j = a + 0.18 * math.sin(wobble + puff.seed * 0.013 + k)
        dist = r * (0.62 + 0.10 * math.cos(a_j * 2 + wobble))
        lx = puff.x - math.cos(a_j) * dist * 1.05
        ly = puff.y - math.sin(a_j) * dist * 0.78
        lr = r * float(rng.uniform(0.42, 0.56))
        P.disc_mask(mask, lx, ly, lr)

    # base franche : on coupe net sous la ligne de flottaison
    cut = int(round(puff.y + r * 0.72))
    if 0 <= cut < shape[0]:
        mask[cut:, :] &= False
    return mask


def render_cloud_layer(w: int, h: int, cx: float, cy: float, phase: float,
                       count: int = 3, rx: float = 9.0, ry: float = 5.0,
                       base_scale: float = 4.0, bolts: bool = True,
                       seed: int = 0, bob: float = 1.0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rend la couronne de nuages pour une frame donnée.

    Renvoie (arrière, avant) : deux grilles d'index. L'appelant doit composer
    `arrière` SOUS le Pokémon et `avant` PAR-DESSUS, afin que les nuages
    passent réellement derrière puis devant la tête pendant la rotation.
    """
    rng = np.random.default_rng(seed * 7919 + int(phase * 10000))
    back = P.new_idx(w, h)
    front = P.new_idx(w, h)
    wobble = 2 * math.pi * phase

    puffs = orbit_puffs(cx, cy, rx, ry, count, phase, base_scale, bob=bob)

    for puff in puffs:
        target = back if puff.behind else front
        sub_rng = np.random.default_rng(puff.seed + int(phase * 1000))
        mask = _puff_mask((h, w), puff, wobble, sub_rng)
        if not mask.any():
            continue
        shaded = P.shade_sphere(mask, puff.x, puff.y, puff.scale,
                                tone=puff.tone, ribs=puff.scale >= 3.0)
        P.blit_idx(target, shaded, 0, 0)

    if bolts:
        # Arc électrique occasionnel entre deux touffes voisines. Rare et
        # fin : s'il apparaît à chaque frame et en épais, la couronne se lit
        # comme un haltère au lieu de nuages séparés.
        fronts = [p for p in puffs if not p.behind]
        if len(fronts) >= 2 and rng.random() < 0.3:
            a, b = rng.choice(len(fronts), size=2, replace=False)
            pa, pb = fronts[int(a)], fronts[int(b)]
            # uniquement si les touffes sont assez proches
            if abs(pa.x - pb.x) < (pa.scale + pb.scale) * 2.6:
                P.draw_bolt(front,
                            (pa.x, pa.y - pa.scale * 0.75),
                            (pb.x, pb.y - pb.scale * 0.75),
                            rng, segments=4, amplitude=2.2,
                            core=C.IDX_RIM, halo=0)
        # étincelles orbitales
        halo = np.zeros((h, w), dtype=bool)
        P.ellipse_mask(halo, cx, cy, rx * 1.25, ry * 0.9)
        halo &= (front == 0) & (back == 0)
        P.sprinkle(front, halo, int(rng.integers(1, 4)), C.IDX_RIM, rng)

    return back, front


def cloud_bounds(cx: float, cy: float, rx: float, ry: float,
                 base_scale: float, bob: float = 1.0) -> Tuple[int, int, int, int]:
    """
    Encombrement maximal des nuages sur un tour complet, en coordonnées tuile.
    Sert à savoir de combien agrandir la tuile pour ne rien rogner.
    """
    maxr = base_scale * 1.75 + 2.0     # lobes + contour
    return (int(math.floor(cx - rx - maxr)),
            int(math.floor(cy - ry * 0.5 - maxr - bob)),
            int(math.ceil(cx + rx + maxr)),
            int(math.ceil(cy + ry * 0.5 + maxr + bob)))
