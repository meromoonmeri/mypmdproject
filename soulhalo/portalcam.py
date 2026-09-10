"""Intérieur du portail : caméra 3D pilotée par le joueur.

Pendant les questions du test, le joueur est DANS le portail. La souris
(ou le stick) n'avance pas le test : elle oriente le regard. On tourne la
tête et on voit réellement une autre portion du tunnel, et la sphère-âme
sous un autre angle.

POURQUOI UNE VRAIE CAMÉRA, ET PAS UN DÉCALAGE D'IMAGE
-----------------------------------------------------
Décaler une image toute faite donne un effet de carton qui glisse : la
sphère garde la même silhouette, on sent le trucage. Ici le tunnel est
une surface en 3D et la sphère un point en 3D. On lance un rayon par
pixel depuis la caméra ; quand la caméra tourne, la géométrie change
vraiment. La sphère passe derrière soi si on se retourne, grossit quand
elle s'approche, se décale en perspective — parce que c'est calculé, pas
mimé.

LES DEUX AXES RESTENT SÉPARÉS
-----------------------------
`phase`  le temps : boucle à l'infini, que le joueur bouge ou non ;
`cam`    le regard : libre, sans le moindre effet sur la boucle.

C'est la règle du projet. `cam` n'entre dans aucun terme périodique : il
oriente, il ne fait pas avancer. Conséquence testée : à caméra figée,
`render(0.0)` et `render(1.0)` sont identiques au bit près, et le test ne
progresse que sur la validation d'une réponse.

REPÈRE
------
La caméra est à l'origine, le tunnel est un cylindre d'axe Z. Le lacet
(yaw) tourne autour de Y, le tangage (pitch) autour de X. L'ordre est
monde = Ry(lacet) · Rx(tangage) · caméra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import pixelart as PX

TAU = 2.0 * np.pi

# Spectre du Dream Halo : bleu nuit -> cyan -> turquoise -> violet ->
# magenta -> blanc. Ce n'est pas un arc-en-ciel : pas de rouge ni de
# jaune, et la montée se fait vers le blanc.
SPECTRE = np.array([
    [0.05, 0.07, 0.22],
    [0.09, 0.26, 0.52],
    [0.13, 0.55, 0.70],
    [0.22, 0.72, 0.68],
    [0.42, 0.36, 0.78],
    [0.72, 0.34, 0.82],
    [0.92, 0.72, 0.95],
    [1.00, 1.00, 1.00],
], np.float32)


def _rampe(t):
    """Échantillonne le spectre. `t` est cyclique : la teinte boucle."""
    t = np.asarray(t, np.float32) % 1.0
    n = len(SPECTRE) - 1
    f = t * n
    i = np.clip(f.astype(np.int32), 0, n - 1)
    g = (f - i)[..., None]
    return SPECTRE[i] * (1 - g) + SPECTRE[i + 1] * g


# --------------------------------------------------------------------------
# la caméra
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """Orientation du regard, lissée.

    Le pointeur saute d'une frame à l'autre ; appliqué tel quel, le décor
    tremblerait. On interpole vers la cible à vitesse bornée.

    Les amplitudes sont bornées : le joueur regarde autour de lui, il ne
    fait pas de tonneau. `max_yaw` est en TOURS (0.16 tour ~ 58 degres de
    chaque côté).
    """
    yaw: float = 0.0            # -1 .. +1, fraction de max_yaw
    pitch: float = 0.0
    # Amplitudes VOLONTAIREMENT faibles. Mesuré à l'image : au-delà d'une
    # vingtaine de degrés, les rayons touchent la paroi tout près et la
    # bouche du tunnel sort du champ — on ne voit plus qu'un mur oblique.
    # 0.05 tour = 18 degrés de chaque côté : on regarde autour de soi tout
    # en gardant le portail et la sphère en vue.
    max_yaw: float = 0.050      # en tours
    max_pitch: float = 0.038
    smooth: float = 0.16
    fov: float = 1.15           # radians, ouverture verticale

    def update(self, mx: float, my: float):
        """Reçoit la souris ou le stick, normalisés dans [-1, +1].

        Ne renvoie aucune information de progression : bouger le regard ne
        peut pas répondre à une question.
        """
        tx = float(np.clip(mx, -1.0, 1.0))
        ty = float(np.clip(my, -1.0, 1.0))
        self.yaw += (tx - self.yaw) * self.smooth
        self.pitch += (ty - self.pitch) * self.smooth
        return self

    @property
    def rad(self):
        """(lacet, tangage) en radians."""
        return (self.yaw * self.max_yaw * TAU,
                self.pitch * self.max_pitch * TAU)

    def matrices(self):
        y, p = self.rad
        return np.cos(y), np.sin(y), np.cos(p), np.sin(p)


# --------------------------------------------------------------------------
# la sphère-âme
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SoulPath:
    """Trajectoire de l'âme dans le portail.

    Les nombres de tours sont ENTIERS : c'est ce qui garantit que la
    trajectoire se referme exactement sur elle-même à chaque boucle.
    """
    rayon: float = 0.42         # rayon de l'orbite
    aplat: float = 0.62         # orbite écrasée : on la voit en biais
    tours: int = 1              # tours d'orbite par boucle — ENTIER
    z_centre: float = 1.55      # distance moyenne devant la caméra
    z_amp: float = 0.55         # va-et-vient en profondeur
    z_tours: int = 1            # ENTIER
    taille: float = 0.34        # rayon apparent à 1 unité de distance

    def position(self, phase: float):
        a = TAU * self.tours * (phase % 1.0)
        z = self.z_centre + self.z_amp * np.cos(TAU * self.z_tours *
                                                (phase % 1.0))
        return (self.rayon * np.cos(a),
                self.rayon * self.aplat * np.sin(a),
                float(z))


def _dessine_ame(buf, cx, cy, r, intensite=1.0):
    """Pose la sphère-âme, design FIXE.

    Le motif ne dépend que du rayon normalisé : cœur blanc, manteau bleu
    pâle, halo diffus. Il est donc rigoureusement identique d'une frame à
    l'autre — seule la taille change avec la distance. C'est l'exigence
    de constance du design de l'âme.
    """
    h, w = buf.shape[:2]
    r = max(1.5, float(r))
    x0, x1 = int(max(0, cx - r * 2.6)), int(min(w, cx + r * 2.6 + 1))
    y0, y1 = int(max(0, cy - r * 2.6)), int(min(h, cy + r * 2.6 + 1))
    if x1 <= x0 or y1 <= y0:
        return buf

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy) / r

    coeur = np.clip(1.0 - d / 0.55, 0, 1) ** 0.6
    manteau = np.clip(1.0 - (d - 0.35) / 0.75, 0, 1) ** 1.4
    halo = np.clip(1.0 - (d - 0.6) / 2.0, 0, 1) ** 2.2

    col = (np.array([1.0, 1.0, 1.0], np.float32) * coeur[..., None] +
           np.array([0.62, 0.85, 1.00], np.float32) * manteau[..., None] * .85 +
           np.array([0.35, 0.55, 0.95], np.float32) * halo[..., None] * .55)

    a = np.clip(coeur + manteau * 0.8 + halo * 0.45, 0, 1) * intensite
    zone = buf[y0:y1, x0:x1]
    # additif : l'âme est de la lumière, elle ne masque pas le tunnel
    buf[y0:y1, x0:x1] = np.clip(zone + col * a[..., None], 0, 1)
    return buf


# --------------------------------------------------------------------------
# la vue
# --------------------------------------------------------------------------
class PortalView:
    """Vue subjective de l'intérieur du portail.

    `render` est PUR : il ne modifie ni la caméra ni l'état du test. Toute
    mutation passe par `Camera.update`.
    """

    def __init__(self, w=320, h=240, path: SoulPath | None = None,
                 rayon=1.0, defile=2, spectre_tours=1, pixel=True):
        self.w, self.h = int(w), int(h)
        self.path = path or SoulPath()
        self.rayon = float(rayon)
        self.defile = int(defile)          # périodes de texture par boucle
        self.spectre_tours = int(spectre_tours)
        self.pixel = bool(pixel)

        # grille de rayons, calculée une fois
        ndc_x = (np.arange(self.w, dtype=np.float32) + 0.5) / self.w * 2 - 1
        ndc_y = (np.arange(self.h, dtype=np.float32) + 0.5) / self.h * 2 - 1
        self.gx, self.gy = np.meshgrid(ndc_x, ndc_y)

    # ------------------------------------------------------------------
    def _rayons(self, cam: Camera):
        """Direction du rayon de chaque pixel, dans le repère du monde."""
        t = np.tan(cam.fov * 0.5)
        aspect = self.w / self.h
        dx = self.gx * t * aspect
        dy = self.gy * t
        dz = np.ones_like(dx)

        cy, sy, cp, sp = cam.matrices()
        # Rx(tangage)
        dy1 = dy * cp - dz * sp
        dz1 = dy * sp + dz * cp
        # Ry(lacet)
        dx2 = dx * cy + dz1 * sy
        dz2 = -dx * sy + dz1 * cy
        return dx2, dy1, dz2

    def _monde_vers_camera(self, p, cam: Camera):
        cy, sy, cp, sp = cam.matrices()
        xw, yw, zw = p
        x1 = xw * cy - zw * sy
        z1 = xw * sy + zw * cy
        y2 = yw * cp + z1 * sp
        z2 = -yw * sp + z1 * cp
        return x1, y2, z2

    def _projette(self, p, cam: Camera):
        """Point 3D -> pixel. Renvoie None si le point est derrière."""
        xc, yc, zc = self._monde_vers_camera(p, cam)
        if zc <= 0.05:
            return None
        t = np.tan(cam.fov * 0.5)
        aspect = self.w / self.h
        sx = (xc / (zc * t * aspect) * 0.5 + 0.5) * self.w
        sy = (yc / (zc * t) * 0.5 + 0.5) * self.h
        return float(sx), float(sy), float(zc)

    # ------------------------------------------------------------------
    def render(self, phase: float, cam: Camera | None = None):
        """Une frame. `phase` dans [0, 1[ ; `cam` n'affecte pas la boucle."""
        cam = cam or Camera()
        ph = float(phase) % 1.0
        dx, dy, dz = self._rayons(cam)

        # Intersection du rayon avec le cylindre (la caméra est sur l'axe,
        # donc la solution est directe). `rad` est la distance parcourue
        # avant de toucher la paroi.
        radial = np.maximum(np.hypot(dx, dy), 1e-6)
        t_hit = self.rayon / radial
        z_hit = t_hit * dz                       # peut être négatif :
        theta = np.arctan2(dy, dx)               # c'est le tunnel derrière

        # coordonnées sur la paroi. Le défilement est un nombre ENTIER de
        # périodes par boucle : le raccord est exact.
        u = theta / TAU
        v = z_hit * 0.55 - ph * self.defile

        # Nervures du portail. Les ANNEAUX (terme en v seul) dominent :
        # ce sont eux qui donnent la lecture « tunnel » quand ils défilent
        # vers soi. Les deux autres harmoniques cassent la régularité.
        # Tout est périodique en u et v, donc tout boucle.
        anneaux = np.sin(TAU * (v * 2 + 0.10 * np.sin(TAU * u * 3)))
        anneaux = np.sign(anneaux) * np.abs(anneaux) ** 0.55  # crêtes nettes
        motif = (0.62 * anneaux
                 + 0.24 * np.sin(TAU * (u * 8 + v * 2))
                 + 0.14 * np.sin(TAU * (u * 16 - v * 3)))

        # la teinte défile dans le spectre le long du tunnel
        teinte = (v * 0.5 + u * 0.25 + ph * self.spectre_tours)
        col = _rampe(teinte)

        # Profondeur : loin = sombre. C'est ce dégradé qui creuse le tunnel.
        # Le plancher évite que la paroi proche tombe au noir quand on
        # tourne la tête — sans lui, regarder de côté donnait un mur noir.
        prof = 0.30 + 0.70 / (1.0 + np.abs(z_hit) * 0.42)
        lum = np.clip(0.30 + 0.85 * motif, 0, 1) * prof

        buf = col * lum[..., None] * 1.35

        # Cœur lumineux au bout du tunnel. Il est ancré sur l'axe du TUNNEL
        # (direction monde +Z), pas sur le centre de l'écran : quand le
        # joueur tourne la tête, la bouche du portail se décale
        # latéralement, comme elle le doit.
        # Exposants élevés : mesuré à l'image, un cœur large noyait la
        # sphère-âme, qui est le sujet. Il doit rester un point de fuite,
        # pas une source qui mange le cadre.
        cz = np.clip(dz / np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz),
                                     1e-6), 0, 1)
        axe = cz ** 150
        buf = buf + axe[..., None] * np.array([0.80, 0.92, 1.00], np.float32)
        halo = cz ** 34
        buf = buf + halo[..., None] * np.array([0.14, 0.24, 0.40], np.float32)

        # ---- la sphère-âme et sa traînée ----
        # La traînée reprend des positions ANTÉRIEURES de la même
        # trajectoire : elle boucle donc avec elle, sans état conservé.
        for k in range(6, 0, -1):
            pk = self.path.position(ph - k * 0.014)
            pr = self._projette(pk, cam)
            if pr is None:
                continue
            sx, sy, zc = pr
            r = self.path.taille * self.h / (zc * 2.4)
            _dessine_ame(buf, sx, sy, r * (0.30 + 0.09 * (6 - k)),
                         intensite=0.10 + 0.045 * (6 - k))

        pr = self._projette(self.path.position(ph), cam)
        if pr is not None:
            sx, sy, zc = pr
            r = self.path.taille * self.h / (zc * 2.4)
            # respiration : nombre ENTIER de cycles par boucle
            pulse = 1.0 + 0.07 * np.cos(TAU * 2 * ph)
            _dessine_ame(buf, sx, sy, r * pulse, intensite=1.0)

        buf = np.clip(buf, 0.0, 1.0)
        if self.pixel:
            # tramage sur la grille logique, AVANT tout agrandissement
            buf = PX.dither(buf, levels=14, strength=0.9)
        return buf

    def frame_image(self, phase, cam=None, scale=3):
        a = self.render(phase, cam)
        from .dxui import upscale
        return Image.fromarray((upscale(a, scale) * 255 + 0.5)
                               .astype(np.uint8))

    # ------------------------------------------------------------------
    def visible(self, phase, cam=None):
        """La sphère est-elle dans le champ ? Sert aux tests et au debug."""
        cam = cam or Camera()
        pr = self._projette(self.path.position(phase), cam)
        if pr is None:
            return False
        sx, sy, _ = pr
        return 0 <= sx < self.w and 0 <= sy < self.h
