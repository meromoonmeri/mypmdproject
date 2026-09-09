"""
Portail : voyage à l'intérieur d'un tunnel d'énergie.

Le halo circulaire donnait une rotation vue de face. Pour donner la
sensation de VOYAGER DANS UN PORTAIL, il faut que la matière *fuie vers
l'extérieur* et grossisse en s'approchant, comme un décor qui défile de
part et d'autre de la caméra.

MÉTHODE — zoom infini en coordonnées log-polaires
-------------------------------------------------
En espace logarithmique, un zoom devient une simple translation :

    z = log2(r) - vitesse * phase

Une même écaille du tunnel garde `z` constant, donc son rayon écran vaut
`r = 2^(z + vitesse*phase)` : elle grossit exponentiellement et sort du
cadre. C'est exactement le mouvement d'un objet qu'on dépasse.

BOUCLAGE — `vitesse` est un ENTIER d'octaves par boucle. Après une boucle,
`z` a glissé d'un nombre entier de tuiles : l'image est rigoureusement
identique. Pas d'approximation.

COUTURE — tuiler le log-rayon met bord à bord le bord externe et le bord
interne de la plaque, ce qui produirait un anneau visible à chaque octave.
La tuile est donc lue en MIROIR (aller-retour) : `f=0` et `f=1` retombent
sur le même rayon de plaque, la continuité est automatique.

Un fondu entre deux copies décalées supprimerait aussi la couture, mais il
mélange en permanence deux bandes de couleurs différentes : le tunnel vire
alors au violet uniforme. Le miroir garde un échantillon net, donc la
saturation d'origine de la plaque.

PARALLAXE — plusieurs nappes traversent le tunnel à des vitesses
différentes (entières). Les nappes rapides se lisent comme des traînées
proches, les lentes comme le fond du couloir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import field as F

TAU = F.TAU


@dataclass(frozen=True)
class Shell:
    """Une nappe du tunnel."""
    speed: int             # octaves traversées par boucle — ENTIER
    spin: int = 0          # tours de rotation par boucle — ENTIER
    alpha: float = 1.0
    zoff: float = 0.0      # décalage de profondeur (décorrèle les nappes)
    streak: int = 1        # échantillons de filé radial (1 = net)
    streak_span: float = 0.16   # longueur du filé, en octaves


@dataclass(frozen=True)
class TunnelParams:
    shells: Sequence[Shell] = field(default_factory=lambda: [
        # fond du couloir : lent, il installe la profondeur
        Shell(speed=1, spin=+1, alpha=1.00, zoff=0.00, streak=1),
        # nappe médiane
        Shell(speed=2, spin=-1, alpha=0.28, zoff=0.33, streak=3, streak_span=0.10),
        # traînées proches : rapides et filées, elles donnent la vitesse
        Shell(speed=4, spin=+2, alpha=0.20, zoff=0.66, streak=6, streak_span=0.22),
    ])
    tile_octaves: float = 4.5   # octaves de rayon couvertes par une tuile
    rad_lo: float = 0.18        # début de la zone colorée lue dans la plaque
    rad_hi: float = 0.98        # fin de la zone colorée
    core_glow: float = 0.55     # lumière au bout du tunnel
    core_size: float = 0.085
    core_pulse: int = 1         # cycles de pulsation par boucle — ENTIER
    tint_shift: int = 0         # cycles de teinte par boucle — ENTIER (0 = off)
    edge_fade: float = 1.55     # extinction vers les coins
    center_fade: float = 0.035  # masque la singularité au centre exact
    bloom: float = 0.16
    exposure: float = 0.95
    saturation: float = 1.45
    grain: float = 0.018
    soul: bool = True           # la sphère-âme, au centre, qui traverse
    soul_radius: float = 0.055
    soul_bob: float = 0.010     # léger flottement (1 cycle par boucle)


class TunnelRenderer:
    """Rend le voyage dans le portail. La bande polaire et les grilles sont
    mises en cache : chaque frame se réduit à des lectures de tableaux."""

    def __init__(self, plate: str, w: int = 960, h: int = 540,
                 params: TunnelParams | None = None,
                 n_theta: int = 1024, n_rad: int = 512):
        self.w, self.h = int(w), int(h)
        self.p = params or TunnelParams()
        self.strip = F.polar_strip(plate, n_theta=n_theta, n_rad=n_rad)

        rn, th = F.polar(self.h, self.w)
        # normalisation sur la DIAGONALE : le tunnel doit couvrir les coins,
        # sinon le cadre trahit une image plate posée sur du noir.
        diag = float(np.hypot(self.h, self.w)) / min(self.h, self.w)
        self.rd = np.maximum(rn / np.float32(diag), np.float32(1e-3))
        self.rn, self.th = rn, th
        # log-rayon exprimé EN TUILES. Sans cette division, une tuile vaut
        # une octave et la plaque se répète ~10 fois à l'écran : toutes les
        # teintes se moyennent et le tunnel vire au violet uniforme.
        self.logr = (np.log2(self.rd) /
                     np.float32(self.p.tile_octaves)).astype(np.float32)

        g = np.random.default_rng(20200306).standard_normal((self.h, self.w))
        self.grain = F.blur(g.astype(np.float32), 1.0)
        self.grain /= max(1e-6, float(np.abs(self.grain).max()))

    # -- échantillonnage ---------------------------------------------------
    def _tile(self, z: np.ndarray, spin: float) -> np.ndarray:
        """Lit la plaque pour une profondeur `z`, sans couture.

        Deux copies décalées d'une demi-tuile, pondérées sin²/cos². La
        couture de chaque copie tombe là où son poids s'annule.
        """
        n_rad, n_theta = self.strip.shape[:2]
        ti = ((self.th / np.float32(TAU) + np.float32(spin)) * n_theta)
        ti = np.mod(ti.astype(np.int64), n_theta).astype(np.int32)

        f = np.mod(z, 1.0).astype(np.float32)
        tri = 1.0 - np.abs(2.0 * f - 1.0)      # miroir : continu en f=0 et f=1
        # la tuile balaie la zone colorée de la plaque, en évitant son bord
        # noir (qui ferait un anneau sombre à chaque répétition)
        rp = np.float32(self.p.rad_lo) + tri * np.float32(self.p.rad_hi - self.p.rad_lo)
        ri = np.clip(rp * (n_rad - 1), 0, n_rad - 1).astype(np.int32)
        return self.strip[ri, ti]

    def _shell(self, s: Shell, phase: float) -> np.ndarray:
        """Une nappe, avec son filé radial éventuel."""
        base = self.logr - np.float32(s.speed) * np.float32(phase) + np.float32(s.zoff)
        spin = s.spin * phase
        if s.streak <= 1:
            return self._tile(base, spin)
        # Filé : on moyenne plusieurs profondeurs voisines. Le filé suit
        # donc la direction du mouvement (radiale), comme un flou de vitesse.
        acc = np.zeros((self.h, self.w, 3), np.float32)
        for k in range(s.streak):
            d = (k / (s.streak - 1) - 0.5) * s.streak_span
            acc += self._tile(base + np.float32(d), spin)
        return acc / float(s.streak)

    # -- rendu -------------------------------------------------------------
    def render(self, phase: float) -> np.ndarray:
        p = self.p
        phase = float(phase) % 1.0
        rd = self.rd

        # Fusion en MOYENNE PONDEREE, en laissant une nappe dominer.
        # L'ecran (1-(1-a)(1-b)) additionne les luminosites et delave tout
        # vers le blanc ; la moyenne simple ramene les trois nappes vers une
        # teinte unique. On garde donc la moyenne, mais avec des poids tres
        # contrastes : la nappe de fond porte la couleur, les nappes rapides
        # ne font qu'ajouter des trainees.
        acc = np.zeros((self.h, self.w, 3), np.float32)
        wsum = 0.0
        for s in p.shells:
            acc += self._shell(s, phase) * np.float32(s.alpha)
            wsum += s.alpha
        rgb = acc / max(1e-6, wsum)

        if p.tint_shift:
            rgb = F.hue_rotate(rgb, p.tint_shift * phase)

        # bouche du tunnel : la lumière vers laquelle on se dirige
        if p.core_glow > 0:
            pulse = 0.5 + 0.5 * float(np.cos(TAU * p.core_pulse * phase))
            size = p.core_size * (1.0 + 0.10 * pulse)
            glow = np.exp(-(rd / np.float32(size)) ** 2)
            rgb += glow[..., None] * np.array([1.0, 0.985, 0.95], np.float32) * \
                np.float32(p.core_glow * (0.80 + 0.20 * pulse))

        # le centre exact est une singularité du log : on l'éteint
        rgb *= F.smoothstep(p.center_fade * 0.25, p.center_fade, rd)[..., None]

        # la sphère-âme traverse le portail
        if p.soul:
            bob = p.soul_bob * float(np.sin(TAU * phase))
            yy, xx = np.indices((self.h, self.w)).astype(np.float32)
            half = min(self.h, self.w) / 2.0
            dx = (xx - (self.w - 1) / 2.0) / half
            dy = (yy - (self.h - 1) / 2.0) / half - np.float32(bob)
            d = np.hypot(dx, dy) / np.float32(p.soul_radius)
            rgb += np.exp(-(d ** 2) * 1.15)[..., None] * \
                np.array([0.72, 0.93, 1.0], np.float32) * np.float32(0.55)
            rgb += np.exp(-(d ** 2) * 3.0)[..., None] * \
                np.array([1.0, 1.0, 0.99], np.float32) * np.float32(1.25)

        if p.bloom > 0:
            hot = np.clip(rgb.mean(2) - 0.60, 0.0, None)
            rgb += F.blur(hot, 15.0)[..., None] * np.float32(p.bloom) * \
                np.array([1.0, 0.96, 0.92], np.float32)

        rgb *= F.smoothstep(p.edge_fade, p.edge_fade * 0.45, rd)[..., None]
        if p.grain:
            rgb *= (1.0 + self.grain * np.float32(p.grain))[..., None]
        rgb *= np.float32(p.exposure)
        rgb = rgb / (1.0 + rgb * 0.48)          # tone map doux
        if p.saturation != 1.0:                 # le tone map désature
            lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
            rgb = lum + (rgb - lum) * np.float32(p.saturation)
        return np.clip(rgb, 0.0, 1.0)

    def frame(self, phase: float):
        from PIL import Image
        return Image.fromarray((self.render(phase) * 255.0 + 0.5).astype(np.uint8))
