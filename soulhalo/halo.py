"""
Parallaxe circulaire : halo multicolore + sphère-âme qui voyage.

Principe de la parallaxe
------------------------
Une parallaxe classique fait glisser des plans à des vitesses différentes.
Ici les plans sont des **couronnes concentriques** : chacune tourne à sa
propre vitesse angulaire, et les vitesses alternent de signe. L'œil lit
cette différence de vitesse comme de la profondeur — on regarde dans un
tunnel, pas sur une image plate.

Chaque couche porte :
  - une vitesse (en tours par boucle, entière => raccord exact),
  - un rayon et une épaisseur,
  - une échelle radiale d'échantillonnage de la plaque peinte,
  - une opacité.

Bouclage
--------
Toutes les vitesses sont des entiers en tours/boucle et tous les termes de
respiration sont des cos(2*pi*k*phase). La frame N se raccorde donc à la
frame 0 au pixel près (vérifié par les tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from PIL import Image

from . import field as F

TAU = F.TAU


# --------------------------------------------------------------------------
# Paramètres
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer:
    """Une couronne du halo."""
    radius: float          # rayon normalisé du centre de l'anneau
    width: float           # épaisseur (profil en cloche)
    speed: int             # tours par boucle — ENTIER (signe = sens)
    rad_scale: float = 1.0  # zoom radial dans la plaque peinte
    alpha: float = 1.0     # opacité
    breathe: float = 0.0   # respiration du rayon (en unités de rayon)
    lobes: int = 0         # ondulation angulaire (0 = anneau lisse)
    lobe_amp: float = 0.0  # amplitude de l'ondulation
    hue_offset: float = 0.0  # décalage de teinte propre à la couronne (tours)
    rad_offset: float = 0.0  # décalage de lecture radiale dans la plaque


@dataclass(frozen=True)
class SoulParams:
    """La sphère : l'âme du joueur qui voyage."""
    orbit_rx: float = 0.46      # demi-axe horizontal de l'orbite
    orbit_ry: float = 0.30      # demi-axe vertical (ellipse = perspective)
    orbit_turns: int = 1        # tours d'orbite par boucle (entier)
    phase0: float = -0.25       # position de départ (0 = droite, -0.25 = haut)
    radius: float = 0.085       # rayon du noyau
    glow: float = 3.4           # étendue du halo autour du noyau
    trail: int = 26             # nombre d'échantillons de traînée
    trail_span: float = 0.19    # longueur de la traînée (en tours)
    core: tuple = (1.0, 0.985, 0.94)
    tint: tuple = (0.62, 0.90, 1.0)
    spark_count: int = 7
    spark_turns: int = 2        # entier => les étincelles bouclent


def default_layers() -> list[Layer]:
    """
    Empilement par défaut : 6 couronnes, vitesses alternées et décroissantes
    du bord vers le centre. Le bord tourne lentement (lointain), les anneaux
    intérieurs plus vite (proches) — c'est ce gradient qui fait la profondeur.
    """
    return [
        Layer(radius=1.30, width=0.52, speed=+1, rad_scale=0.62, alpha=0.55,
              breathe=0.020, lobes=3, lobe_amp=0.030),
        Layer(radius=1.00, width=0.30, speed=-2, rad_scale=0.78, alpha=0.80,
              breathe=0.016, lobes=5, lobe_amp=0.026),
        Layer(radius=0.78, width=0.20, speed=+3, rad_scale=0.90, alpha=0.95,
              breathe=0.013, lobes=7, lobe_amp=0.020),
        Layer(radius=0.60, width=0.15, speed=-4, rad_scale=1.00, alpha=1.00,
              breathe=0.011, lobes=9, lobe_amp=0.015),
        Layer(radius=0.44, width=0.12, speed=+6, rad_scale=1.12, alpha=0.92,
              breathe=0.009, lobes=11, lobe_amp=0.011),
        Layer(radius=0.28, width=0.11, speed=-8, rad_scale=1.28, alpha=0.72,
              breathe=0.007, lobes=13, lobe_amp=0.008),
    ]


@dataclass(frozen=True)
class HaloParams:
    layers: Sequence[Layer] = field(default_factory=default_layers)
    soul: SoulParams = field(default_factory=SoulParams)
    core_dark: float = 0.30     # rayon du puits sombre central
    vignette: float = 1.34      # début de l'extinction vers le noir
    rays: int = 12              # rayons crépusculaires
    ray_turns: int = 1          # entier
    ray_strength: float = 0.16
    grain: float = 0.030        # grain de papier
    bloom: float = 0.30         # intensité du voile lumineux
    exposure: float = 1.05
    saturation: float = 1.85    # ravive les teintes noyées par l'addition
    plate_offset: float = 0.0   # évite le cœur sombre d'une plaque en anneaux
    rainbow: int = 0            # cycles de teinte par boucle — ENTIER (0 = éteint)
    rainbow_spread: float = 1.0  # étalement du spectre entre les couronnes
    spikes: int = 8             # branches d'étoile autour de l'âme
    spike_len: float = 0.72
    spike_strength: float = 0.55


# --------------------------------------------------------------------------
# Rendu
# --------------------------------------------------------------------------
class HaloRenderer:
    """Garde en cache la bande polaire et les grilles : le rendu par frame
    se réduit à des lectures de tableaux."""

    def __init__(self, plate: str, w: int = 960, h: int = 540,
                 params: HaloParams | None = None,
                 n_theta: int = 1024, n_rad: int = 384):
        self.w, self.h = int(w), int(h)
        self.p = params or HaloParams()
        self.strip = F.polar_strip(plate, n_theta=n_theta, n_rad=n_rad)
        self.rn, self.th = F.polar(self.h, self.w)
        # grain fixe (ne scintille pas d'une frame à l'autre)
        g = np.random.default_rng(20200306).standard_normal((self.h, self.w))
        self.grain = F.blur(g.astype(np.float32), 1.0).astype(np.float32)
        self.grain /= max(1e-6, float(np.abs(self.grain).max()))

    # -- l'âme -------------------------------------------------------------
    def soul_xy(self, phase: float, s: SoulParams | None = None):
        """Position de la sphère en coordonnées normalisées."""
        s = s or self.p.soul
        a = TAU * (s.orbit_turns * phase + s.phase0)
        return float(s.orbit_rx * np.cos(a)), float(s.orbit_ry * np.sin(a))

    def _mask_from_xy(self, x: float, y: float, rad: float):
        """Distance normalisée à un point donné (en unités de rayon d'écran)."""
        half = min(self.h, self.w) / 2.0
        cx = (self.w - 1) / 2.0 + x * half
        cy = (self.h - 1) / 2.0 + y * half
        yy, xx = np.indices((self.h, self.w)).astype(np.float32)
        d = np.hypot(xx - np.float32(cx), yy - np.float32(cy)) / np.float32(half)
        return (d / np.float32(max(1e-6, rad))).astype(np.float32)

    # -- frame -------------------------------------------------------------
    def render(self, phase: float) -> np.ndarray:
        """Rend une frame. `phase` dans [0,1) ; 1.0 == 0.0 exactement."""
        p = self.p
        phase = float(phase) % 1.0
        rn, th = self.rn, self.th
        rgb = np.zeros((self.h, self.w, 3), np.float32)

        # --- couronnes en parallaxe --------------------------------------
        # Accumulation en MOYENNE PONDEREE, pas en somme : additionner six
        # couches colorées les fait converger vers le blanc et tue les
        # teintes. On garde donc la couleur et on ne somme que les poids.
        acc = np.zeros((self.h, self.w, 3), np.float32)
        wsum = np.zeros((self.h, self.w), np.float32)
        nlayers = max(1, len(p.layers))
        for li, L in enumerate(p.layers):
            rad = L.radius
            if L.breathe:
                rad += L.breathe * float(np.cos(TAU * phase))
            rr = rn
            if L.lobes and L.lobe_amp:
                # ondulation qui tourne avec la couche
                rr = rn - L.lobe_amp * np.cos(L.lobes * th - TAU * L.speed * phase)
            m = F.ring(rr, rad, L.width) * np.float32(L.alpha)
            col = F.sample_strip(self.strip, rn, th,
                                 rot=L.speed * phase, rad_scale=L.rad_scale,
                                 rad_offset=L.rad_offset + p.plate_offset)
            # ARC-EN-CIEL DANS LA PARALLAXE : chaque couronne reçoit sa
            # propre teinte, décalée le long du spectre, et l'ensemble
            # défile avec la phase. Comme `rainbow` est un ENTIER de cycles
            # par boucle et que hue_rotate d'un tour entier est l'identité
            # exacte, le bouclage reste intact.
            if p.rainbow:
                spread = p.rainbow_spread * li / nlayers
                acc_turns = p.rainbow * phase + spread + L.hue_offset
                col = F.hue_rotate(col, acc_turns)
            acc += col * m[..., None]
            wsum += m
        rgb = acc / np.maximum(wsum, 1e-4)[..., None]
        # l'intensité lumineuse vient de la densité cumulée des couches
        rgb *= np.clip(wsum, 0.0, 1.9)[..., None]

        # --- rayons crépusculaires ---------------------------------------
        if p.rays and p.ray_strength > 0:
            v = 0.5 + 0.5 * np.cos(p.rays * th - TAU * p.ray_turns * phase)
            v = (v ** 3).astype(np.float32)
            falloff = F.smoothstep(1.25, 0.18, rn)
            rgb += (v * falloff * np.float32(p.ray_strength))[..., None] * \
                np.array([1.0, 0.95, 0.86], np.float32)

        # --- puits sombre central ----------------------------------------
        rgb *= F.smoothstep(p.core_dark * 0.34, p.core_dark * 1.30, rn)[..., None]

        # --- l'âme : traînée, noyau, étincelles ---------------------------
        s = p.soul
        for i in range(s.trail, 0, -1):
            f = i / float(s.trail)
            ph = phase - s.trail_span * f
            x, y = self.soul_xy(ph, s)
            d = self._mask_from_xy(x, y, s.radius * (1.0 - 0.55 * f))
            a = float(np.exp(-3.1 * f)) * 0.30
            rgb += np.exp(-(d * d) * 1.7)[..., None] * \
                np.array(s.tint, np.float32) * np.float32(a)

        sx, sy = self.soul_xy(phase, s)
        d = self._mask_from_xy(sx, sy, s.radius)
        rgb += np.exp(-((d / s.glow) ** 2) * 1.15)[..., None] * \
            np.array(s.tint, np.float32) * np.float32(0.60)
        rgb += np.exp(-(d ** 2) * 2.3)[..., None] * \
            np.array(s.core, np.float32) * np.float32(1.30)
        rgb += F.smoothstep(1.0, 0.42, d)[..., None] * \
            np.array(s.core, np.float32) * np.float32(0.95)

        # branches d'étoile : la sphère projette des pointes fines,
        # comme le flare de la référence.
        if p.spikes and p.spike_strength > 0:
            half = min(self.h, self.w) / 2.0
            cx = (self.w - 1) / 2.0 + sx * half
            cy = (self.h - 1) / 2.0 + sy * half
            yy, xx = np.indices((self.h, self.w)).astype(np.float32)
            dx = xx - np.float32(cx)
            dy = yy - np.float32(cy)
            dist = np.hypot(dx, dy) / np.float32(half)
            ang = np.arctan2(dy, dx)
            v = np.abs(np.cos(p.spikes * 0.5 * ang)) ** 26
            fall = np.exp(-(dist / np.float32(p.spike_len)) ** 2 * 3.0)
            rgb += (v * fall)[..., None] * np.array(s.core, np.float32) * \
                np.float32(p.spike_strength)

        for k in range(s.spark_count):
            a = TAU * (s.spark_turns * phase + k / float(s.spark_count))
            ex = sx + np.cos(a) * s.radius * 2.5
            ey = sy + np.sin(a) * s.radius * 1.7
            dd = self._mask_from_xy(float(ex), float(ey), s.radius * 0.20)
            rgb += np.exp(-(dd * dd) * 2.0)[..., None] * \
                np.array(s.core, np.float32) * np.float32(0.30)

        # --- voile lumineux ----------------------------------------------
        if p.bloom > 0:
            lum = rgb.mean(2)
            hot = np.clip(lum - 0.62, 0.0, None)
            rgb += F.blur(hot, 13.0)[..., None] * np.float32(p.bloom) * \
                np.array([1.0, 0.96, 0.90], np.float32)

        # --- vignette, grain, exposition ---------------------------------
        rgb *= F.smoothstep(p.vignette, p.vignette * 0.52, rn)[..., None]
        if p.grain:
            rgb *= (1.0 + self.grain * np.float32(p.grain))[..., None]
        rgb *= np.float32(p.exposure)

        # tone map doux : garde les hautes lumières sans écrêter brutalement
        rgb = rgb / (1.0 + rgb * 0.52)

        # saturation en fin de chaîne : le tone map désature, on récupère
        # le violet et le turquoise sans réintroduire d'écrêtage.
        if p.saturation != 1.0:
            lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
            rgb = lum + (rgb - lum) * np.float32(p.saturation)

        return np.clip(rgb, 0.0, 1.0)

    def frame(self, phase: float) -> Image.Image:
        return Image.fromarray((self.render(phase) * 255.0 + 0.5).astype(np.uint8), "RGB")


def render_frames(plate: str, n: int, w: int = 960, h: int = 540,
                  params: HaloParams | None = None) -> list[Image.Image]:
    r = HaloRenderer(plate, w=w, h=h, params=params)
    return [r.frame(i / float(n)) for i in range(n)]
