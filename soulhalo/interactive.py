"""
Couche interactive : la souris pilote le point de vue, pas le temps.

DEUX AXES INDÉPENDANTS
----------------------
`phase`  le temps de l'animation — boucle à l'infini, jamais interrompu ;
`look`   l'angle de vue, piloté par la souris — libre, sans effet sur la boucle.

C'est la règle du projet appliquée à l'entrée du joueur : bouger la souris
ne fait jamais avancer la séquence, cela ne fait que déplacer la caméra.
Le fond continue de tourner exactement pareil.

CE QUE CONTIENT CE MODULE
-------------------------
`LookState`      lissage de la souris (le pointeur saute, la caméra non) ;
`DreamBackdrop`  fond circulaire en parallaxe dont la teinte défile dans le
                 spectre, et dont l'angle suit la souris ;
`Carousel`       défilement des Pokémon avec paillettes, halo arc-en-ciel
                 au survol et aura de type à la sélection.

BOUCLAGE — toute vitesse est un entier de tours par boucle. `look` n'entre
jamais dans un terme périodique : il décale, il ne dérive pas.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import field as F

TAU = F.TAU
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def load_types(path="personality_test/types.json"):
    return json.load(open(_p(path), encoding="utf-8"))


# --------------------------------------------------------------------------
# la souris
# --------------------------------------------------------------------------
@dataclass
class LookState:
    """Lisse la position de la souris.

    Le pointeur saute d'une frame à l'autre ; appliqué tel quel à la caméra,
    le fond tremblerait. On interpole donc vers la cible à vitesse bornée.
    `smooth` est la fraction du chemin parcourue par frame.
    """
    x: float = 0.0          # -1 (gauche) .. +1 (droite)
    y: float = 0.0          # -1 (haut)   .. +1 (bas)
    smooth: float = 0.18
    max_tilt: float = 0.30  # amplitude max du décalage de point de vue

    def update(self, tx: float, ty: float):
        tx = float(np.clip(tx, -1.0, 1.0))
        ty = float(np.clip(ty, -1.0, 1.0))
        self.x += (tx - self.x) * self.smooth
        self.y += (ty - self.y) * self.smooth
        return self

    @property
    def offset(self):
        return self.x * self.max_tilt, self.y * self.max_tilt

    @property
    def angle(self):
        """Angle de vue en tours : la souris fait tourner le décor."""
        return self.x * 0.12


# --------------------------------------------------------------------------
# fond de rêve : parallaxe circulaire dont la couleur défile
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DreamParams:
    rings: tuple = ((1.20, 0.46, +1), (0.92, 0.32, -2), (0.68, 0.24, +3),
                    (0.46, 0.18, -4), (0.28, 0.14, +6))
    spectrum_cycles: int = 1    # tours de spectre par boucle — ENTIER
    breathe: int = 1            # ENTIER
    saturation: float = 1.50
    exposure: float = 1.00
    vignette: float = 1.45
    parallax: float = 0.55      # amplitude du deplacement des plans


class DreamBackdrop:
    """Fond circulaire en parallaxe, teinte défilante, angle piloté souris.

    Les plans les plus proches se déplacent PLUS que les plans lointains :
    c'est ce gradient qui donne le relief quand la souris bouge.
    """

    def __init__(self, plate="research/plates/plate_rainbow_rings.png",
                 w=960, h=540, params: DreamParams | None = None):
        self.w, self.h = int(w), int(h)
        self.p = params or DreamParams()
        self.strip = F.polar_strip(_p(plate), n_theta=1024, n_rad=512)
        yy, xx = np.indices((self.h, self.w)).astype(np.float32)
        self.yy, self.xx = yy, xx
        self.half = min(self.h, self.w) / 2.0
        self.cx = (self.w - 1) / 2.0
        self.cy = (self.h - 1) / 2.0

    def _polar_at(self, ox, oy):
        """Recalcule r et theta pour un centre decale (le point de vue)."""
        dx = (self.xx - self.cx) / self.half - np.float32(ox)
        dy = (self.yy - self.cy) / self.half - np.float32(oy)
        return np.hypot(dx, dy).astype(np.float32), np.arctan2(dy, dx).astype(np.float32)

    def render(self, phase: float, look: LookState | None = None) -> np.ndarray:
        p = self.p
        phase = float(phase) % 1.0
        lx, ly = (0.0, 0.0) if look is None else look.offset
        la = 0.0 if look is None else look.angle

        acc = np.zeros((self.h, self.w, 3), np.float32)
        wsum = np.zeros((self.h, self.w), np.float32)
        n = len(p.rings)
        for i, (rad, wid, speed) in enumerate(p.rings):
            # profondeur : 0 = lointain, 1 = proche. Les plans proches
            # bougent davantage -> parallaxe reelle sous la souris.
            depth = (i + 1) / n
            k = p.parallax * depth
            rn, th = self._polar_at(lx * k, ly * k)
            br = 0.010 * float(np.cos(TAU * p.breathe * phase))
            m = F.ring(rn, rad + br, wid)
            col = F.sample_strip(self.strip, rn, th,
                                 rot=speed * phase + la * depth,
                                 rad_scale=0.88, rad_offset=0.12)
            acc += col * m[..., None]
            wsum += m
        rgb = acc / np.maximum(wsum, 1e-4)[..., None]
        rgb *= np.clip(wsum, 0.0, 1.7)[..., None]

        # LE SPECTRE DEFILE : entier de tours => raccord exact
        if p.spectrum_cycles:
            rgb = F.hue_rotate(rgb, p.spectrum_cycles * phase)

        rn0, _ = self._polar_at(lx * 0.2, ly * 0.2)
        rgb *= F.smoothstep(p.vignette, p.vignette * 0.45, rn0)[..., None]
        rgb = np.clip(rgb, 0, None) * np.float32(p.exposure)
        rgb = rgb / (1.0 + rgb * 0.48)
        lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
        return np.clip(lum + (rgb - lum) * np.float32(p.saturation), 0.0, 1.0)


# --------------------------------------------------------------------------
# carrousel de Pokémon
# --------------------------------------------------------------------------
@dataclass
class CarouselParams:
    slots: int = 5              # vignettes visibles
    spacing: float = 0.62       # écart horizontal, en demi-hauteurs
    scale: int = 4              # agrandissement du sprite (plus proche voisin)
    arc: float = 0.055          # courbure du carrousel
    hover_halo: float = 1.60    # halo arc-en-ciel au survol
    aura: float = 1.70          # aura de type à la sélection
    sparkle: float = 0.85
    sparkle_turns: int = 2      # ENTIER
    dim_bg: float = 0.45        # attenuation du fond derriere le carrousel


class Carousel:
    """Défilement des Pokémon, avec paillettes, halo de survol et aura.

    `pos` est continu : 3.0 = le 4e Pokémon est centré, 3.5 = à mi-chemin.
    Le joueur fait défiler ; l'animation de fond, elle, ne s'arrête jamais.
    """

    def __init__(self, roster, types, sprites_dir="personality_test/sprites",
                 w=960, h=540, params: CarouselParams | None = None,
                 sparkle_plate="research/plates/plate_sparkles.png"):
        self.roster, self.types = roster, types
        self.w, self.h = int(w), int(h)
        self.p = params or CarouselParams()
        self.dir = _p(sprites_dir)
        self._cache = {}
        sp = _p(sparkle_plate)
        self.sparkles = None
        if os.path.exists(sp):
            a = np.asarray(Image.open(sp).convert("RGB"), np.float32) / 255.0
            self.sparkles = a
        yy, xx = np.indices((self.h, self.w)).astype(np.float32)
        self.yy, self.xx = yy, xx
        self.half = min(self.h, self.w) / 2.0

    # -- sprites -----------------------------------------------------------
    def sprite(self, dex, frame=0):
        """Frame d'Idle. La taille de case vient d'AnimData.xml."""
        key = (dex, frame)
        if key in self._cache:
            return self._cache[key]
        import xml.etree.ElementTree as ET
        base = os.path.join(self.dir, dex)
        xml, sheet = os.path.join(base, "AnimData.xml"), os.path.join(base, "Idle-Anim.png")
        if not (os.path.exists(xml) and os.path.exists(sheet)):
            self._cache[key] = None
            return None
        root = ET.parse(xml).getroot()
        node = next((a for a in root.iter("Anim") if a.findtext("Name") == "Idle"), None)
        if node is None or node.findtext("FrameWidth") is None:
            self._cache[key] = None
            return None
        fw, fh = int(node.findtext("FrameWidth")), int(node.findtext("FrameHeight"))
        im = Image.open(sheet).convert("RGBA")
        ncol = max(1, im.width // fw)
        f = frame % ncol
        # ligne 0 = direction "face" ; colonne = frame d'animation
        out = im.crop((f * fw, 0, f * fw + fw, fh))
        self._cache[key] = out
        return out

    def idle_frames(self, dex):
        """Nombre de frames d'Idle et leurs durees, lues dans le XML."""
        import xml.etree.ElementTree as ET
        xml = os.path.join(self.dir, dex, "AnimData.xml")
        if not os.path.exists(xml):
            return []
        root = ET.parse(xml).getroot()
        node = next((a for a in root.iter("Anim") if a.findtext("Name") == "Idle"), None)
        if node is None:
            return []
        return [int(d.text) for d in node.iter("Duration")]

    def frame_at(self, dex, tick):
        """Frame courante d'apres les DUREES reelles du XML.

        PMD n'anime pas a cadence fixe : chaque frame a sa duree propre
        (ex. Salameche 12, 8, 8, 8). Ignorer ces durees donnerait une
        respiration fausse.
        """
        dur = self.idle_frames(dex)
        if not dur:
            return 0
        total = sum(dur)
        t = int(tick) % total
        acc = 0
        for i, d in enumerate(dur):
            acc += d
            if t < acc:
                return i
        return 0

    # -- rendu -------------------------------------------------------------
    def _slot_xy(self, offset):
        """Position d'une vignette, `offset` en nombre de crans depuis le centre."""
        x = offset * self.p.spacing
        y = self.p.arc * offset * offset       # léger arc, comme un éventail
        return x, y

    def _blit(self, out, img, cx, cy, k, alpha, tint=None, tint_amt=0.0):
        if img is None:
            return
        iw, ih = img.size
        im = img.resize((max(1, iw * k), max(1, ih * k)), Image.NEAREST)
        a = np.asarray(im, np.float32) / 255.0
        rgb, al = a[..., :3], a[..., 3:] * alpha
        if tint is not None and tint_amt > 0:
            rgb = rgb * (1 - tint_amt) + np.asarray(tint, np.float32) * tint_amt
        ph, pw = rgb.shape[:2]
        x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
        x1, y1, x2, y2 = max(0, x0), max(0, y0), min(self.w, x0 + pw), min(self.h, y0 + ph)
        if x2 <= x1 or y2 <= y1:
            return
        sub = out[y1:y2, x1:x2]
        r = rgb[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        aa = al[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        out[y1:y2, x1:x2] = sub * (1.0 - aa) + r * aa

    def _radial_glow(self, cx, cy, radius, color, strength):
        dx = (self.xx - cx) / self.half
        dy = (self.yy - cy) / self.half
        d = np.hypot(dx, dy) / max(radius, 1e-4)
        return np.exp(-(d ** 2) * 1.3)[..., None] * \
            np.asarray(color, np.float32) * np.float32(strength)

    def render(self, bg, pos, phase, hover=None, selected=None, tick=0):
        """Compose le carrousel sur `bg`.

        `pos`      position continue (3.0 = 4e centre)
        `hover`    index survolé  -> halo arc-en-ciel
        `selected` index choisi   -> aura de type
        `tick`     compteur de frames, pour animer les sprites
        """
        p = self.p
        # Le fond de reve est tres lumineux : pose tels quels, le halo de
        # survol et l'aura de type ne s'y voyaient pas (0.003 de difference
        # mesuree, soit rien). On l'attenue donc derriere le carrousel, ce
        # qui rend de la place aux lumieres additives.
        out = bg.copy() * np.float32(1.0 - p.dim_bg)
        phase = float(phase) % 1.0
        n = len(self.roster)
        centre = int(round(pos))

        # paillettes, en fond du carrousel
        if self.sparkles is not None and p.sparkle > 0:
            sh, sw = self.sparkles.shape[:2]
            u = np.mod(self.xx / self.w + p.sparkle_turns * phase, 1.0)
            v = np.mod(self.yy / self.h * 0.8 + 0.13 * phase, 1.0)
            xi = (u * (sw - 1)).astype(np.int32)
            yi = (v * (sh - 1)).astype(np.int32)
            tw = 0.55 + 0.45 * float(np.cos(TAU * phase))
            out = 1.0 - (1.0 - out) * (1.0 - self.sparkles[yi, xi] *
                                       np.float32(p.sparkle * tw))

        half_slots = p.slots // 2
        for k in range(centre - half_slots, centre + half_slots + 1):
            if not (0 <= k < n):
                continue
            off = k - pos
            if abs(off) > half_slots + 0.5:
                continue
            x, y = self._slot_xy(off)
            cx = (self.w - 1) / 2.0 + x * self.half
            cy = (self.h - 1) / 2.0 + y * self.half
            near = float(np.clip(1.0 - abs(off) / (half_slots + 0.5), 0.0, 1.0))
            scale = max(1, int(round(p.scale * (0.55 + 0.45 * near))))
            alpha = 0.35 + 0.65 * near
            mon = self.roster[k]

            # AURA DE TYPE : seulement sur le Pokémon choisi
            if selected == k:
                col = self.types["types"].get(mon["type"], {}).get("rgb", [1, 1, 1])
                puls = 0.72 + 0.28 * float(np.cos(TAU * phase))
                out = out + self._radial_glow(cx, cy, 0.30, col, p.aura * puls)

            # HALO ARC-EN-CIEL : au survol
            if hover == k:
                hue = F.hue_rotate(np.array([[[1.0, 0.25, 0.25]]], np.float32),
                                   phase)[0, 0]
                puls = 0.75 + 0.25 * float(np.cos(TAU * 2 * phase))
                out = out + self._radial_glow(cx, cy, 0.24, hue,
                                              p.hover_halo * puls)

            fr = self.frame_at(mon["dex"], tick)
            self._blit(out, self.sprite(mon["dex"], fr), cx, cy, scale, alpha)

        return np.clip(out, 0.0, 1.0)
