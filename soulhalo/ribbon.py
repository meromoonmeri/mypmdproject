"""
Ruban de résultat, style PMD — maintien de la souris et révélation.

Le quiz aboutit sur un RUBAN circulaire de lumière dont la couleur est
celle de la nature dominante. Le joueur MAINTIENT le bouton : le ruban se
charge, s'intensifie, puis le portrait et le sprite du Pokémon
apparaissent, assortis au fond.

DEUX RÉGIMES, JAMAIS MÉLANGÉS
-----------------------------
`idle(phase)`   boucle infinie, le joueur peut rester indéfiniment ;
`charge(t, phase)` état de maintien, piloté par le joueur : `t` est la
                fraction maintenue (0 → 1), `phase` continue de boucler
                par-dessous. Relâcher avant 1 ramène simplement `t` à 0,
                sans rupture — l'animation de fond n'a jamais été
                interrompue.

BOUCLAGE — toute vitesse est un entier de tours par boucle et toute
respiration s'écrit cos(2*pi*k*phase) : `idle(0) == idle(1)` au bit près.
La charge se superpose sans casser cette propriété, car `t` est indépendant
de `phase`.

COULEUR DE NATURE — le ruban est peint une fois en arc-en-ciel, puis
teinté par rotation de teinte (YIQ, luminance préservée). Changer de nature
ne coûte donc rien et ne délave jamais la plaque.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import field as F

TAU = F.TAU
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_natures(path="personality_test/natures.json"):
    if not os.path.isabs(path):
        path = os.path.join(_HERE, path)
    d = json.load(open(path, encoding="utf-8"))
    return {n["id"]: n for n in d["natures"]}


@dataclass(frozen=True)
class RibbonParams:
    spin: int = 1               # tours du ruban par boucle — ENTIER
    counter_spin: int = -2      # nappe inverse — ENTIER
    breathe: int = 1            # cycles de respiration — ENTIER
    radius: float = 0.62
    width: float = 0.30
    glow: float = 0.55
    sparks: int = 14            # étincelles qui courent sur le ruban
    spark_turns: int = 2        # ENTIER
    saturation: float = 1.45
    exposure: float = 1.05
    vignette: float = 1.30
    # charge (maintien du bouton)
    charge_spin: int = 3        # accélération à pleine charge — ENTIER
    charge_glow: float = 1.9
    charge_squeeze: float = 0.22   # le ruban se resserre en se chargeant
    # badge d'Équipe de Secours (œuf + deux ailes) au centre du ruban
    badge: str = "personality_test/badge_base.png"
    badge_scale: float = 0.46   # largeur du badge, en fraction de l'écran
    badge_bob: int = 1          # cycles de flottement — ENTIER
    badge_glow: float = 0.55


class RibbonRenderer:
    """Ruban de résultat. La plaque et les grilles sont mises en cache."""

    def __init__(self, plate="research/plates/plate_ribbon.png",
                 w=960, h=540, params: RibbonParams | None = None,
                 nature: str = "jolly", natures=None):
        if not os.path.isabs(plate):
            plate = os.path.join(_HERE, plate)
        self.w, self.h = int(w), int(h)
        self.p = params or RibbonParams()
        self.strip = F.polar_strip(plate, n_theta=1024, n_rad=512)
        self.rn, self.th = F.polar(self.h, self.w)
        diag = float(np.hypot(self.h, self.w)) / min(self.h, self.w)
        self.rd = (self.rn / np.float32(diag)).astype(np.float32)
        self.natures = natures if natures is not None else load_natures()
        self.badge, self.badge_a = self._load_badge()
        self.set_nature(nature)

    def _load_badge(self):
        """Charge le badge et en tire un masque alpha.

        Le badge est peint en BLANC sur noir : sa luminance sert donc
        directement d'alpha, et sa couleur sera celle de la nature. Un
        badge deja colore ne pourrait pas etre reteinte proprement.
        """
        p = self.p.badge
        if not p:
            return None, None
        if not os.path.isabs(p):
            p = os.path.join(_HERE, p)
        if not os.path.exists(p):
            return None, None
        src = Image.open(p).convert("RGB")
        # Recadrage sur le badge REEL : la plaque generee est carree mais le
        # badge n'occupe qu'une bande centrale. Sans ce recadrage, la mise a
        # l'echelle porte sur le vide autour et le badge sort minuscule.
        m = np.asarray(src, np.float32).mean(2) / 255.0
        ys, xs = np.where(m > 0.16)
        if len(xs):
            pad = 4
            src = src.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                            min(src.width, xs.max() + pad),
                            min(src.height, ys.max() + pad)))
        bw = int(self.w * self.p.badge_scale)
        bh = max(1, int(bw * src.height / src.width))
        src = src.resize((bw, bh), Image.LANCZOS)
        a = np.asarray(src, np.float32) / 255.0
        lum = a.mean(2)
        alpha = np.clip(lum * 1.6, 0.0, 1.0)
        return a, alpha

    def _blit_badge(self, rgb, phase, t):
        """Pose le badge teinte au centre, avec flottement et lueur."""
        if self.badge is None:
            return rgb
        p = self.p
        bh, bw = self.badge.shape[:2]
        bob = 0.012 * float(np.sin(TAU * p.badge_bob * phase))
        cx = (self.w - bw) // 2
        cy = int((self.h - bh) / 2 + bob * self.h)
        x1, y1 = max(0, cx), max(0, cy)
        x2, y2 = min(self.w, cx + bw), min(self.h, cy + bh)
        if x2 <= x1 or y2 <= y1:
            return rgb
        sb = self.badge[y1 - cy:y2 - cy, x1 - cx:x2 - cx]
        sa = self.badge_a[y1 - cy:y2 - cy, x1 - cx:x2 - cx][..., None]
        # Teinte de la nature. Le badge blanc garde son modele d'ombres,
        # mais on NE peut PAS empiler teinte + accent + lueur : les trois
        # additionnes saturent les trois canaux et le badge redevient
        # blanc, quelle que soit la nature. L'accent est donc reserve aux
        # hautes lumieres (le reflet de l'oeuf) et la lueur reste modeste.
        hi = np.clip((sb.mean(2, keepdims=True) - 0.72) / 0.28, 0.0, 1.0)
        col = sb * self.tint * np.float32(1.0 + p.badge_glow * (0.5 + 1.1 * t))
        col += hi * self.accent * np.float32(0.35 + 0.9 * t)
        sub = rgb[y1:y2, x1:x2]
        rgb[y1:y2, x1:x2] = sub * (1.0 - sa) + col * sa
        return rgb

    # -- nature ------------------------------------------------------------
    def set_nature(self, nature: str):
        if nature not in self.natures:
            raise KeyError(f"nature inconnue : {nature}")
        self.nature = nature
        n = self.natures[nature]
        self.multicolore = bool(n.get("multicolore"))
        # la plaque est un arc-en-ciel : on la fait tourner pour amener la
        # teinte voulue en tete. 0 deg de la plaque ~ vert (mesure).
        self.hue_turn = 0.0 if self.multicolore else ((n["hue"] - 96) % 360) / 360.0
        self.tint = np.array(n["rgb"], np.float32)
        self.accent = np.array(n["accent"], np.float32)

    # -- rendu -------------------------------------------------------------
    def _ribbon(self, phase, t=0.0):
        p = self.p
        rd, th = self.rd, self.th
        # le ruban se resserre et s'intensifie pendant le maintien
        rad = p.radius * (1.0 - p.charge_squeeze * t)
        wid = p.width * (1.0 - 0.25 * t)
        breathe = 0.012 * float(np.cos(TAU * p.breathe * phase))
        m = F.ring(rd, rad + breathe, wid)

        spin = p.spin + p.charge_spin * t
        col = F.sample_strip(self.strip, rd, th,
                             rot=spin * phase, rad_scale=0.92, rad_offset=0.06)
        turn = self.hue_turn + (phase if self.multicolore else 0.0)
        if turn:
            col = F.hue_rotate(col, turn)

        rgb = col * m[..., None]
        # nappe inverse : donne l'epaisseur et le cisaillement
        col2 = F.sample_strip(self.strip, rd, th,
                              rot=p.counter_spin * phase,
                              rad_scale=0.78, rad_offset=0.10)
        if turn:
            col2 = F.hue_rotate(col2, turn)
        rgb += col2 * F.ring(rd, rad * 0.86, wid * 0.66)[..., None] * 0.45

        # etincelles qui courent le long du ruban
        if p.sparks:
            v = 0.5 + 0.5 * np.cos(p.sparks * th - TAU * p.spark_turns * phase)
            rgb += (v ** 12 * m)[..., None] * self.accent * \
                np.float32(0.5 + 1.4 * t)
        return rgb, m

    def idle(self, phase: float) -> np.ndarray:
        """Boucle infinie. Le joueur peut rester indefiniment."""
        return self._compose(phase, 0.0)

    def charge(self, t: float, phase: float) -> np.ndarray:
        """Maintien du bouton. `t` = fraction maintenue (0 -> 1)."""
        return self._compose(phase, float(np.clip(t, 0.0, 1.0)))

    def _compose(self, phase, t):
        p = self.p
        phase = float(phase) % 1.0
        rgb, m = self._ribbon(phase, t)

        # coeur : la lumiere qui grandit pendant la charge
        pulse = 0.5 + 0.5 * float(np.cos(TAU * p.breathe * phase))
        core = np.exp(-(self.rd / np.float32(0.10 + 0.22 * t)) ** 2)
        rgb += core[..., None] * self.tint * \
            np.float32(p.glow * (0.4 + 0.6 * pulse) + p.charge_glow * t)
        # noyau blanc, seulement en fin de charge
        if t > 0:
            wc = np.exp(-(self.rd / np.float32(0.045 + 0.05 * t)) ** 2)
            rgb += wc[..., None] * np.array([1.0, 0.99, 0.96], np.float32) * \
                np.float32(1.7 * t * t)

        # le badge est pose AVANT la vignette pour rester dans l'ambiance,
        # mais APRES le coeur pour ne pas etre noye par la lueur de charge
        rgb = self._blit_badge(rgb, phase, t)

        rgb *= F.smoothstep(p.vignette, p.vignette * 0.42, self.rd)[..., None]
        rgb = np.clip(rgb, 0, None) * np.float32(p.exposure * (1.0 + 0.35 * t))
        rgb = rgb / (1.0 + rgb * 0.48)
        lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
        rgb = lum + (rgb - lum) * np.float32(p.saturation)
        return np.clip(rgb, 0.0, 1.0)

    def frame(self, phase, t=0.0):
        return Image.fromarray((self._compose(phase, t) * 255.0 + 0.5).astype(np.uint8))


# --------------------------------------------------------------------------
# révélation : portrait + sprite, assortis au fond
# --------------------------------------------------------------------------
def paste_reveal(bg: np.ndarray, portrait: Image.Image | None,
                 sprite: Image.Image | None, u: float,
                 tint=(1.0, 1.0, 1.0), scale: int = 4) -> np.ndarray:
    """Compose la revelation par-dessus le fond.

    `u` va de 0 (rien) a 1 (entierement visible). La revelation passe par
    une SILHOUETTE teintee avant de laisser voir les couleurs, comme dans
    PMD : on ne montre jamais l'image complete d'un coup.

    Les sprites PMD sont en pixel art : l'agrandissement est fait au PLUS
    PROCHE VOISIN, sinon les contours bavent.
    """
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0:
        return bg
    out = bg.copy()
    h, w = out.shape[:2]
    tint = np.asarray(tint, np.float32)

    def blit(img, cx, cy, k, alpha, silhouette):
        if img is None:
            return
        iw, ih = img.size
        im = img.resize((iw * k, ih * k), Image.NEAREST)
        a = np.asarray(im.convert("RGBA"), np.float32) / 255.0
        rgb, al = a[..., :3], a[..., 3:] * alpha
        if silhouette > 0:
            # silhouette teintee : la couleur revient en fondu
            flat = np.ones_like(rgb) * tint
            rgb = rgb * (1.0 - silhouette) + flat * silhouette
        ph, pw = rgb.shape[:2]
        x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
        x1, y1 = max(0, x0), max(0, y0)
        x2, y2 = min(w, x0 + pw), min(h, y0 + ph)
        if x2 <= x1 or y2 <= y1:
            return
        sub = out[y1:y2, x1:x2]
        r = rgb[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        aa = al[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        out[y1:y2, x1:x2] = sub * (1.0 - aa) + r * aa

    # la silhouette se resorbe sur la premiere moitie de la revelation
    sil = float(np.clip(1.0 - u * 1.8, 0.0, 1.0))
    blit(sprite, w * 0.5, h * 0.54, scale, min(1.0, u * 2.2), sil)
    blit(portrait, w * 0.5, h * 0.235, max(1, scale - 1),
         float(np.clip((u - 0.35) / 0.5, 0.0, 1.0)), sil * 0.6)
    return np.clip(out, 0.0, 1.0)


def first_frame(sheet_path: str, xml_path: str, anim="Idle", direction=0):
    """Extrait une frame d'une planche SpriteCollab.

    Les planches sont des grilles : une LIGNE par direction, une COLONNE
    par frame. La taille de case vient d'AnimData.xml, jamais devinee.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    node = None
    for a in root.iter("Anim"):
        if a.findtext("Name") == anim:
            node = a
            break
    if node is None:
        return None
    fw = node.findtext("FrameWidth")
    fh = node.findtext("FrameHeight")
    if fw is None or fh is None:      # anim par renvoi (CopyOf)
        return None
    fw, fh = int(fw), int(fh)
    sheet = Image.open(sheet_path).convert("RGBA")
    box = (0, direction * fh, fw, direction * fh + fh)
    return sheet.crop(box)
