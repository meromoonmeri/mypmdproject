"""
Séquence d'intro : de l'écran noir jusqu'au voyage dans le portail.

PRINCIPE — ANIMATION = BOUCLE, DIALOGUE = PROGRESSION
-----------------------------------------------------
Chaque état est une boucle infinie autonome. Rien n'avance tout seul : le
passage d'un état au suivant est déclenché par le joueur. Les transitions
sont donc, elles aussi, des séquences de frames — mais jouées UNE fois.

    S1 VIDE       noir, la sphère s'allume            [boucle]
      -> T1 la sphère s'ébranle                       [une fois]
    S2 CIEL       voyage parmi les étoiles            [boucle]
      -> T2 arrêt, flash blanc progressif             [une fois]
    S3 HALO       spectre arc-en-ciel, le portail naît [boucle]
      -> T3 aspiration dans le portail                [une fois]
    S4 PORTAIL    voyage dans le tunnel               [boucle]

BOUCLAGE — chaque état boucle exactement : toute vitesse est un entier de
tours par boucle, toute respiration s'écrit cos(2*pi*k*phase). La dernière
frame se raccorde à la première au bit près.

RACCORD — une transition part de l'état N à sa phase 0 et arrive sur l'état
N+1 à sa phase 0. Le joueur ne voit donc jamais de saut, quel que soit le
moment où il appuie : on termine la boucle en cours, puis on joue la
transition.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import field as F
from .tunnel import TunnelParams, TunnelRenderer

TAU = F.TAU
SOUL_CORE = np.array([1.0, 0.995, 0.97], np.float32)
SOUL_TINT = np.array([0.66, 0.90, 1.0], np.float32)


# --------------------------------------------------------------------------
# outils communs
# --------------------------------------------------------------------------
def _grid(h: int, w: int):
    rn, th = F.polar(h, w)
    diag = float(np.hypot(h, w)) / min(h, w)
    return (rn / np.float32(diag)).astype(np.float32), th, rn


def _soul(rgb, h, w, cx, cy, radius, glow=1.0, halo=1.0):
    """La sphère-âme. Même dessin partout : c'est le fil de la séquence."""
    yy, xx = np.indices((h, w)).astype(np.float32)
    half = min(h, w) / 2.0
    dx = (xx - (w - 1) / 2.0) / half - np.float32(cx)
    dy = (yy - (h - 1) / 2.0) / half - np.float32(cy)
    d = np.hypot(dx, dy) / np.float32(max(radius, 1e-4))
    rgb += np.exp(-(d ** 2) * 1.10)[..., None] * SOUL_TINT * np.float32(0.55 * halo)
    rgb += np.exp(-(d ** 2) * 3.20)[..., None] * SOUL_CORE * np.float32(1.30 * glow)
    return rgb


def _tone(rgb, exposure=1.0, saturation=1.0):
    rgb = np.clip(rgb, 0.0, None) * np.float32(exposure)
    rgb = rgb / (1.0 + rgb * 0.48)
    if saturation != 1.0:
        lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2, keepdims=True)
        rgb = lum + (rgb - lum) * np.float32(saturation)
    return np.clip(rgb, 0.0, 1.0)


# --------------------------------------------------------------------------
# S1 — le vide : noir, la sphère s'allume et respire
# --------------------------------------------------------------------------
class StateVoid:
    name = "01_vide"
    doc = "Ecran noir. La sphere-ame s'illumine et respire."

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rd, self.th, _ = _grid(h, w)

    def render(self, phase):
        phase = float(phase) % 1.0
        rgb = np.zeros((self.h, self.w, 3), np.float32)
        # respiration : 1 cycle entier par boucle => raccord exact
        pulse = 0.5 + 0.5 * float(np.cos(TAU * phase))
        r = 0.045 * (1.0 + 0.16 * pulse)
        rgb = _soul(rgb, self.h, self.w, 0.0, 0.0, r,
                    glow=0.55 + 0.45 * pulse, halo=0.45 + 0.55 * pulse)
        # halo très diffus, il fait exister le noir autour
        rgb += np.exp(-(self.rd / np.float32(0.42)) ** 2)[..., None] * \
            SOUL_TINT * np.float32(0.055 * (0.6 + 0.4 * pulse))
        return _tone(rgb, 1.0, 1.20)


# --------------------------------------------------------------------------
# S2 — le ciel : bleu nuit, étoiles en parallaxe, la sphère voyage
# --------------------------------------------------------------------------
class StateSky:
    name = "02_ciel"
    doc = "Bleu nuit etoile. Trois nappes d'etoiles en parallaxe."

    def __init__(self, w, h, plate="research/plates/plate_starfield.png"):
        self.w, self.h = w, h
        self.rd, self.th, _ = _grid(h, w)
        im = np.asarray(Image.open(plate).convert("RGB"), np.float32) / 255.0
        self.plate = im
        self.ph, self.pw = im.shape[:2]
        yy, xx = np.indices((h, w)).astype(np.float32)
        self.u = xx / float(w)
        self.v = yy / float(h)
        # trois nappes : vitesses ENTIERES et croissantes = parallaxe
        self.shells = [(1, 0.55, 1.00), (2, 0.85, 0.62), (4, 1.35, 0.34)]

    def render(self, phase):
        phase = float(phase) % 1.0
        rgb = np.zeros((self.h, self.w, 3), np.float32)
        for speed, zoom, alpha in self.shells:
            # défilement horizontal ENTIER de tuiles par boucle : raccord exact
            uu = np.mod(self.u * zoom + speed * phase, 1.0)
            vv = np.mod(self.v * zoom, 1.0)
            xi = (uu * (self.pw - 1)).astype(np.int32)
            yi = (vv * (self.ph - 1)).astype(np.int32)
            rgb += self.plate[yi, xi] * np.float32(alpha)
        rgb /= sum(a for _, _, a in self.shells)
        pulse = 0.5 + 0.5 * float(np.cos(TAU * phase))
        rgb = _soul(rgb, self.h, self.w, 0.0, 0.0,
                    0.052 * (1.0 + 0.10 * pulse),
                    glow=0.85 + 0.15 * pulse, halo=0.9)
        return _tone(rgb, 1.06, 1.25)


# --------------------------------------------------------------------------
# S3 — le halo : spectre arc-en-ciel, le portail apparaît
# --------------------------------------------------------------------------
class StateHalo:
    name = "03_halo"
    doc = "Halo de spectre arc-en-ciel. Le portail se forme."

    def __init__(self, w, h, plate="research/plates/plate_rainbow_rings.png"):
        self.w, self.h = w, h
        self.rd, self.th, _ = _grid(h, w)
        self.strip = F.polar_strip(plate, n_theta=1024, n_rad=512)
        # couronnes : vitesses ENTIERES, sens alternes => profondeur
        self.rings = [(1.12, 0.30, +1, 0.55), (0.90, 0.22, -2, 0.75),
                      (0.70, 0.18, +3, 0.95), (0.52, 0.15, -4, 1.00),
                      (0.36, 0.13, +6, 0.85), (0.24, 0.11, -8, 0.65)]

    def render(self, phase):
        phase = float(phase) % 1.0
        rd, th = self.rd, self.th
        acc = np.zeros((self.h, self.w, 3), np.float32)
        wsum = np.zeros((self.h, self.w), np.float32)
        for i, (rad, wd, speed, alpha) in enumerate(self.rings):
            m = F.ring(rd, rad, wd) * np.float32(alpha)
            col = F.sample_strip(self.strip, rd, th, rot=speed * phase,
                                 rad_scale=0.86, rad_offset=0.16)
            # spectre etale entre couronnes + defilement (entier => boucle)
            # La plaque fournit DEJA un spectre etale sur le rayon
            # (mesure : 17 deg au bord -> 261 deg au centre). Ajouter un
            # decalage par couronne le detruisait en ramenant tout vers
            # deux teintes voisines. On ne garde donc que le defilement
            # global, entier => bouclage exact.
            col = F.hue_rotate(col, phase)
            acc += col * m[..., None]
            wsum += m
        rgb = acc / np.maximum(wsum, 1e-4)[..., None]
        rgb *= np.clip(wsum, 0.0, 1.7)[..., None]
        rgb *= F.smoothstep(1.45, 0.60, rd)[..., None]
        # La sphere est posee APRES la vignette et plus lumineuse ici : les
        # couronnes de ce plan sont tres vives et la noyaient completement
        # (elle etait 6x plus sombre que la moyenne de l'image).
        pulse = 0.5 + 0.5 * float(np.cos(TAU * phase))
        rgb = _soul(rgb, self.h, self.w, 0.0, 0.0,
                    0.058 * (1.0 + 0.08 * pulse),
                    glow=2.1 + 0.3 * pulse, halo=1.5)
        return _tone(rgb, 1.02, 1.55)


# --------------------------------------------------------------------------
# S4 — le portail : voyage dans le tunnel
# --------------------------------------------------------------------------
class StatePortal:
    name = "04_portail"
    doc = "Voyage a l'interieur du tunnel d'energie."

    def __init__(self, w, h, plate="research/plates/plate_portal_tunnel.png"):
        self.r = TunnelRenderer(plate, w=w, h=h, params=TunnelParams())

    def render(self, phase):
        return self.r.render(phase)


# --------------------------------------------------------------------------
# transitions — jouées UNE fois, déclenchées par le joueur
# --------------------------------------------------------------------------
def _mix(a, b, u):
    u = float(np.clip(u, 0.0, 1.0))
    u = u * u * (3.0 - 2.0 * u)          # ease in/out : pas d'a-coup
    return a * (1.0 - u) + b * u


class Transition:
    """Fond enchaîné entre deux états, avec un voile optionnel.

    Part de `src` à sa phase 0 et arrive sur `dst` à sa phase 0 : les deux
    boucles se raccordent donc proprement, quel que soit le moment où le
    joueur a appuyé.
    """

    def __init__(self, src, dst, frames, flash=0.0, flash_at=0.55,
                 name="transition", doc=""):
        self.src, self.dst = src, dst
        self.frames, self.flash, self.flash_at = frames, flash, flash_at
        self.name, self.doc = name, doc

    def render(self, i):
        u = i / float(max(1, self.frames - 1))
        a = self.src.render(0.0)
        b = self.dst.render(0.0)
        rgb = _mix(a, b, u)
        if self.flash > 0:
            # voile blanc en cloche, centré sur `flash_at`
            k = np.exp(-((u - self.flash_at) / 0.22) ** 2)
            rgb = rgb + (1.0 - rgb) * np.float32(self.flash * k)
        return np.clip(rgb, 0.0, 1.0)


# --------------------------------------------------------------------------
# construction complète
# --------------------------------------------------------------------------
@dataclass
class BuildSpec:
    w: int = 960
    h: int = 540
    loop_frames: int = 36
    trans_frames: int = 18
    fps: int = 20


def build(out="output/intro", spec: BuildSpec | None = None, gif=True):
    sp = spec or BuildSpec()
    w, h, n = sp.w, sp.h, sp.loop_frames

    s1 = StateVoid(w, h)
    s2 = StateSky(w, h)
    s3 = StateHalo(w, h)
    s4 = StatePortal(w, h)

    trans = [
        Transition(s1, s2, sp.trans_frames, flash=0.0, name="T1_ebranlement",
                   doc="La sphere s'ebranle et part vers le ciel etoile."),
        Transition(s2, s3, sp.trans_frames, flash=0.92, flash_at=0.50,
                   name="T2_flash",
                   doc="Arret du voyage, flash blanc progressif, le halo apparait."),
        Transition(s3, s4, sp.trans_frames, flash=0.35, flash_at=0.62,
                   name="T3_aspiration",
                   doc="Le halo s'ouvre, la sphere est aspiree dans le portail."),
    ]

    os.makedirs(out, exist_ok=True)
    manifest = {"fps": sp.fps, "resolution": f"{w}x{h}",
                "principe": "ANIMATION = BOUCLE INFINIE ; DIALOGUE = PROGRESSION",
                "etats": [], "transitions": []}

    for st in (s1, s2, s3, s4):
        d = os.path.join(out, st.name)
        os.makedirs(d, exist_ok=True)
        ims = []
        for i in range(n):
            a = st.render(i / n)
            im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
            im.save(os.path.join(d, f"frame_{i:03d}.png"), optimize=True)
            ims.append(im)
        exact = np.array_equal(st.render(0.0), st.render(1.0))
        if gif:
            ims[0].save(os.path.join(out, f"{st.name}.gif"), save_all=True,
                        append_images=ims[1:], duration=int(1000 / sp.fps),
                        loop=0, optimize=True)
        manifest["etats"].append({
            "nom": st.name, "description": st.doc, "frames": n, "fps": sp.fps,
            "duree_s": round(n / sp.fps, 3), "type": "boucle infinie",
            "point_de_boucle": f"frame_{n-1:03d} -> frame_000",
            "raccord_exact": bool(exact),
            "sortie": "declenchee par le joueur (touche CONTINUER), jamais par timer",
        })

    for t in trans:
        d = os.path.join(out, t.name)
        os.makedirs(d, exist_ok=True)
        ims = []
        for i in range(t.frames):
            a = t.render(i)
            im = Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))
            im.save(os.path.join(d, f"frame_{i:03d}.png"), optimize=True)
            ims.append(im)
        if gif:
            ims[0].save(os.path.join(out, f"{t.name}.gif"), save_all=True,
                        append_images=ims[1:], duration=int(1000 / sp.fps),
                        loop=0, optimize=True)
        manifest["transitions"].append({
            "nom": t.name, "description": t.doc, "frames": t.frames,
            "fps": sp.fps, "duree_s": round(t.frames / sp.fps, 3),
            "type": "jouee UNE fois",
            "depart": t.src.name + " phase 0", "arrivee": t.dst.name + " phase 0",
            "flash_blanc": t.flash,
            "declencheur": "appui du joueur sur CONTINUER",
        })

    json.dump(manifest, open(os.path.join(out, "sequence.json"), "w"),
              indent=2, ensure_ascii=False)
    return manifest
