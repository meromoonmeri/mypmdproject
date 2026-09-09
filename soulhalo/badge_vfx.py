"""VFX d'apparition du badge : ruée vers le portail, matérialisation, impact.

Enchaînement, joué UNE FOIS sur l'action du joueur :

    A. ruée      — lignes de vitesse, le portail grandit et nous avale
    B. matière   — les motes d'énergie convergent vers le centre
    C. impact    — voile blanc + onde de choc
    D. badge     — le ruban et le badge se posent, l'énergie retombe

Puis la main passe à `idle(phase)`, boucle infinie où le joueur peut rester
aussi longtemps qu'il veut.

RÈGLE STRUCTURANTE — la jointure est exacte, pas approchée :

    burst(1.0) == idle(0.0)      au bit près

C'est ce qui évite le « pop » à la bascule. Un fondu qui *converge* vers la
boucle sans l'atteindre laisse toujours un saut d'une frame, invisible en
GIF mais bien visible en jeu. Ici la garantie est structurelle : le dernier
segment fond vers `idle(0.0)` avec un poids qui vaut exactement 1 à u=1, et
un test le vérifie.

Tout est dessiné dans l'écran logique 320x240 du moteur, puis agrandi d'un
facteur ENTIER — même espace de coordonnées que le sélecteur.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import field as F
from .dxui import MenuFrame, SCREEN_H, SCREEN_W, draw_text, text_width, upscale
from .ribbon import RibbonRenderer, load_natures

TAU = F.TAU
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ss(e0: float, e1: float, x: float) -> float:
    """Smoothstep scalaire, bornée."""
    if abs(e1 - e0) < 1e-9:
        return 0.0 if x < e0 else 1.0
    t = float(np.clip((x - e0) / (e1 - e0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _bell(x: float, c: float, w: float) -> float:
    """Cloche gaussienne, pour une passe qui monte puis redescend."""
    return float(np.exp(-(((x - c) / max(1e-6, w)) ** 2)))


@dataclass
class BurstParams:
    """Minutage du VFX, en fraction de la progression `u`.

    Les valeurs sont ordonnées comme la lecture : ruée, matière, impact,
    dépôt. Les changer décale les passes sans casser la jointure finale,
    qui ne dépend que de `fade_*`.
    """
    # A. ruée vers le portail
    rays_at: float = 0.20
    rays_w: float = 0.16
    rays_gain: float = 1.15
    portal_in: float = 0.45         # le portail a fini de grandir
    portal_out: float = 0.68        # on est dedans, il s'efface
    portal_gain: float = 0.95
    zoom_far: float = 2.05          # échelle de lecture au loin
    zoom_near: float = 0.42         # ... et une fois dessus
    # B. matérialisation
    motes_in: float = 0.30
    motes_out: float = 0.74
    motes_gain: float = 1.30
    motes_far: float = 0.60         # anneau de motes large
    motes_near: float = 2.90        # ... resserré sur le centre
    motes_spin: float = -1.25
    # C. impact
    flash_at: float = 0.70
    flash_w: float = 0.05
    flash_gain: float = 0.92
    shock_at: float = 0.735
    shock_w: float = 0.055
    shock_gain: float = 1.45
    shock_from: float = 3.00        # onde serrée...
    shock_to: float = 0.75          # ... qui s'ouvre
    # D. dépôt du badge
    settle_in: float = 0.66
    settle_out: float = 0.90
    fade_in: float = 0.90           # début du fondu vers la boucle
    fade_out: float = 1.00          # jointure exacte
    # UI DX
    panel: bool = True


class BadgeReveal:
    """Apparition du badge : un one-shot qui atterrit sur une boucle.

    `burst(u)` joue la séquence, `idle(phase)` tourne indéfiniment. Les deux
    partagent le même ruban et le même badge, donc la bascule est invisible.
    """

    def __init__(self, w=SCREEN_W, h=SCREEN_H, scale=3, nature="jolly",
                 params: BurstParams | None = None, nom: str | None = None):
        self.w, self.h, self.scale = int(w), int(h), int(scale)
        self.p = params or BurstParams()
        self.natures = load_natures()
        # Le ruban travaille a la MEME resolution logique : il porte deja le
        # badge, sa teinte de nature et sa boucle. Le reutiliser evite de
        # redecouper et reteinter le badge une seconde fois - et garantit
        # que la boucle d'arrivee est exactement la sienne.
        self.ribbon = RibbonRenderer(w=self.w, h=self.h, nature=nature,
                                     natures=self.natures)
        self.frame_ui = MenuFrame()
        self.set_nature(nature, nom)

        self.rn, self.th = F.polar(self.h, self.w)
        diag = float(np.hypot(self.h, self.w)) / min(self.h, self.w)
        self.rd = (self.rn / np.float32(diag)).astype(np.float32)
        self._strips = {}

    # -- plaques -----------------------------------------------------------
    def _strip(self, name):
        """Bande polaire d'une plaque, mise en cache."""
        if name not in self._strips:
            path = os.path.join(_HERE, "research/plates", name)
            self._strips[name] = F.polar_strip(path, n_theta=768, n_rad=384)
        return self._strips[name]

    def set_nature(self, nature: str, nom: str | None = None):
        self.nature = nature
        self.ribbon.set_nature(nature)
        n = self.natures[nature]
        self.tint = np.array(n["rgb"], np.float32)
        self.accent = np.array(n["accent"], np.float32)
        self.nature_fr = n.get("fr", nature)
        self.nom = nom
        return self

    # -- passes ------------------------------------------------------------
    def _rays(self, u, phase):
        """Lignes de vitesse : la ruée vers le portail.

        Les stries balaient vers l'EXTERIEUR (échelle de lecture qui
        diminue) : c'est ce qui donne la sensation d'avancer, et non de
        reculer.
        """
        s = self.p.zoom_far * (1.0 - u) + 0.55 * u
        col = F.sample_strip(self._strip("plate_rays.png"), self.rd, self.th,
                             rot=0.35 * phase, rad_scale=max(0.2, s))
        return col

    def _portal(self, u, phase):
        """Le portail grandit jusqu'à remplir l'écran."""
        k = _ss(0.0, self.p.portal_in, u)
        s = self.p.zoom_far * (1.0 - k) + self.p.zoom_near * k
        col = F.sample_strip(self._strip("plate_portal_tunnel.png"),
                             self.rd, self.th, rot=0.20 * phase,
                             rad_scale=max(0.2, s), rad_offset=0.04)
        return col

    def _motes(self, u, phase):
        """Motes d'énergie qui convergent : la matérialisation.

        L'échelle de lecture AUGMENTE, ce qui tire l'anneau brillant de la
        plaque vers le centre de l'écran. Les motes se rassemblent là où le
        badge va naître.
        """
        k = _ss(self.p.motes_in, self.p.motes_out, u)
        s = self.p.motes_far * (1.0 - k) + self.p.motes_near * k
        col = F.sample_strip(self._strip("plate_motes.png"), self.rd, self.th,
                             rot=self.p.motes_spin * u, rad_scale=max(0.2, s))
        return col

    def _shock(self, u, phase):
        """Onde de choc : anneaux qui s'ouvrent depuis le centre."""
        k = _ss(self.p.shock_at - self.p.shock_w * 2.0,
                self.p.shock_at + self.p.shock_w * 3.0, u)
        s = self.p.shock_from * (1.0 - k) + self.p.shock_to * k
        col = F.sample_strip(self._strip("plate_shockwave.png"),
                             self.rd, self.th, rot=0.0, rad_scale=max(0.2, s))
        return col

    # -- UI DX -------------------------------------------------------------
    def _panel(self, rgb, alpha=1.0):
        """Bandeau DX : cadre du moteur, fonte bitmap, sous le badge."""
        if not self.p.panel or alpha <= 0.001:
            return rgb
        buf = rgb if alpha >= 0.999 else rgb.copy()
        s = self.frame_ui.s
        pw, ph = 176, 34
        x = (self.w - pw) // 2
        y = self.h - ph - 10
        self.frame_ui.draw(buf, x, y, pw, ph, title=None, divider=False)
        lbl = self.nature_fr.upper()
        draw_text(buf, "NATURE", x + 6, y + 5, s.title)
        draw_text(buf, lbl, x + pw - 6 - text_width(lbl), y + 5, self.tint)
        if self.nom:
            draw_text(buf, self.nom.upper(), x + 6, y + 19, s.text)
        if alpha < 0.999:
            rgb[:] = rgb * (1.0 - alpha) + buf * alpha
            return rgb
        return buf

    # -- rendu -------------------------------------------------------------
    def idle(self, phase: float) -> np.ndarray:
        """Boucle infinie, après l'apparition. Le joueur peut y rester."""
        rgb = self.ribbon.idle(float(phase) % 1.0).copy()
        return np.clip(self._panel(rgb, 1.0), 0.0, 1.0)

    def burst(self, u: float) -> np.ndarray:
        """Séquence jouée UNE FOIS. `u` va de 0 à 1.

        `u` n'est PAS un temps : c'est la progression de l'effet. Le jeu
        l'avance frame par frame après l'action du joueur, exactement comme
        les transitions de la séquence d'intro.
        """
        p = self.p
        u = float(np.clip(u, 0.0, 1.0))
        phase = u                      # animation locale pendant la passe

        # D'abord la cible : le ruban tel qu'il sera une fois posé. `t` est
        # la charge du ruban - il arrive gonfle d'energie puis se calme.
        mat = _ss(p.settle_in, p.settle_out, u)
        t = 1.0 - _ss(p.settle_in, 1.0, u)
        rgb = self.ribbon.charge(t, phase) * np.float32(mat)

        # A. ruée
        wr = _bell(u, p.rays_at, p.rays_w) * p.rays_gain
        if wr > 0.004:
            rgb += self._rays(u, phase) * np.float32(wr)
        wp = _ss(0.0, p.portal_in * 0.7, u) * (1.0 - _ss(p.portal_in,
                                                         p.portal_out, u))
        if wp > 0.004:
            rgb += self._portal(u, phase) * np.float32(wp * p.portal_gain)

        # B. matérialisation
        wm = _ss(p.motes_in, p.motes_in + 0.22, u) * \
            (1.0 - _ss(p.motes_out - 0.10, p.motes_out, u))
        if wm > 0.004:
            m = self._motes(u, phase)
            # les motes prennent la couleur de la nature en approchant
            m = m * (1.0 - 0.55 * mat) + m.mean(2, keepdims=True) * \
                self.tint * np.float32(0.55 * mat)
            rgb += m * np.float32(wm * p.motes_gain)

        # C. impact
        ws = _bell(u, p.shock_at, p.shock_w) * p.shock_gain
        if ws > 0.004:
            rgb += self._shock(u, phase) * np.float32(ws)

        rgb = np.clip(rgb, 0.0, None)
        rgb = rgb / (1.0 + rgb * 0.38)          # tone map, garde les cœurs

        wf = _bell(u, p.flash_at, p.flash_w) * p.flash_gain
        if wf > 0.004:
            white = np.array([1.0, 0.995, 0.97], np.float32)
            rgb = rgb * (1.0 - wf) + white * wf

        # D. bandeau DX, puis jointure EXACTE avec la boucle
        rgb = self._panel(rgb, _ss(p.settle_out, 1.0, u))
        k = _ss(p.fade_in, p.fade_out, u)
        if k > 0.0:
            rgb = rgb * (1.0 - k) + self.idle(0.0) * k
        return np.clip(rgb, 0.0, 1.0)

    # -- images ------------------------------------------------------------
    def burst_image(self, u):
        return self._img(self.burst(u))

    def idle_image(self, phase):
        return self._img(self.idle(phase))

    def _img(self, a):
        return Image.fromarray((upscale(a, self.scale) * 255.0 + 0.5)
                               .astype(np.uint8))
