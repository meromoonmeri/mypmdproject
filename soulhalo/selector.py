"""
Écran de sélection complet : fond de rêve + carrousel DX + talent/nature.

Assemble les briques : le fond de `interactive` continue de boucler, la
molette fait défiler les vignettes de `dxui`, les paillettes accompagnent
le mouvement.

Le temps (phase) et l'entrée joueur (molette, curseur) restent séparés :
faire tourner la molette ne fait jamais avancer l'animation de fond.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import field as F
from .dxui import (PORTRAIT_SIZE, ChoiceList, MenuFrame, PokemonCard,
                   SpriteBank, WheelState, draw_text, load_roster, new_screen,
                   text_width, upscale)
from .interactive import DreamBackdrop, LookState

TAU = F.TAU


class SelectorScreen:
    """Écran de choix : 42 Pokémon, molette, paillettes, talent et nature."""

    def __init__(self, w=320, h=240, scale=3,
                 roster_path="personality_test/types.json"):
        self.w, self.h, self.scale = w, h, scale
        self.data = load_roster(roster_path)
        self.roster = self.data["roster"]
        self.types = self.data["types"]
        self.bank = SpriteBank()
        self.frame = MenuFrame()
        self.card = PokemonCard(self.bank, self.frame)
        self.list = ChoiceList(self.frame)
        self.wheel = WheelState(n=len(self.roster))
        self.look = LookState()
        # le fond de reve tourne a la resolution logique
        self.dream = DreamBackdrop(w=w, h=h)
        self.talent_idx = 0
        self.nature_idx = 0
        self.natures = ["Hardi", "Brave", "Jovial", "Docile", "Calme",
                        "Timide", "Malicieux", "Bizarre"]

    # -- paillettes --------------------------------------------------------
    def _sparkles(self, buf, phase, intensity):
        """Paillettes qui suivent le defilement.

        Elles n'apparaissent qu'en mouvement : `intensity` vient de la
        vitesse de la molette. Un scintillement permanent fatiguerait l'oeil.
        """
        if intensity <= 0.01:
            return buf
        rng = np.random.default_rng(7)          # graine FIXE : pas de bruit
        n = 40
        xs = rng.random(n)
        ys = rng.random(n)
        ph = rng.random(n)
        yy, xx = np.indices((self.h, self.w)).astype(np.float32)
        add = np.zeros((self.h, self.w, 3), np.float32)
        for i in range(n):
            t = (phase + ph[i]) % 1.0
            cx = (xs[i] * 1.2 - 0.1) * self.w
            cy = ((ys[i] + t * 0.35) % 1.0) * self.h
            tw = 0.5 + 0.5 * np.cos(TAU * (2 * t))
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            add += np.exp(-d2 / 6.0)[..., None] * np.float32(tw * 0.9)
        return np.clip(buf + add * float(intensity), 0.0, 1.0)

    # -- rendu -------------------------------------------------------------
    def step(self, mouse=(0.0, 0.0)):
        """Avance l'etat d'une frame : souris et inertie de la molette.

        SEPARE de `render` a dessein. Un rendu qui modifie l'etat rendrait
        deux appels a la meme phase differents, et l'animation ne bouclerait
        plus - c'est exactement ce qu'un test a detecte.
        """
        self.look.update(*mouse)
        self.wheel.update()
        return self

    def render(self, phase, tick=0, mouse=None):
        if mouse is not None:
            self.step(mouse)
        buf = self.dream.render(phase, self.look) * 0.55

        pos = self.wheel.pos
        sel = self.wheel.index
        cw = self.card.p.w
        gap = 8
        cy = 16
        # vignettes centrees sur la position continue
        for k in range(sel - 3, sel + 4):
            if not (0 <= k < len(self.roster)):
                continue
            off = k - pos
            if abs(off) > 3.2:
                continue
            cx = int(self.w / 2 + off * (cw + gap) - cw / 2)
            mon = self.roster[k]
            self.card.draw(buf, mon["dex"], cx, cy, tick=tick,
                           selected=(k == sel), hover=1.0 if k == sel else 0.0,
                           tint=self.types[mon["type"]]["rgb"])

        buf = self._sparkles(buf, phase, min(1.0, abs(self.wheel.vel) * 1.6))

        mon = self.roster[sel]
        s = self.frame.s
        # nom + type
        draw_text(buf, mon["nom"], 6, 124, s.title)
        tcol = self.types[mon["type"]]["rgb"]
        draw_text(buf, self.types[mon["type"]]["fr"], self.w - 6 -
                  text_width(self.types[mon["type"]]["fr"]), 124, tcol)
        # talent et nature
        self.list.draw(buf, 6, 136, 148, mon["talents"],
                       index=min(self.talent_idx, len(mon["talents"]) - 1),
                       title="Talent", tick=tick)
        self.list.draw(buf, 162, 136, 152, self.natures[:4],
                       index=self.nature_idx % 4, title="Nature", tick=tick)
        return np.clip(buf, 0.0, 1.0)

    def frame_image(self, phase, tick=0, mouse=None):
        a = self.render(phase, tick, mouse)
        return Image.fromarray((upscale(a, self.scale) * 255 + 0.5).astype(np.uint8))
