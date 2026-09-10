"""Écran du test de personnalité : panneaux DX, fond qui boucle, 2.5D.

Chaque question est un ÉTAT indépendant. Le fond continue de boucler
pendant qu'on réfléchit ; seule la réponse du joueur fait avancer le test.
Aucun timer ne change d'état.

Le texte est découpé en lignes tenant dans la boîte de dialogue, à la
manière de PMD : une boîte basse, large, sur laquelle le texte s'écrit.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import field as F
from . import pixelart as PX
from .dxui import (LINE_HEIGHT, SCREEN_H, SCREEN_W, ChoiceList, MenuFrame,
                   VERT_SPACE, draw_text, new_screen, pmdo_style, text_width,
                   upscale)
from .portalcam import Camera, PortalView
from .quiz import QuizState

TAU = F.TAU


def wrap(texte, largeur_px, spacing=1):
    """Découpe en lignes tenant dans `largeur_px`, sans couper les mots."""
    mots = str(texte).split()
    lignes, cur = [], ""
    for m in mots:
        essai = (cur + " " + m).strip()
        if text_width(essai, spacing) <= largeur_px or not cur:
            cur = essai
        else:
            lignes.append(cur)
            cur = m
    if cur:
        lignes.append(cur)
    return lignes


class QuizScreen:
    """Écran de question. `render` est pur, `step` avance l'état."""

    def __init__(self, w=SCREEN_W, h=SCREEN_H, scale=3, seed=None,
                 quiz: QuizState | None = None, pixel=True, style=None):
        self.w, self.h, self.scale = int(w), int(h), int(scale)
        self.quiz = quiz if quiz is not None else QuizState(seed=seed)
        # UI PMDO PAR DÉFAUT : pendant le test on utilise la fenêtre
        # standard du moteur, pas la transposition DX.
        self.frame = MenuFrame(style or pmdo_style())
        self.list = ChoiceList(self.frame)
        # Le décor des questions, c'est l'INTÉRIEUR DU PORTAIL, vu à la
        # première personne. La caméra est pilotée par le joueur.
        self.cam = Camera()
        self.portal = PortalView(w=self.w, h=self.h, pixel=False)
        self.choix = 0
        self.pixel = bool(pixel)

    # -- entrée joueur -----------------------------------------------------
    def step(self, mouse=(0.0, 0.0)):
        """Souris / stick : oriente le REGARD, rien d'autre.

        Ne renvoie aucune progression et ne touche pas au quiz : bouger la
        caméra ne peut pas répondre à une question. Seul `valider()` avance.
        """
        self.cam.update(*mouse)
        return self

    # alias : le stick droit d'une manette pilote la même caméra
    look_at = step

    def move(self, d):
        q = self.quiz.question
        if q is None:
            return self
        self.choix = int(np.clip(self.choix + d, 0, q.n - 1))
        return self

    def valider(self):
        """SEUL moyen d'avancer. Aucun timer ne le fait a notre place."""
        if not self.quiz.fini:
            self.quiz.answer(self.choix)
            self.choix = 0
        return self

    # -- rendu -------------------------------------------------------------
    def render(self, phase, tick=0):
        # Fond : on est DANS le portail. Il boucle tout seul ; la caméra
        # n'influe que sur l'orientation. Assombri pour que le texte de la
        # boîte de dialogue reste lisible par-dessus.
        buf = self.portal.render(float(phase) % 1.0, self.cam) * 0.55
        s = self.frame.s
        q = self.quiz.question
        if q is None:
            return np.clip(buf, 0.0, 1.0)

        # bandeau de progression : « QUESTION 3 / 8 ». C'est un REPERE, pas
        # un score : aucun point n'apparait nulle part.
        rep = f"QUESTION {self.quiz.numero} / {self.quiz.total}"
        rw = text_width(rep) + 14
        self.frame.panel(buf, self.w - rw - 6, 6, rw, 13,
                         color=s.teal, lo=s.teal_lo, radius=3, hatch=False)
        draw_text(buf, rep, self.w - rw - 1, 10, s.title,
                  outline=s.border_out)

        # boite de dialogue, en bas comme dans PMD
        lignes = wrap(q.texte, self.w - 32)[:3]
        bh = 14 + len(lignes) * 11
        by = self.h - bh - 8 - (q.n * VERT_SPACE + 8)
        self.frame.panel(buf, 8, by, self.w - 16, bh, radius=4)
        for i, ln in enumerate(lignes):
            draw_text(buf, ln, 15, by + 7 + i * 11, s.text,
                      outline=s.text_outline)

        # reponses
        ly = by + bh + 4
        self.list.draw(buf, 8, ly, self.w - 16,
                       [r["label"] for r in q.reponses],
                       index=self.choix, tick=tick)

        buf = np.clip(buf, 0.0, 1.0)
        if self.pixel:
            # tramage APRES composition, sur la grille logique : applique
            # apres agrandissement, il ferait des points de 3x3 pixels.
            buf = PX.dither(buf, levels=14, strength=0.9)
        return buf

    def frame_image(self, phase, tick=0):
        a = self.render(phase, tick)
        return Image.fromarray((upscale(a, self.scale) * 255 + 0.5)
                               .astype(np.uint8))
