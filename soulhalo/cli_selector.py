"""Rendu de l'écran de sélection : défilement à la molette + choix.

    .venv/bin/python -m soulhalo.cli_selector

Produit `output/selector/` : les frames du défilement, celles du choix de
nature, les deux GIF et la fiche technique JSON.

Le scénario joué est une SUITE D'ENTREES JOUEUR, pas une animation pilotée
par le temps : on donne des coups de molette, puis on descend dans la liste
des natures. Entre deux entrées, seule la phase avance — et elle boucle.
"""
from __future__ import annotations

import json
import os

from .dxui import load_roster
from .selector import SelectorScreen

OUT = "output/selector"
FPS = 20


def _save_gif(frames, path, fps=FPS):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, disposal=2)


def render_scroll(s, n=48, scale=3):
    """Défilement : trois coups de molette, l'inertie fait le reste."""
    os.makedirs(f"{OUT}/scroll", exist_ok=True)
    frames = []
    for i in range(n):
        # les coups de molette arrivent à des moments choisis, comme un
        # joueur qui fait défiler ; le reste est de la glisse.
        if i in (2, 8, 14, 22, 30):
            s.wheel.scroll(2)
        s.step()
        phase = i / n                      # la phase boucle sur la séquence
        img = s.frame_image(phase, tick=i)
        img.save(f"{OUT}/scroll/frame_{i:03d}.png")
        frames.append(img)
    _save_gif(frames, f"{OUT}/selector.gif")
    return frames


def render_choice(s, n=36, scale=3):
    """Choix : on descend dans les 13 natures, la liste défile."""
    os.makedirs(f"{OUT}/choice", exist_ok=True)
    frames = []
    for i in range(n):
        if i > 0 and i % 3 == 0 and s.nature_idx < 12:
            s.move_nature(1)
        s.step()
        img = s.frame_image(i / n, tick=i)
        img.save(f"{OUT}/choice/frame_{i:03d}.png")
        frames.append(img)
    _save_gif(frames, f"{OUT}/choice.gif")
    return frames


def main():
    os.makedirs(OUT, exist_ok=True)
    s = SelectorScreen()
    scroll = render_scroll(s)
    s.wheel.pos = 9.0                      # on s'arrête sur un Pokémon
    s.wheel.vel = 0.0
    choice = render_choice(s)

    roster = load_roster()["roster"]
    with open("personality_test/natures.json", encoding="utf-8") as f:
        natures = json.load(f)["natures"]

    fiche = {
        "moteur": ("parametres PMDO reels (RogueCollab/RogueEssence) : "
                   "VERT_SPACE=14, LINE_HEIGHT=12, TITLE_OFFSET=16, "
                   "PortraitSize=40"),
        "ecran_logique": "320x240 puis agrandissement ENTIER au plus proche voisin",
        "roster": len(roster),
        "natures": len(natures),
        "loops": [
            {"nom": "scroll", "frames": len(scroll), "fps": FPS,
             "duree_s": round(len(scroll) / FPS, 2),
             "boucle": "phase 0 -> 1, render(0.0) == render(1.0)",
             "entree": "molette (WheelState, friction 0.82, pos continu)",
             "effet": "paillettes proportionnelles a |vel|, graine fixe"},
            {"nom": "choice", "frames": len(choice), "fps": FPS,
             "duree_s": round(len(choice) / FPS, 2),
             "boucle": "phase 0 -> 1",
             "entree": "curseur nature, fenetre defilante de 4 lignes sur 13",
             "effet": "curseur clignotant de periode entiere (tick % 30)"},
        ],
        "vignette": "portrait 40x40 encadre AU-DESSUS, sprite anime en dessous",
        "sprites": "SpriteCollab, durees lues dans AnimData.xml",
        "separation": "step() avance l'etat, render() est pur",
        "selection": "espece + talent + nature -> SelectorScreen.selection()",
    }
    with open(f"{OUT}/selector.json", "w", encoding="utf-8") as f:
        json.dump(fiche, f, ensure_ascii=False, indent=2)
    print(f"{len(scroll)} frames scroll + {len(choice)} frames choice -> {OUT}")


if __name__ == "__main__":
    main()
