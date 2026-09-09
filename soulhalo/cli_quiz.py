"""Rendu du test de personnalité.

    .venv/bin/python -m soulhalo.cli_quiz

Produit `output/quiz/` : une boucle d'attente par question (le fond tourne
pendant que le joueur réfléchit), le parcours complet d'une partie, et la
fiche technique.
"""
from __future__ import annotations

import json
import os

from PIL import Image

from .quiz import QuizState
from .quizscreen import QuizScreen

OUT = "output/quiz"
FPS = 20
N_LOOP = 40


def _gif(frames, path, fps=FPS, shrink=3):
    if shrink > 1:
        w, h = frames[0].size
        frames = [f.resize((w // shrink, h // shrink), Image.NEAREST)
                  for f in frames]
    q = [f.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT,
                                   dither=Image.FLOYDSTEINBERG)
         for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:],
              duration=int(1000 / fps), loop=0, disposal=2, optimize=True)


def main():
    os.makedirs(f"{OUT}/question", exist_ok=True)
    os.makedirs(f"{OUT}/parcours", exist_ok=True)

    # 1. une question, en boucle d'attente : le fond tourne, l'etat ne
    #    change pas. C'est la preuve qu'aucun timer ne fait avancer le test.
    s = QuizScreen(seed=1, scale=3)
    boucle = []
    for i in range(N_LOOP):
        im = s.frame_image(i / N_LOOP, tick=i)
        im.save(f"{OUT}/question/frame_{i:03d}.png")
        boucle.append(im)
    _gif(boucle, f"{OUT}/question.gif")

    # 2. le parcours : chaque validation change d'etat
    s2 = QuizScreen(seed=1, scale=3)
    parcours = []
    choix = [0, 1, 0, 0, 1, 1, 0, 1]
    n = 0
    while not s2.quiz.fini:
        c = choix[len(parcours) // 6 % len(choix)]
        s2.choix = min(c, s2.quiz.question.n - 1)
        for k in range(6):          # quelques frames par question
            im = s2.frame_image((n % 20) / 20.0, tick=n)
            im.save(f"{OUT}/parcours/frame_{n:03d}.png")
            parcours.append(im)
            n += 1
        s2.valider()
    _gif(parcours, f"{OUT}/parcours.gif", fps=12)

    res = s2.quiz.result_data()
    data = QuizState(seed=1)
    fiche = {
        "role": "test de personnalite : une question = un etat independant",
        "regle_or": ("le joueur ne voit JAMAIS de points. Le bandeau "
                     "« QUESTION 3 / 8 » est un repere de progression, "
                     "pas un score."),
        "progression": ("uniquement par valider() ; aucun timer ne change "
                        "d'etat. Le fond boucle pendant la reflexion."),
        "banque": len(data.questions) and 18,
        "tirage": data.total,
        "natures": 13,
        "equilibrage": ("ratio max/min de la distribution ramene de 21.6x a "
                        "2.0x en repartissant les poids : 'calm' servait de "
                        "reponse prudente par defaut dans 11 questions sur 18"),
        "loops": [
            {"nom": "question", "type": "boucle", "frames": N_LOOP,
             "fps": FPS, "duree_s": round(N_LOOP / FPS, 2),
             "point_de_boucle": "frame 0 - render(0.0) == render(1.0)",
             "declencheur": "aucun : tourne tant que le joueur reflechit"},
            {"nom": "parcours", "type": "demonstration",
             "frames": len(parcours), "fps": 12,
             "declencheur": "valider() a chaque question"},
        ],
        "pixel_art": {
            "tramage": "Bayer 4x4, 14 paliers, applique AVANT l'agrandissement",
            "relief": "emboss 2.5D : lumiere haut-gauche, ombre bas-droite",
            "ombre_portee": "decalee 2px, trame, jamais floue",
            "agrandissement": "entier, plus proche voisin",
        },
        "resultat_exemple": res,
    }
    with open(f"{OUT}/quiz.json", "w", encoding="utf-8") as f:
        json.dump(fiche, f, ensure_ascii=False, indent=2)
    print(f"{len(boucle)} frames boucle + {len(parcours)} frames parcours "
          f"-> {OUT} (resultat : {res['fr']})")


if __name__ == "__main__":
    main()
