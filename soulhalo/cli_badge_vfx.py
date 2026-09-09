"""Rendu du VFX d'apparition du badge.

    .venv/bin/python -m soulhalo.cli_badge_vfx

Produit `output/badge_vfx/` :
    burst/    les frames de l'apparition, jouees UNE FOIS
    idle/     la boucle infinie qui suit
    full.gif  les deux bout a bout, tel que le joueur le voit
    natures/  la meme apparition dans plusieurs natures
    badge_vfx.json  la fiche technique, loop par loop
"""
from __future__ import annotations

import json
import os

from PIL import Image

from .badge_vfx import BadgeReveal

OUT = "output/badge_vfx"
FPS = 24
N_BURST = 48
N_IDLE = 40


def _gif(frames, path, fps=FPS, shrink=3):
    """GIF quantifie.

    Sans quantification explicite, Pillow ecrit un GIF enorme (26 Mo pour
    2 secondes). Le format ne porte de toute facon que 256 couleurs : on
    choisit la palette nous-memes, et on reduit la taille d'apercu. Les
    frames PNG pleine resolution restent la source.
    """
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
    os.makedirs(f"{OUT}/burst", exist_ok=True)
    os.makedirs(f"{OUT}/idle", exist_ok=True)
    os.makedirs(f"{OUT}/natures", exist_ok=True)

    r = BadgeReveal(nature="jolly", nom="Evoli", scale=3)

    # apparition : u va de 0 a 1, une seule fois
    burst = []
    for i in range(N_BURST):
        u = i / (N_BURST - 1)
        im = r.burst_image(u)
        im.save(f"{OUT}/burst/frame_{i:03d}.png")
        burst.append(im)
    _gif(burst, f"{OUT}/burst.gif")

    # boucle : phase de 0 a 1, indefiniment
    idle = []
    for i in range(N_IDLE):
        im = r.idle_image(i / N_IDLE)
        im.save(f"{OUT}/idle/frame_{i:03d}.png")
        idle.append(im)
    _gif(idle, f"{OUT}/idle.gif")

    # ce que voit le joueur : l'apparition PUIS la boucle, sans couture
    _gif(burst + idle + idle, f"{OUT}/full.gif", fps=20)

    # la meme apparition selon la nature
    for nat, nom in (("hardy", "Salameche"), ("calm", "Carapuce"),
                     ("impish", "Zorua"), ("quirky", "Evoli")):
        r2 = BadgeReveal(nature=nat, nom=nom, scale=2)
        fr = [r2.burst_image(i / 23) for i in range(24)]
        _gif(fr, f"{OUT}/natures/{nat}.gif")

    p = r.p
    fiche = {
        "role": ("apparition du badge de resultat : ruee vers le portail, "
                 "materialisation d'energie, impact, depot du badge"),
        "ecran_logique": "320x240, agrandissement ENTIER (PMDO)",
        "regle": ("burst() est joue UNE FOIS sur l'action du joueur ; "
                  "idle() boucle ensuite indefiniment. Aucun timer ne "
                  "declenche la bascule."),
        "jointure": "burst(1.0) == idle(0.0) au bit pres, verifie par test",
        "loops": [
            {
                "nom": "burst", "type": "one-shot", "frames": N_BURST,
                "fps": FPS, "duree_s": round(N_BURST / FPS, 2),
                "debut": 0, "fin": N_BURST - 1,
                "point_de_boucle": "aucun - enchaine sur idle a la frame finale",
                "declencheur": "touche CONTINUER / validation du joueur",
                "couches": [
                    {"nom": "rays", "plaque": "plate_rays.png",
                     "role": "lignes de vitesse, ruee vers le portail",
                     "pic_u": p.rays_at, "gain": p.rays_gain,
                     "echelle": f"{p.zoom_far} -> 0.55 (balayage vers l'exterieur)"},
                    {"nom": "portal", "plaque": "plate_portal_tunnel.png",
                     "role": "le portail grandit puis nous avale",
                     "entree_u": 0.0, "plein_u": p.portal_in,
                     "sortie_u": p.portal_out,
                     "echelle": f"{p.zoom_far} -> {p.zoom_near}"},
                    {"nom": "motes", "plaque": "plate_motes.png",
                     "role": "materialisation : l'energie converge au centre",
                     "entree_u": p.motes_in, "sortie_u": p.motes_out,
                     "echelle": f"{p.motes_far} -> {p.motes_near}",
                     "rotation_tours": p.motes_spin,
                     "teinte": "vire vers la couleur de la nature en approchant"},
                    {"nom": "flash", "role": "voile blanc d'impact",
                     "pic_u": p.flash_at, "largeur": p.flash_w,
                     "gain": p.flash_gain},
                    {"nom": "shockwave", "plaque": "plate_shockwave.png",
                     "role": "onde de choc qui s'ouvre",
                     "pic_u": p.shock_at, "largeur": p.shock_w,
                     "echelle": f"{p.shock_from} -> {p.shock_to}"},
                    {"nom": "ribbon+badge", "source": "soulhalo.ribbon",
                     "role": "le ruban et le badge oeuf-et-ailes se posent",
                     "entree_u": p.settle_in, "pose_u": p.settle_out,
                     "charge": "1 -> 0 : arrive gonfle d'energie puis se calme"},
                    {"nom": "panel_dx", "role": "bandeau nom + nature",
                     "entree_u": p.settle_out,
                     "style": "MenuFrame du moteur, fonte bitmap 5x7"},
                ],
            },
            {
                "nom": "idle", "type": "boucle", "frames": N_IDLE,
                "fps": FPS, "duree_s": round(N_IDLE / FPS, 2),
                "debut": 0, "fin": N_IDLE - 1,
                "point_de_boucle": "frame 0 - render(0.0) == render(1.0)",
                "declencheur": "aucun : tourne tant que le joueur reste",
                "couches": [
                    {"nom": "ribbon", "role": "ruban qui tourne, respire",
                     "tours_par_boucle": "entiers (spin 1, contre-spin -2)"},
                    {"nom": "badge", "role": "flottement 1 cycle par boucle"},
                    {"nom": "panel_dx", "role": "bandeau nom + nature"},
                ],
            },
        ],
        "couleur": "pilotee par la nature (natures.json), badge et motes teintes",
    }
    with open(f"{OUT}/badge_vfx.json", "w", encoding="utf-8") as f:
        json.dump(fiche, f, ensure_ascii=False, indent=2)
    print(f"{len(burst)} frames burst + {len(idle)} frames idle -> {OUT}")


if __name__ == "__main__":
    main()
