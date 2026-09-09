"""Ligne de commande : rend la parallaxe circulaire en PNG / planche / GIF."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from PIL import Image

from .halo import HaloParams, HaloRenderer, SoulParams, default_layers


def build_params(a) -> HaloParams:
    layers = default_layers()
    if a.speed != 1.0:
        layers = [replace(L, speed=int(round(L.speed * a.speed)) or (1 if L.speed > 0 else -1))
                  for L in layers]
    soul = SoulParams(orbit_rx=a.orbit, orbit_ry=a.orbit * a.tilt,
                      orbit_turns=a.orbit_turns, radius=a.soul,
                      trail=a.trail)
    return HaloParams(layers=layers, soul=soul, saturation=a.saturation,
                      grain=a.grain, bloom=a.bloom,
                      rainbow=a.rainbow, rainbow_spread=a.rainbow_spread,
                      plate_offset=a.plate_offset, core_dark=a.core_dark)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="soulhalo",
        description="Parallaxe circulaire : halo multicolore et sphere-ame qui voyage.")
    p.add_argument("--plate", default="research/plates/plate_nebula.png",
                   help="plaque aquarelle peinte (carree)")
    p.add_argument("--out", default="output/soulhalo", help="dossier de sortie")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.0,
                   help="multiplicateur des vitesses de couches")
    p.add_argument("--orbit", type=float, default=0.46, help="rayon de l'orbite")
    p.add_argument("--tilt", type=float, default=0.65,
                   help="ecrasement vertical de l'orbite (perspective)")
    p.add_argument("--orbit-turns", type=int, default=1,
                   help="tours d'orbite par boucle (entier)")
    p.add_argument("--soul", type=float, default=0.085, help="rayon du noyau")
    p.add_argument("--trail", type=int, default=26, help="longueur de trainee")
    p.add_argument("--saturation", type=float, default=1.85)
    p.add_argument("--grain", type=float, default=0.030)
    p.add_argument("--bloom", type=float, default=0.30)
    p.add_argument("--rainbow", type=int, default=1,
                   help="cycles d'arc-en-ciel par boucle - ENTIER (0 = eteint)")
    p.add_argument("--rainbow-spread", type=float, default=1.0,
                   help="etalement du spectre entre les couronnes")
    p.add_argument("--plate-offset", type=float, default=0.0,
                   help="decale la lecture radiale (evite un coeur sombre)")
    p.add_argument("--core-dark", type=float, default=0.30,
                   help="rayon du puits sombre central")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--sheet", action="store_true", help="planche contact 4xN")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    fdir = os.path.join(a.out, "frames")
    os.makedirs(fdir, exist_ok=True)

    r = HaloRenderer(a.plate, w=a.width, h=a.height, params=build_params(a))
    frames = []
    for i in range(a.frames):
        im = r.frame(i / float(a.frames))
        im.save(os.path.join(fdir, f"frame_{i:03d}.png"), optimize=True)
        frames.append(im)

    if not a.no_gif:
        frames[0].save(os.path.join(a.out, "soulhalo.gif"), save_all=True,
                       append_images=frames[1:], duration=int(1000 / a.fps),
                       loop=0, optimize=True)
    if a.sheet:
        cols = 4
        rows = (len(frames) + cols - 1) // cols
        w, h = frames[0].size
        b = Image.new("RGB", (w * cols, h * rows))
        for i, f in enumerate(frames):
            b.paste(f, (i % cols * w, i // cols * h))
        b.save(os.path.join(a.out, "contact.png"), optimize=True)

    print(f"[OK] {a.frames} frames {a.width}x{a.height} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
