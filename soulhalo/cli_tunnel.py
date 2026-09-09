"""Ligne de commande : voyage dans le portail."""
from __future__ import annotations
import argparse, os, time
from .tunnel import TunnelRenderer, TunnelParams


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="soulhalo.cli_tunnel",
        description="Portail : la sphere-ame voyage dans un tunnel d'energie.")
    p.add_argument("--plate", default="research/plates/plate_portal_tunnel.png")
    p.add_argument("--out", default="output/portal")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--tile-octaves", type=float, default=4.5,
                   help="octaves de rayon par tuile (bas = bandes serrees)")
    p.add_argument("--core-glow", type=float, default=0.55)
    p.add_argument("--tint-shift", type=int, default=0,
                   help="cycles de teinte par boucle - ENTIER")
    p.add_argument("--saturation", type=float, default=1.45)
    p.add_argument("--bloom", type=float, default=0.16)
    p.add_argument("--no-soul", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-sheet", action="store_true")
    a = p.parse_args(argv)

    prm = TunnelParams(tile_octaves=a.tile_octaves, core_glow=a.core_glow,
                       tint_shift=a.tint_shift, saturation=a.saturation,
                       bloom=a.bloom, soul=not a.no_soul)
    os.makedirs(a.out, exist_ok=True)
    fdir = os.path.join(a.out, "frames"); os.makedirs(fdir, exist_ok=True)

    t0 = time.time()
    r = TunnelRenderer(a.plate, w=a.width, h=a.height, params=prm)
    frames = []
    for i in range(a.frames):
        im = r.frame(i / float(a.frames))
        im.save(os.path.join(fdir, f"frame_{i:03d}.png"), optimize=True)
        frames.append(im)

    if not a.no_gif:
        frames[0].save(os.path.join(a.out, "portal.gif"), save_all=True,
                       append_images=frames[1:], duration=int(1000 / a.fps),
                       loop=0, optimize=True)
    if not a.no_sheet:
        from PIL import Image
        cols, tw = 4, 240
        th = int(tw * a.height / a.width)
        rows = (len(frames) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * th))
        for i, im in enumerate(frames):
            sheet.paste(im.resize((tw, th), Image.LANCZOS),
                        ((i % cols) * tw, (i // cols) * th))
        sheet.save(os.path.join(a.out, "contact.png"), optimize=True)

    print(f"[OK] {a.frames} frames {a.width}x{a.height} -> {a.out} "
          f"en {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
