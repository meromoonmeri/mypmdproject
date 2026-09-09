"""
Ligne de commande de pmdmax.

    python -m pmdmax build    <pokemon> [options]   sprite Dynamax complet
    python -m pmdmax fx       [options]             planche FX fond magenta
    python -m pmdmax verify   <dossier>             validation stricte
    python -m pmdmax preview  <dossier> [options]   GIF / planche de contact
    python -m pmdmax ase      <dossier> <anim>      export Aseprite
    python -m pmdmax list                           Pokémon disponibles
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import aseprite as ASE
from . import beam as BM
from . import config as C
from . import dynamax as DX
from . import preview as PV
from . import sheet as SH
from . import transform as TR
from . import verify as VF

DEFAULT_SRC_ROOT = os.path.join(C.REPO, "external", "SpriteCollab", "sprite")
DEFAULT_OUT_ROOT = os.path.join(C.REPO, "output")


def _resolve_src(pokemon: str, root: str) -> str:
    """Accepte '0025', '25', 'pikachu' ou un chemin direct."""
    if os.path.isdir(pokemon) and os.path.isfile(
            os.path.join(pokemon, C.MULTI_SHEET_XML)):
        return pokemon
    if pokemon.isdigit():
        cand = os.path.join(root, pokemon.zfill(4))
        if os.path.isdir(cand):
            return cand
    cand = os.path.join(root, pokemon)
    if os.path.isdir(cand):
        return cand
    raise SystemExit(
        f"Sprite introuvable : '{pokemon}'.\n"
        f"Cherché dans {root}. Lance tools/fetch_upstream.sh pour récupérer "
        f"des Pokémon depuis SpriteCollab.")


def _params_from_args(a: argparse.Namespace) -> DX.DynamaxParams:
    return DX.DynamaxParams(
        scale=a.scale,
        scale_mode=a.scale_mode,
        cloud_count=a.clouds,
        cloud_scale=a.cloud_scale,
        cloud_gap=a.cloud_gap,
        orbit_rx=a.orbit_rx,
        bolts=not a.no_bolts,
        seed=a.seed,
        revolutions=a.revolutions,
        aura=not a.no_aura,
        aura_reach=a.aura_reach,
        aura_strength=a.aura_strength,
        aura_cycles=a.aura_cycles,
    )


# --------------------------------------------------------------------------

def cmd_build(a: argparse.Namespace) -> int:
    src = _resolve_src(a.pokemon, a.src_root)
    name = os.path.basename(src.rstrip("/"))
    out = a.out or os.path.join(DEFAULT_OUT_ROOT, f"{name}-dynamax")
    params = _params_from_args(a)

    print(f"Source      : {src}")
    print(f"Sortie      : {out}")
    print(f"Échelle     : x{params.scale} ({params.scale_mode})")
    print(f"Nuages      : {params.cloud_count}\n")

    only: Optional[List[str]] = a.anims.split(",") if a.anims else None
    report = DX.dynamax_folder(src, out, params, only=only)

    # --- animation de transformation ------------------------------------
    if not a.no_transform:
        data = SH.read_anim_data(os.path.join(src, C.MULTI_SHEET_XML))
        base_entry = data.get(a.transform_base) or data.get("Idle") or \
            next((x for x in data.sorted_anims() if not x.is_ref), None)
        if base_entry is None:
            print("  [!] aucune animation de base pour la transformation")
        else:
            base = SH.read_sheet(src, base_entry)
            tp = TR.TransformParams(n_frames=a.transform_frames,
                                    seed=a.seed,
                                    anim_name=a.transform_name)
            sheet, fx, (tw, th) = TR.build_transform_anim(base, params, tp)

            out_data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
            sheet.entry.index = C.free_action_index(out_data.used_indexes())
            SH.write_sheet(sheet, out)
            out_data.put(sheet.entry)
            SH.write_anim_data(out_data, os.path.join(out, C.MULTI_SHEET_XML))
            print(f"  [ok] {sheet.name:14s} {tw}x{th} x{sheet.n_frames}f "
                  f"(transformation, index {sheet.entry.index})")

            # planche FX à part, sur fond magenta
            fx_dir = os.path.join(out, "fx")
            TR.render_fx_sheet(fx, tw, th,
                               os.path.join(fx_dir, "Dynamax-FX-magenta.png"),
                               magenta=True)
            TR.render_fx_sheet(fx, tw, th,
                               os.path.join(fx_dir, "Dynamax-FX-alpha.png"),
                               magenta=False)
            TR.render_fx_sheet(fx, tw, th,
                               os.path.join(fx_dir, "Dynamax-FX-magenta-x4.png"),
                               magenta=True, scale=4)
            print(f"  [ok] FX          -> {fx_dir}/")

    # --- validation --------------------------------------------------------
    print()
    rep = VF.verify_folder(out)
    print(rep.summary())

    if a.gif:
        anim = a.gif if isinstance(a.gif, str) else "Walk"
        try:
            p = PV.anim_gif(out, anim, os.path.join(out, "preview",
                                                    f"{anim}.gif"), scale=3)
            print(f"\nAperçu GIF  : {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[!] GIF impossible : {exc}")

    return 0 if rep.ok else 1


def cmd_fx(a: argparse.Namespace) -> int:
    """Planche FX seule, sans Pokémon."""
    frames = BM.render_beam_frames(a.width, a.height,
                                   cx=a.width / 2,
                                   ground_y=a.height - a.ground_margin,
                                   head_y=a.head_y,
                                   n_frames=a.frames, seed=a.seed,
                                   column_width=a.column_width)
    out = a.out or os.path.join(DEFAULT_OUT_ROOT, "fx")
    os.makedirs(out, exist_ok=True)

    p1 = os.path.join(out, "Dynamax-FX-magenta.png")
    p2 = os.path.join(out, "Dynamax-FX-alpha.png")
    p3 = os.path.join(out, "Dynamax-FX-magenta-x4.png")
    TR.render_fx_sheet(frames, a.width, a.height, p1, magenta=True)
    TR.render_fx_sheet(frames, a.width, a.height, p2, magenta=False)
    TR.render_fx_sheet(frames, a.width, a.height, p3, magenta=True, scale=4)

    print(f"{a.frames} frames de {a.width}x{a.height}")
    print(f"  {p1}   (fond magenta #FF00FF)")
    print(f"  {p2}   (fond transparent)")
    print(f"  {p3}   (magenta, x4 pour inspection)")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    rep = VF.verify_folder(a.folder, require_complete=a.complete)
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_preview(a: argparse.Namespace) -> int:
    if a.contact:
        p = PV.contact_sheet(a.folder, a.out or os.path.join(a.folder, "preview",
                                                             "contact.png"),
                             anims=a.anims.split(",") if a.anims else None,
                             scale=a.scale)
        print(p)
        return 0
    anim = a.anims.split(",")[0] if a.anims else "Walk"
    p = PV.anim_gif(a.folder, anim,
                    a.out or os.path.join(a.folder, "preview", f"{anim}.gif"),
                    direction=a.direction, scale=a.scale)
    print(p)
    return 0


def cmd_ase(a: argparse.Namespace) -> int:
    data = SH.read_anim_data(os.path.join(a.folder, C.MULTI_SHEET_XML))
    entry = data.get(a.anim)
    if entry is None or entry.is_ref:
        raise SystemExit(f"animation '{a.anim}' introuvable dans {a.folder}")
    sheet = SH.read_sheet(a.folder, entry)
    tw, th = entry.size
    out = a.out or os.path.join(a.folder, "ase", f"{entry.name}.ase")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ASE.sheet_to_ase(out, sheet.anim, sheet.offsets, sheet.shadow, tw, th,
                     entry.durations, direction=a.direction,
                     split_clouds=True, cloud_colors=C.DYNA_RAMP[1:])
    print(f"{out}  ({tw}x{th}, {sheet.n_frames} frames, calques séparés)")
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    root = a.src_root
    if not os.path.isdir(root):
        raise SystemExit(f"{root} absent. Lance tools/fetch_upstream.sh")
    names = sorted(d for d in os.listdir(root)
                   if os.path.isfile(os.path.join(root, d, C.MULTI_SHEET_XML)))
    print(f"{len(names)} sprite(s) dans {root} :")
    for n in names:
        data = SH.read_anim_data(os.path.join(root, n, C.MULTI_SHEET_XML))
        real = [x for x in data.anims if not x.is_ref]
        print(f"  {n}  ({len(real)} animations, ShadowSize {data.shadow_size})")
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pmdmax",
        description="Génère des sprites Dynamax au format strict PMDCollab/SpriteCollab.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- build ---------------------------------------------------------
    b = sub.add_parser("build", help="génère un sprite Dynamax complet")
    b.add_argument("pokemon", help="numéro (0025), ou chemin d'un dossier sprite")
    b.add_argument("--out", help="dossier de sortie")
    b.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    b.add_argument("--scale", type=float, default=1.6,
                   help="facteur d'agrandissement (défaut 1.6)")
    b.add_argument("--scale-mode", choices=["epx", "nearest"], default="epx")
    b.add_argument("--clouds", type=int, default=3, help="nombre de nuages")
    b.add_argument("--cloud-scale", type=float, default=0.0, help="0 = auto")
    b.add_argument("--cloud-gap", type=int, default=6)
    b.add_argument("--orbit-rx", type=float, default=0.0, help="0 = auto")
    b.add_argument("--revolutions", type=float, default=1.0,
                   help="tours de nuages par boucle d'animation")
    b.add_argument("--no-bolts", action="store_true",
                   help="pas d'arcs électriques entre les nuages")
    b.add_argument("--no-aura", action="store_true",
                   help="désactive l'aura de fluide rouge")
    b.add_argument("--aura-reach", type=float, default=0.0,
                   help="portée de l'aura en pixels (0 = auto)")
    b.add_argument("--aura-strength", type=float, default=1.0,
                   help="intensité de l'aura (0..1)")
    b.add_argument("--aura-cycles", type=float, default=1.0,
                   help="ondulations d'aura par boucle d'animation")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--anims", help="liste d'animations (ex: Walk,Idle,Attack)")
    b.add_argument("--no-transform", action="store_true")
    b.add_argument("--transform-frames", type=int, default=16)
    b.add_argument("--transform-name", default="Special0")
    b.add_argument("--transform-base", default="Idle")
    b.add_argument("--gif", nargs="?", const="Walk", default=None,
                   help="génère un GIF d'aperçu")
    b.set_defaults(func=cmd_build)

    # -- fx ------------------------------------------------------------
    f = sub.add_parser("fx", help="planche FX de transformation (fond magenta)")
    f.add_argument("--out")
    f.add_argument("--width", type=int, default=64)
    f.add_argument("--height", type=int, default=96)
    f.add_argument("--frames", type=int, default=16)
    f.add_argument("--ground-margin", type=int, default=16,
                   help="hauteur du sol depuis le bas de la tuile")
    f.add_argument("--head-y", type=float, default=26.0)
    f.add_argument("--column-width", type=float, default=13.0)
    f.add_argument("--seed", type=int, default=1)
    f.set_defaults(func=cmd_fx)

    # -- verify --------------------------------------------------------
    v = sub.add_parser("verify", help="valide un dossier sprite")
    v.add_argument("folder")
    v.add_argument("--complete", action="store_true",
                   help="exiger aussi les animations obligatoires")
    v.set_defaults(func=cmd_verify)

    # -- preview -------------------------------------------------------
    pv = sub.add_parser("preview", help="GIF ou planche de contact")
    pv.add_argument("folder")
    pv.add_argument("--anims")
    pv.add_argument("--out")
    pv.add_argument("--direction", type=int, default=0)
    pv.add_argument("--scale", type=int, default=3)
    pv.add_argument("--contact", action="store_true")
    pv.set_defaults(func=cmd_preview)

    # -- ase -----------------------------------------------------------
    ae = sub.add_parser("ase", help="export Aseprite (.ase) multi-calques")
    ae.add_argument("folder")
    ae.add_argument("anim")
    ae.add_argument("--out")
    ae.add_argument("--direction", type=int, default=0)
    ae.set_defaults(func=cmd_ase)

    # -- list ----------------------------------------------------------
    ls = sub.add_parser("list", help="liste les sprites disponibles")
    ls.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    ls.set_defaults(func=cmd_list)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
