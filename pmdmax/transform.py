"""
L'animation de transformation Dynamax, en plusieurs frames.

Déroulé (storyboard) :
  1. le Pokémon, taille normale, se ramasse ; des étincelles montent du sol ;
  2. une COLONNE d'énergie rouge OPAQUE tombe du ciel et le percute ;
  3. impact : flash blanc, onde de choc, le Pokémon commence à grandir ;
  4. plusieurs ÉCLAIRS rouges montent en SPIRALE autour de lui ;
  5. les éclairs condensent les NUAGES au-dessus de sa tête ;
  6. la colonne se dissipe : le Pokémon est à sa taille Dynamax, les nuages
     tournoient.

Le résultat est écrit comme une animation SpriteCollab normale (triplet
Anim/Offsets/Shadow + entrée AnimData.xml) dans un slot `Special*`, afin de
rester dans les clous du format sans usurper un index réservé.

En parallèle, `render_fx_sheet` exporte les frames du seul FX sur fond
MAGENTA : c'est la planche demandée pour retoucher/monter l'effet à part.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from . import beam as BM
from . import config as C
from . import dynamax as DX
from . import graphicscale as GS
from . import pixels as P
from . import sheet as SH


#: Durées (en frames de jeu, 1 = 1/60 s) pour les 16 étapes.
#: Lentes au début (montée en tension), très courtes à l'impact, puis
#: on s'attarde sur la formation des nuages.
DEFAULT_DURATIONS = [
    6, 6,          # gather
    4, 3, 3, 2,    # descend (accélère)
    2, 3,          # impact / flash
    4, 4, 4, 4,    # spiral
    5, 5, 6,       # condense
    10,            # settle
]


@dataclass
class TransformParams:
    n_frames: int = 16
    seed: int = 1
    column_width: float = 13.0
    anim_name: str = "Special0"
    durations: Optional[List[int]] = None
    directions: int = 1          # 1 = mono-directionnel (suffit pour un FX)
    flash_sprite: bool = True    # blanchir le sprite à l'impact

    def resolved_durations(self) -> List[int]:
        if self.durations:
            return list(self.durations)
        base = DEFAULT_DURATIONS
        if self.n_frames == len(base):
            return list(base)
        # ré-échantillonnage si l'on change le nombre de frames
        return [base[int(i * len(base) / self.n_frames)]
                for i in range(self.n_frames)]


def _whiten(rgba: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Blanchit un sprite sans créer de demi-transparence : les pixels opaques
    passent au ton clair de la palette Dynamax. Utilisé sur la frame de flash.
    """
    out = rgba.copy()
    m = out[:, :, 3] > 0
    tint = np.array(C.DYNA_CORE if amount >= 0.75 else C.DYNA_RIM,
                    dtype=np.uint8)
    out[m] = tint
    return out


def build_transform_anim(src: SH.AnimSheet, params: DX.DynamaxParams,
                         tp: TransformParams) -> Tuple[SH.AnimSheet, List[BM.BeamFrame], Tuple[int, int]]:
    """
    Construit l'animation de transformation.

    `src` doit être une animation de repos (Idle de préférence) : on en prend
    la frame 0, direction Down, comme pose de départ.

    Renvoie (feuille, frames FX, (tile_w, tile_h)).
    """
    n = tp.n_frames
    src_rc = src.rel_center()

    # --- taille de tuile : celle du sprite Dynamax final ----------------
    tile_w, tile_h, cloud_scale, orbit_rx = DX._needed_tile_size(src, params)
    # la colonne tombe du haut de la tuile : on garantit un peu de hauteur
    tile_h = max(tile_h, DX.GS_round8(int(tile_h * 1.15)))

    # --- pose de départ --------------------------------------------------
    direction = 0                      # Down
    frame0 = 0
    a_tile = src.tile("anim", frame0, direction)
    o_tile = src.tile("offsets", frame0, direction)
    s_tile = src.tile("shadow", frame0, direction)

    bounds = SH.covered_bounds(a_tile)
    sdw_c = src.shadow_center(frame0, direction)
    offs = src.frame_offsets(frame0, direction)
    pivot = sdw_c or (offs.center if offs else src_rc)

    new_rc = (tile_w // 2 - C.DRAW_CENTER_X, tile_h // 2 - C.DRAW_CENTER_Y)
    delta = (pivot[0] - src_rc[0], pivot[1] - src_rc[1])
    new_pivot = (new_rc[0] + delta[0], new_rc[1] + delta[1])

    # altitude visée pour la couronne de nuages, à pleine taille
    if bounds is not None:
        top_full = new_pivot[1] - int((pivot[1] - bounds[1]) * params.scale)
    else:
        top_full = new_pivot[1] - int(tile_h * 0.4)
    head_y = max(2, top_full - params.cloud_gap)
    ground_y = new_pivot[1]

    # --- FX --------------------------------------------------------------
    fx_frames = BM.render_beam_frames(
        tile_w, tile_h, cx=float(new_pivot[0]), ground_y=float(ground_y),
        head_y=float(head_y), n_frames=n, seed=tp.seed,
        column_width=tp.column_width)

    # --- composition -----------------------------------------------------
    n_d = tp.directions
    anim = SH.blank(tile_w * n, tile_h * n_d)
    off_sheet = SH.blank(tile_w * n, tile_h * n_d)
    sdw_sheet = SH.blank(tile_w * n, tile_h * n_d)

    for i, fx in enumerate(fx_frames):
        # échelle du Pokémon à cette frame : 1.0 -> params.scale
        t = fx.scale_hint - 1.0
        span = max(1e-6, 2.0 - 1.0)          # scale_hint va de 1.0 à 2.0
        frac = max(0.0, min(1.0, t / span))
        cur_scale = 1.0 + (params.scale - 1.0) * frac

        tile_anim = SH.blank(tile_w, tile_h)
        tile_off = SH.blank(tile_w, tile_h)

        # 1. FX derrière (onde de choc au sol, nuages du fond)
        SH.paste(tile_anim, P.to_rgba(fx.behind), 0, 0)

        # 2. le Pokémon, à l'échelle du moment
        big = GS.graphic_scale(a_tile, cur_scale, mode=params.scale_mode)
        if tp.flash_sprite and fx.flash:
            big = _whiten(big, 1.0)
        bx = new_pivot[0] - int((pivot[0] + 0.5) * cur_scale)
        by = new_pivot[1] - int((pivot[1] + 0.5) * cur_scale)
        SH.paste(tile_anim, big, bx, by)

        # 3. FX devant (colonne, spirales, nuages de devant)
        SH.paste(tile_anim, P.to_rgba(fx.idx), 0, 0)

        # 4. offsets, suivant l'échelle courante
        if offs is not None:
            def mp(p: Tuple[int, int]) -> Tuple[int, int]:
                return (bx + int((p[0] + 0.5) * cur_scale),
                        by + int((p[1] + 0.5) * cur_scale))
            for pt, col in ((offs.head, C.OFF_HEAD), (offs.lhand, C.OFF_LHAND),
                            (offs.rhand, C.OFF_RHAND), (offs.center, C.OFF_CENTER)):
                nx, ny = mp(pt)
                if 0 <= nx < tile_w and 0 <= ny < tile_h:
                    tile_off[ny, nx] = np.array(col, dtype=np.uint8)

        # 5. ombre, qui grandit avec le Pokémon
        if sdw_c is not None:
            tile_sdw = DX._scale_shadow_tile(s_tile, cur_scale, sdw_c,
                                             new_pivot, tile_w, tile_h)
        else:
            tile_sdw = SH.blank(tile_w, tile_h)

        x = i * tile_w
        for d in range(n_d):
            y = d * tile_h
            anim[y:y + tile_h, x:x + tile_w] = tile_anim
            off_sheet[y:y + tile_h, x:x + tile_w] = tile_off
            sdw_sheet[y:y + tile_h, x:x + tile_w] = tile_sdw

    durations = tp.resolved_durations()
    entry = SH.AnimEntry(
        name=tp.anim_name,
        index=-1,                       # attribué par l'appelant
        frame_width=tile_w,
        frame_height=tile_h,
        durations=durations,
        hit_frame=min(7, n - 1),        # l'impact
        return_frame=n - 1,
    )
    sheet = SH.AnimSheet(name=tp.anim_name, entry=entry, anim=anim,
                         offsets=off_sheet, shadow=sdw_sheet)
    return sheet, fx_frames, (tile_w, tile_h)


def render_fx_sheet(fx_frames: List[BM.BeamFrame], tile_w: int, tile_h: int,
                    out_path: str, magenta: bool = True,
                    scale: int = 1) -> None:
    """Écrit la planche du FX seul (fond magenta ou transparent)."""
    bg = BM.MAGENTA if magenta else None
    sheet = BM.frames_to_sheet(fx_frames, tile_w, tile_h, background=bg)
    if scale > 1:
        sheet = GS.graphic_scale(sheet, scale, mode="nearest")
    SH.save_rgba(sheet, out_path)
