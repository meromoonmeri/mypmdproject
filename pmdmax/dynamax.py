"""
Pipeline principal : sprite SpriteCollab normal -> sprite Dynamax.

Pour CHAQUE animation, CHAQUE direction et CHAQUE frame :
  1. le Pokémon est agrandi (graphic scale EPX, palette inchangée) ;
  2. la tuile est agrandie pour accueillir le sprite + les nuages, en
     restant un multiple de 8 ;
  3. les 4 points d'ancrage (tête / main G / centre / main D) sont
     recalculés à la nouvelle échelle ;
  4. l'ombre est agrandie elle aussi (un Pokémon géant projette une
     grande ombre) et son pixel blanc repositionné ;
  5. la couronne de nuages est composée en deux passes — ceux qui passent
     DERRIÈRE la tête sous le sprite, ceux qui passent DEVANT par-dessus —
     avec une phase qui avance frame par frame : les nuages tournoient donc
     en boucle, calés sur la durée de l'animation.

Le point d'ancrage au sol (pixel blanc de -Shadow.png) sert de pivot : c'est
lui qui reste fixe pendant l'agrandissement, sinon le Pokémon décollerait
du sol ou s'y enfoncerait.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import clouds as CL
from . import config as C
from . import graphicscale as GS
from . import pixels as P
from . import sheet as SH


# --------------------------------------------------------------------------
# Réglages
# --------------------------------------------------------------------------

@dataclass
class DynamaxParams:
    """Tous les curseurs du rendu Dynamax."""
    scale: float = 1.5              # facteur d'agrandissement du Pokémon
    scale_mode: str = "epx"         # "epx" (lissé) ou "nearest" (gros pixels)
    cloud_count: int = 3            # 3 nuages = 3 tours de Dynamax
    cloud_scale: float = 0.0        # 0 = auto (proportionnel au Pokémon)
    cloud_gap: int = 6              # espace entre le sommet du crâne et l'orbite
    orbit_rx: float = 0.0           # 0 = auto (proportionnel à la largeur)
    orbit_ry: float = 4.0
    bolts: bool = True              # arcs électriques entre les nuages
    bob: float = 1.0                # flottement vertical des nuages
    seed: int = 1
    revolutions: float = 1.0        # tours effectués sur une boucle d'anim
    shadow_scale: float = 0.0       # 0 = suit `scale`

    def resolved_cloud_scale(self, sprite_w: int) -> float:
        if self.cloud_scale > 0:
            return self.cloud_scale
        # Petites touffes distinctes, comme dans Sword/Shield : ~1/9 de la
        # largeur du sprite. Plus gros, elles avalent la tête du Pokémon.
        return max(2.0, min(5.0, sprite_w / 9.0))

    def resolved_orbit_rx(self, sprite_w: int) -> float:
        if self.orbit_rx > 0:
            return self.orbit_rx
        # orbite un peu plus étroite que le sprite : la couronne doit
        # rester lisible comme un groupe au-dessus du crâne
        return max(5.0, min(16.0, sprite_w * 0.34))


# --------------------------------------------------------------------------
# Cache de nuages
# --------------------------------------------------------------------------

class CloudCache:
    """
    Les nuages ne dépendent que de (phase, taille, orbite) — pas de la
    direction ni du Pokémon. On les rend une fois dans un petit canevas
    local, puis on les recolle où il faut. Sans ce cache, une seule
    spritesheet 8 directions x 16 frames coûterait des milliers de rendus.
    """

    def __init__(self, params: DynamaxParams):
        self.p = params
        self._store: Dict[tuple, Tuple[np.ndarray, np.ndarray, int, int]] = {}

    def get(self, phase_key: int, n_phases: int, cloud_scale: float,
            rx: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Renvoie (back, front, cx_local, cy_local) pour la phase demandée.
        `phase_key` est un entier pour que le cache soit exact.
        """
        key = (phase_key, n_phases, round(cloud_scale, 2), round(rx, 2),
               self.p.cloud_count, self.p.bolts, self.p.seed, self.p.bob)
        hit = self._store.get(key)
        if hit is not None:
            return hit

        ry = self.p.orbit_ry
        margin = cloud_scale * 1.8 + 4
        w = int(math.ceil(2 * (rx + margin)))
        h = int(math.ceil(2 * (ry * 0.5 + margin + self.p.bob)))
        w += w % 2
        h += h % 2
        cx, cy = w / 2.0, h / 2.0

        phase = (phase_key / max(1, n_phases)) * self.p.revolutions
        back, front = CL.render_cloud_layer(
            w, h, cx, cy, phase=phase,
            count=self.p.cloud_count, rx=rx, ry=ry,
            base_scale=cloud_scale, bolts=self.p.bolts,
            seed=self.p.seed, bob=self.p.bob)

        out = (back, front, int(cx), int(cy))
        self._store[key] = out
        return out


# --------------------------------------------------------------------------
# Transformation d'une tuile
# --------------------------------------------------------------------------

@dataclass
class TileResult:
    anim: np.ndarray
    offsets: np.ndarray
    shadow: np.ndarray


def _scale_shadow_tile(src: np.ndarray, factor: float, anchor: Tuple[int, int],
                       new_anchor: Tuple[int, int], out_w: int, out_h: int
                       ) -> np.ndarray:
    """
    Agrandit les blobs d'ombre autour du point d'ancrage, puis repose le
    pixel blanc. On traite les 3 calques (vert/rouge/bleu) séparément pour
    qu'EPX ne mélange jamais deux couleurs réservées.
    """
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    ax, ay = anchor
    nax, nay = new_anchor

    for color in (C.SDW_LARGE, C.SDW_MED, C.SDW_SMALL):
        layer_mask = np.all(src == np.array(color, dtype=np.uint8), axis=-1)
        if not layer_mask.any():
            continue
        layer = np.zeros_like(src)
        layer[layer_mask] = np.array(color, dtype=np.uint8)
        big = GS.graphic_scale(layer, factor, mode="nearest")
        bx = nax - int((ax + 0.5) * factor)
        by = nay - int((ay + 0.5) * factor)
        SH.paste(out, big, bx, by)

    # le pixel blanc doit être UNIQUE : on l'écrit en dernier, en dur.
    if 0 <= nax < out_w and 0 <= nay < out_h:
        out[nay, nax] = np.array(C.SDW_CENTER, dtype=np.uint8)
    return out


def build_tile(anim_tile: np.ndarray, off_tile: np.ndarray, sdw_tile: np.ndarray,
               out_w: int, out_h: int, params: DynamaxParams,
               cloud: Optional[Tuple[np.ndarray, np.ndarray, int, int]],
               src_rel_center: Tuple[int, int]) -> TileResult:
    """
    Construit une tuile Dynamax à partir d'une tuile d'origine.

    `cloud` = (back, front, cx_local, cy_local) issu du CloudCache, ou None
    pour ne pas ajouter de nuages (frames vides).
    """
    f = params.scale
    anim_out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    off_out = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    # --- points d'ancrage d'origine ------------------------------------
    # Un pixel BLANC dans -Offsets.png vaut pour les 4 ancres à la fois
    # (convention SpriteBot). Plusieurs sprites du dépôt l'utilisent.
    white = SH.find_pixel(off_tile, C.OFF_ALL)
    center = SH.find_pixel(off_tile, C.OFF_CENTER) or white
    sdw_center = SH.find_pixel(sdw_tile, C.SDW_CENTER)
    bounds = SH.covered_bounds(anim_tile)

    new_rc = (out_w // 2 - C.DRAW_CENTER_X, out_h // 2 - C.DRAW_CENTER_Y)

    # Frame vide (ni sprite, ni offsets, ni ombre) : on la laisse vide.
    if bounds is None and center is None and sdw_center is None:
        return TileResult(anim_out, off_out,
                          np.zeros((out_h, out_w, 4), dtype=np.uint8))

    # Pivot : le point au sol. À défaut, le centre du corps, puis le
    # centre de tuile — c'est ce qui garde le Pokémon posé sur le sol.
    pivot = sdw_center or center or src_rel_center
    delta = (pivot[0] - src_rel_center[0], pivot[1] - src_rel_center[1])
    new_pivot = (new_rc[0] + delta[0], new_rc[1] + delta[1])

    # --- sprite ---------------------------------------------------------
    big = GS.graphic_scale(anim_tile, f, mode=params.scale_mode)
    px = new_pivot[0] - int((pivot[0] + 0.5) * f)
    py = new_pivot[1] - int((pivot[1] + 0.5) * f)

    def map_pt(p: Tuple[int, int]) -> Tuple[int, int]:
        return (px + int((p[0] + 0.5) * f), py + int((p[1] + 0.5) * f))

    # --- nuages : derrière -> sprite -> devant ---------------------------
    head = SH.find_pixel(off_tile, C.OFF_HEAD) or white or center
    cloud_anchor: Optional[Tuple[int, int]] = None
    if cloud is not None and bounds is not None:
        top_y = py + int((bounds[1] + 0.5) * f)
        head_x = map_pt(head)[0] if head else (px + int(((bounds[0] + bounds[2]) / 2) * f))
        cloud_anchor = (head_x, top_y - params.cloud_gap)

        back, front, ccx, ccy = cloud
        bx, by = cloud_anchor[0] - ccx, cloud_anchor[1] - ccy
        SH.paste(anim_out, P.to_rgba(back), bx, by)

    SH.paste(anim_out, big, px, py)

    if cloud is not None and cloud_anchor is not None:
        back, front, ccx, ccy = cloud
        bx, by = cloud_anchor[0] - ccx, cloud_anchor[1] - ccy
        SH.paste(anim_out, P.to_rgba(front), bx, by)

    # --- offsets ---------------------------------------------------------
    # Ordre d'écriture : centre en dernier pour qu'il ne soit jamais écrasé
    # (SpriteBot exige un vert par tuile ; le noir peut coexister).
    if center is not None:
        lhand = SH.find_pixel(off_tile, C.OFF_LHAND) or white
        rhand = SH.find_pixel(off_tile, C.OFF_RHAND) or white
        pts = [(head, C.OFF_HEAD), (lhand, C.OFF_LHAND),
               (rhand, C.OFF_RHAND), (center, C.OFF_CENTER)]
        mapped = [(map_pt(pt), col) for pt, col in pts if pt is not None]

        # Si les quatre ancres retombent sur le MÊME pixel, on réécrit le
        # raccourci blanc plutôt que d'empiler 4 couleurs sur 1 pixel
        # (sinon seule la dernière survivrait et on perdrait l'info).
        if len(mapped) == 4 and len({m[0] for m in mapped}) == 1:
            nx, ny = mapped[0][0]
            if 0 <= nx < out_w and 0 <= ny < out_h:
                off_out[ny, nx] = np.array(C.OFF_ALL, dtype=np.uint8)
        else:
            for (nx, ny), col in mapped:
                if 0 <= nx < out_w and 0 <= ny < out_h:
                    off_out[ny, nx] = np.array(col, dtype=np.uint8)

    # --- ombre ------------------------------------------------------------
    sdw_f = params.shadow_scale if params.shadow_scale > 0 else f
    if sdw_center is not None:
        sdw_out = _scale_shadow_tile(sdw_tile, sdw_f, sdw_center, new_pivot,
                                     out_w, out_h)
    else:
        sdw_out = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    return TileResult(anim_out, off_out, sdw_out)


# --------------------------------------------------------------------------
# Transformation d'une animation
# --------------------------------------------------------------------------

def _needed_tile_size(src: SH.AnimSheet, params: DynamaxParams
                      ) -> Tuple[int, int, float, float]:
    """
    Calcule la taille de tuile nécessaire pour contenir, à toutes les frames,
    le sprite agrandi ET la couronne de nuages, en multiples de 8.

    Renvoie (tile_w, tile_h, cloud_scale, orbit_rx).
    """
    f = params.scale
    rc = src.rel_center()

    # emprise maximale du sprite par rapport au pivot, sur toutes les frames
    max_l = max_r = max_u = max_d = 1.0
    sprite_w = 8

    for d in range(src.n_dirs):
        for fr in range(src.n_frames):
            tile = src.tile("anim", fr, d)
            b = SH.covered_bounds(tile)
            if b is None:
                continue
            sprite_w = max(sprite_w, b[2] - b[0])
            sc = src.shadow_center(fr, d)
            off = src.frame_offsets(fr, d)
            pivot = sc or (off.center if off else rc)
            delta = (pivot[0] - rc[0], pivot[1] - rc[1])
            # bornes exprimées depuis le pivot, mises à l'échelle, puis
            # ramenées au centre de tuile (le pivot n'est pas au centre)
            max_l = max(max_l, (pivot[0] - b[0]) * f - delta[0])
            max_r = max(max_r, (b[2] - pivot[0]) * f + delta[0])
            max_u = max(max_u, (pivot[1] - b[1]) * f - delta[1])
            max_d = max(max_d, (b[3] - pivot[1]) * f + delta[1])

    cloud_scale = params.resolved_cloud_scale(int(sprite_w * f))
    orbit_rx = params.resolved_orbit_rx(int(sprite_w * f))
    cloud_margin_x = orbit_rx + cloud_scale * 1.8 + 3
    cloud_margin_y = cloud_scale * 1.8 + params.orbit_ry * 0.5 + params.bob + 3

    # les nuages débordent en haut et sur les côtés
    max_l = max(max_l, cloud_margin_x)
    max_r = max(max_r, cloud_margin_x)
    max_u = max_u + params.cloud_gap + cloud_margin_y

    # l'ombre agrandie déborde aussi
    sdw_f = params.shadow_scale if params.shadow_scale > 0 else f
    max_d = max(max_d, 6 * sdw_f)
    max_l = max(max_l, 14 * sdw_f)
    max_r = max(max_r, 14 * sdw_f)

    # tuile symétrique autour du centre (le centre de tuile doit rester le
    # centre géométrique, sinon le jeu décale le sprite)
    half_w = max(max_l, max_r)
    half_h_up = max_u + C.DRAW_CENTER_Y      # DRAW_CENTER_Y = -4
    half_h_dn = max_d - C.DRAW_CENTER_Y
    half_h = max(half_h_up, half_h_dn)

    tile_w = GS_round8(int(math.ceil(half_w * 2)) + 2)
    tile_h = GS_round8(int(math.ceil(half_h * 2)) + 2)
    return tile_w, tile_h, cloud_scale, orbit_rx


def GS_round8(v: int) -> int:
    """Arrondi au multiple de 8 supérieur (convention des sheets PMD)."""
    return max(8, ((v + 7) // 8) * 8)


def dynamax_anim(src: SH.AnimSheet, params: DynamaxParams,
                 cache: CloudCache, with_clouds: bool = True) -> SH.AnimSheet:
    """Convertit une animation complète en version Dynamax."""
    tile_w, tile_h, cloud_scale, orbit_rx = _needed_tile_size(src, params)
    n_f, n_d = src.n_frames, src.n_dirs
    rc = src.rel_center()

    anim = SH.blank(tile_w * n_f, tile_h * n_d)
    offs = SH.blank(tile_w * n_f, tile_h * n_d)
    sdw = SH.blank(tile_w * n_f, tile_h * n_d)

    for fr in range(n_f):
        cloud = cache.get(fr, n_f, cloud_scale, orbit_rx) if with_clouds else None
        for d in range(n_d):
            res = build_tile(
                src.tile("anim", fr, d),
                src.tile("offsets", fr, d),
                src.tile("shadow", fr, d),
                tile_w, tile_h, params, cloud, rc)
            x, y = fr * tile_w, d * tile_h
            anim[y:y + tile_h, x:x + tile_w] = res.anim
            offs[y:y + tile_h, x:x + tile_w] = res.offsets
            sdw[y:y + tile_h, x:x + tile_w] = res.shadow

    entry = SH.AnimEntry(
        name=src.entry.name,
        index=src.entry.index,
        frame_width=tile_w,
        frame_height=tile_h,
        durations=list(src.entry.durations),
        rush_frame=src.entry.rush_frame,
        hit_frame=src.entry.hit_frame,
        return_frame=src.entry.return_frame,
    )
    return SH.AnimSheet(name=src.name, entry=entry,
                        anim=anim, offsets=offs, shadow=sdw)


# --------------------------------------------------------------------------
# Dossier complet
# --------------------------------------------------------------------------

def dynamax_folder(src_dir: str, out_dir: str, params: DynamaxParams,
                   only: Optional[List[str]] = None,
                   verbose: bool = True) -> Dict[str, object]:
    """
    Convertit un dossier sprite SpriteCollab entier.

    `only` limite le traitement à certaines animations (pour tester vite).
    Renvoie un rapport (animations traitées, palette, avertissements).
    """
    xml_path = os.path.join(src_dir, C.MULTI_SHEET_XML)
    data = SH.read_anim_data(xml_path)
    os.makedirs(out_dir, exist_ok=True)

    cache = CloudCache(params)
    out_data = SH.AnimData(shadow_size=data.shadow_size)
    palette: set = set()
    done: List[str] = []
    warnings: List[str] = []

    for entry in data.sorted_anims():
        if entry.is_ref:
            out_data.put(entry)          # <CopyOf> : rien à redessiner
            continue
        if only and entry.name not in only:
            continue

        base = os.path.join(src_dir, entry.name)
        if not os.path.isfile(base + "-Anim.png"):
            warnings.append(f"{entry.name}: -Anim.png absent, ignoré")
            continue

        src_sheet = SH.read_sheet(src_dir, entry)
        big = dynamax_anim(src_sheet, params, cache)
        SH.write_sheet(big, out_dir)
        out_data.put(big.entry)
        done.append(entry.name)

        flat = big.anim.reshape(-1, 4)
        flat = flat[flat[:, 3] > 0]
        if len(flat):
            palette |= set(map(tuple, np.unique(flat, axis=0).tolist()))

        if verbose:
            print(f"  [ok] {entry.name:14s} "
                  f"{big.entry.frame_width}x{big.entry.frame_height} "
                  f"x{big.n_frames}f x{big.n_dirs}d")

    SH.write_anim_data(out_data, os.path.join(out_dir, C.MULTI_SHEET_XML))

    # on recopie les métadonnées non graphiques du dossier source
    for extra in ("credits.txt", "tracker.json"):
        s = os.path.join(src_dir, extra)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(out_dir, extra))

    if len(palette) > C.PALETTE_SOFT_LIMIT:
        warnings.append(
            f"{len(palette)} couleurs (> {C.PALETTE_SOFT_LIMIT}) : la "
            f"soumission devra inclure `--colors {len(palette)}`.")

    return {"anims": done, "palette": len(palette),
            "warnings": warnings, "out_dir": out_dir}
