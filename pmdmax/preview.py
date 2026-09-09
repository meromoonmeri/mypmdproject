"""
Aperçus : GIF animés et planches de contact.

Rien ici n'entre dans une soumission SpriteCollab — c'est uniquement pour
juger le rendu à l'œil, en respectant les vraies durées d'animation
(1 frame de jeu = 1/60 s).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import config as C
from . import graphicscale as GS
from . import sheet as SH

#: Fond des aperçus (le vert-canard utilisé par SpriteBot pour ses GIF).
PREVIEW_BG = (0, 128, 128, 255)


def _flatten(tile: np.ndarray, bg: Tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(tile.shape, dtype=np.uint8)
    out[:, :] = np.array(bg, dtype=np.uint8)
    m = tile[:, :, 3] > 0
    out[m] = tile[m]
    return out


def anim_gif(folder: str, anim_name: str, out_path: str, direction: int = 0,
             scale: int = 3, bg: Tuple[int, int, int, int] = PREVIEW_BG,
             with_shadow: bool = True, shadow_size: Optional[int] = None) -> str:
    """
    Exporte une animation en GIF, à la vitesse réelle du jeu.

    L'ombre est rendue comme dans le jeu : les blobs visibles dépendent de
    ShadowSize (vert toujours, rouge si >=1, bleu si >=2).
    """
    data = SH.read_anim_data(os.path.join(folder, C.MULTI_SHEET_XML))
    entry = data.get(anim_name)
    if entry is None:
        raise ValueError(f"animation '{anim_name}' absente de {folder}")
    if entry.is_ref:
        raise ValueError(f"'{anim_name}' est un <CopyOf> de '{entry.copy_of}'")
    sdw_size = data.shadow_size if shadow_size is None else shadow_size

    sheet = SH.read_sheet(folder, entry)
    tw, th = entry.size
    direction = min(direction, sheet.n_dirs - 1)

    imgs: List[Image.Image] = []
    durations: List[int] = []

    for f in range(sheet.n_frames):
        tile = sheet.tile("anim", f, direction)
        frame = np.zeros((th, tw, 4), dtype=np.uint8)
        frame[:, :] = np.array(bg, dtype=np.uint8)

        if with_shadow:
            s = sheet.tile("shadow", f, direction)
            vis = np.zeros(s.shape[:2], dtype=bool)
            vis |= np.all(s == np.array(C.SDW_SMALL, dtype=np.uint8), axis=-1)
            if sdw_size >= 1:
                vis |= np.all(s == np.array(C.SDW_MED, dtype=np.uint8), axis=-1)
            if sdw_size >= 2:
                vis |= np.all(s == np.array(C.SDW_LARGE, dtype=np.uint8), axis=-1)
            frame[vis] = np.array((0, 0, 0, 255), dtype=np.uint8)

        m = tile[:, :, 3] > 0
        frame[m] = tile[m]

        if scale > 1:
            frame = GS.graphic_scale(frame, scale, mode="nearest")
        imgs.append(Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE))
        durations.append(max(20, int(entry.durations[f] * 1000 / 60)))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imgs[0].save(out_path, format="GIF", save_all=True,
                 append_images=imgs[1:], duration=durations, loop=0,
                 disposal=2, optimize=False)
    return out_path


def contact_sheet(folder: str, out_path: str, anims: Optional[Sequence[str]] = None,
                  direction: int = 0, scale: int = 2,
                  bg: Tuple[int, int, int, int] = (28, 28, 36, 255),
                  max_cols: int = 12) -> str:
    """Planche de contact : une ligne par animation, ses frames en colonnes."""
    data = SH.read_anim_data(os.path.join(folder, C.MULTI_SHEET_XML))
    entries = [a for a in data.sorted_anims() if not a.is_ref]
    if anims:
        wanted = {a.lower() for a in anims}
        entries = [e for e in entries if e.name.lower() in wanted]
    if not entries:
        raise ValueError("aucune animation à afficher")

    rows = []
    for e in entries:
        sheet = SH.read_sheet(folder, e)
        tw, th = e.size
        d = min(direction, sheet.n_dirs - 1)
        n = min(sheet.n_frames, max_cols)
        row = np.zeros((th, tw * n, 4), dtype=np.uint8)
        row[:, :] = np.array(bg, dtype=np.uint8)
        for f in range(n):
            t = sheet.tile("anim", f, d)
            reg = row[:, f * tw:(f + 1) * tw]
            m = t[:, :, 3] > 0
            reg[m] = t[m]
        rows.append((e.name, row))

    W = max(r.shape[1] for _, r in rows)
    H = sum(r.shape[0] for _, r in rows)
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :] = np.array(bg, dtype=np.uint8)
    y = 0
    for _, r in rows:
        out[y:y + r.shape[0], 0:r.shape[1]] = r
        y += r.shape[0]

    if scale > 1:
        out = GS.graphic_scale(out, scale, mode="nearest")
    SH.save_rgba(out, out_path)
    return out_path


def compare_gif(folder_a: str, folder_b: str, anim_name: str, out_path: str,
                direction: int = 0, scale: int = 3,
                bg: Tuple[int, int, int, int] = PREVIEW_BG) -> str:
    """GIF côte à côte : sprite d'origine à gauche, Dynamax à droite."""
    frames: List[Image.Image] = []
    durations: List[int] = []

    packs = []
    for folder in (folder_a, folder_b):
        data = SH.read_anim_data(os.path.join(folder, C.MULTI_SHEET_XML))
        entry = data.get(anim_name)
        if entry is None or entry.is_ref:
            raise ValueError(f"'{anim_name}' indisponible dans {folder}")
        packs.append((SH.read_sheet(folder, entry), entry))

    n = min(p[0].n_frames for p in packs)
    tw = sum(p[1].size[0] for p in packs)
    th = max(p[1].size[1] for p in packs)

    for f in range(n):
        canvas = np.zeros((th, tw, 4), dtype=np.uint8)
        canvas[:, :] = np.array(bg, dtype=np.uint8)
        x = 0
        for sh_, entry in packs:
            w, h = entry.size
            d = min(direction, sh_.n_dirs - 1)
            t = sh_.tile("anim", f, d)
            reg = canvas[th - h:th, x:x + w]
            m = t[:, :, 3] > 0
            reg[m] = t[m]
            x += w
        if scale > 1:
            canvas = GS.graphic_scale(canvas, scale, mode="nearest")
        frames.append(Image.fromarray(canvas).convert("P", palette=Image.ADAPTIVE))
        durations.append(max(20, int(packs[0][1].durations[f] * 1000 / 60)))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    frames[0].save(out_path, format="GIF", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0,
                   disposal=2, optimize=False)
    return out_path
