"""
Validateur strict, calqué sur PMDCollab/SpriteBot (SpriteUtils.verifySprite).

Il rejoue les mêmes contrôles que le bot de soumission, pour qu'un dossier
produit ici ne se fasse pas refuser :

  - AnimData.xml : ShadowSize 0..2, noms d'action valides, index uniques,
    index réservés respectés, animations obligatoires présentes ;
  - triplets Anim/Offsets/Shadow présents et de dimensions IDENTIQUES ;
  - feuille divisible par FrameWidth/FrameHeight ;
  - 1 ou 8 directions, jamais autre chose ;
  - nombre de frames == nombre de <Duration> ;
  - Rush/Hit/ReturnFrame dans les bornes ;
  - aucun pixel semi-transparent ;
  - un unique pixel vert (centre) et un unique blanc (ombre) par tuile ;
  - comptage de la palette (>15 => `--colors N` requis).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import config as C
from . import sheet as SH


@dataclass
class Report:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    palette: Set[tuple] = field(default_factory=set)
    anims: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = []
        status = "VALIDE" if self.ok else "REFUSE"
        lines.append(f"[{status}] {len(self.anims)} animation(s), "
                     f"{len(self.palette)} couleur(s)")
        for e in self.errors:
            lines.append(f"  ERREUR  : {e}")
        for w in self.warnings:
            lines.append(f"  attention: {w}")
        return "\n".join(lines)


def _check_offsets_tile(tile: np.ndarray, rep: Report, where: str) -> bool:
    """
    Contrôle des pixels d'offset d'une tuile. True si un centre est défini.

    Rappel : un pixel BLANC vaut pour les quatre ancres (SpriteBot
    utils.getOffsetFromRGB), et ne peut donc pas coexister avec du
    rouge/vert/bleu — le bot lève MultipleOffsetError dans ce cas.
    """
    whites = SH.find_pixels(tile, C.OFF_ALL)
    greens = SH.find_pixels(tile, C.OFF_CENTER)

    if len(whites) > 1:
        rep.err(f"{where}: {len(whites)} pixels blancs dans -Offsets.png "
                f"(un seul autorisé)")
    if len(greens) > 1:
        rep.err(f"{where}: {len(greens)} pixels verts dans -Offsets.png "
                f"(un seul autorisé)")
    for col, name in ((C.OFF_HEAD, "noir"), (C.OFF_LHAND, "rouge"),
                      (C.OFF_RHAND, "bleu")):
        n = len(SH.find_pixels(tile, col))
        if n > 1:
            rep.err(f"{where}: {n} pixels {name} dans -Offsets.png")

    if whites and (greens or SH.find_pixels(tile, C.OFF_LHAND)
                   or SH.find_pixels(tile, C.OFF_RHAND)):
        rep.err(f"{where}: pixel blanc ET pixels r/v/b dans -Offsets.png "
                f"(SpriteBot lève MultipleOffsetError)")

    # aucune couleur exotique ne doit traîner dans la feuille d'offsets
    allowed = {C.OFF_HEAD, C.OFF_LHAND, C.OFF_CENTER, C.OFF_RHAND, C.OFF_ALL}
    flat = tile.reshape(-1, 4)
    flat = flat[flat[:, 3] > 0]
    if len(flat):
        found = set(map(tuple, np.unique(flat, axis=0).tolist()))
        rogue = found - allowed
        if rogue:
            rep.err(f"{where}: couleurs interdites dans -Offsets.png: "
                    f"{sorted(rogue)[:4]}")
    return len(greens) == 1 or len(whites) == 1


def _check_shadow_tile(tile: np.ndarray, rep: Report, where: str) -> bool:
    whites = SH.find_pixels(tile, C.SDW_CENTER)
    if len(whites) > 1:
        rep.err(f"{where}: {len(whites)} pixels blancs dans -Shadow.png "
                f"(un seul autorisé)")
    allowed = {C.SDW_CENTER, C.SDW_SMALL, C.SDW_MED, C.SDW_LARGE}
    flat = tile.reshape(-1, 4)
    flat = flat[flat[:, 3] > 0]
    if len(flat):
        found = set(map(tuple, np.unique(flat, axis=0).tolist()))
        rogue = found - allowed
        if rogue:
            rep.err(f"{where}: couleurs interdites dans -Shadow.png: "
                    f"{sorted(rogue)[:4]}")
    return len(whites) == 1


def verify_folder(folder: str, require_complete: bool = False) -> Report:
    """Valide un dossier sprite complet."""
    rep = Report()
    xml_path = os.path.join(folder, C.MULTI_SHEET_XML)
    if not os.path.isfile(xml_path):
        rep.err(f"{C.MULTI_SHEET_XML} introuvable dans {folder}")
        return rep

    try:
        data = SH.read_anim_data(xml_path)
    except Exception as exc:  # noqa: BLE001
        rep.err(f"{C.MULTI_SHEET_XML} illisible : {exc}")
        return rep

    # --- XML : règles globales ------------------------------------------
    if not 0 <= data.shadow_size <= 2:
        rep.err(f"ShadowSize invalide : {data.shadow_size} (attendu 0..2)")

    seen_idx: Dict[int, str] = {}
    seen_names: Set[str] = set()
    for a in data.anims:
        if a.name not in C.ACTIONS:
            rep.err(f"Nom d'animation invalide : '{a.name}'")
        low = a.name.lower()
        if low in seen_names:
            rep.err(f"Animation '{a.name}' déclarée deux fois")
        seen_names.add(low)
        if a.index >= 0:
            if a.index in seen_idx:
                rep.err(f"Index {a.index} partagé par '{seen_idx[a.index]}' "
                        f"et '{a.name}'")
            seen_idx[a.index] = a.name
        if not a.is_ref and a.index < 0:
            rep.err(f"'{a.name}' a sa propre feuille mais aucun <Index>")

    for idx, required in C.ACTION_MAP.items():
        if idx in seen_idx and seen_idx[idx] != required:
            rep.err(f"L'index {idx} est réservé à '{required}' "
                    f"(trouvé '{seen_idx[idx]}')")

    if require_complete:
        missing = [C.ACTIONS[i] for i in C.COMPLETION_ACTIONS[0]
                   if C.ACTIONS[i].lower() not in seen_names]
        if missing:
            rep.err(f"Animations obligatoires manquantes : {', '.join(missing)}")

    # --- feuilles ---------------------------------------------------------
    for entry in data.sorted_anims():
        if entry.is_ref:
            if entry.copy_of and entry.copy_of.lower() not in seen_names:
                rep.err(f"'{entry.name}' fait <CopyOf> vers '{entry.copy_of}' "
                        f"qui n'existe pas")
            continue

        paths = SH.sheet_files(folder, entry.name)
        missing = [k for k, p in paths.items() if not os.path.isfile(p)]
        if missing:
            rep.err(f"{entry.name}: fichiers manquants ({', '.join(missing)})")
            continue

        for key, p in paths.items():
            if os.path.getsize(p) > C.ZIP_SIZE_LIMIT:
                rep.err(f"{entry.name}: {os.path.basename(p)} dépasse "
                        f"{C.ZIP_SIZE_LIMIT} octets")

        anim = SH.load_rgba(paths["anim"])
        offs = SH.load_rgba(paths["offsets"])
        sdw = SH.load_rgba(paths["shadow"])

        if anim.shape != offs.shape or anim.shape != sdw.shape:
            rep.err(f"{entry.name}: Anim/Offsets/Shadow de tailles "
                    f"différentes ({anim.shape[:2]}, {offs.shape[:2]}, "
                    f"{sdw.shape[:2]})")
            continue

        tw, th = entry.size
        H, W = anim.shape[:2]
        if tw <= 0 or th <= 0:
            rep.err(f"{entry.name}: FrameWidth/FrameHeight invalides")
            continue
        if W % tw or H % th:
            rep.err(f"{entry.name}: feuille {W}x{H} non divisible par "
                    f"{tw}x{th}")
            continue

        n_f, n_d = W // tw, H // th
        if n_d not in (1, 8):
            rep.err(f"{entry.name}: {n_d} directions (1 ou 8 attendues)")
        if n_f != len(entry.durations):
            rep.err(f"{entry.name}: {n_f} frames mais "
                    f"{len(entry.durations)} <Duration>")
        for label, val in (("RushFrame", entry.rush_frame),
                           ("HitFrame", entry.hit_frame),
                           ("ReturnFrame", entry.return_frame)):
            if val >= len(entry.durations):
                rep.err(f"{entry.name}: {label}={val} hors bornes "
                        f"({len(entry.durations)} frames)")
        if any(d <= 0 for d in entry.durations):
            rep.err(f"{entry.name}: une <Duration> est <= 0")

        # semi-transparence : rédhibitoire pour le bot
        alpha = anim[:, :, 3]
        semi = np.count_nonzero((alpha > 0) & (alpha < 255))
        if semi:
            ys, xs = np.where((alpha > 0) & (alpha < 255))
            rep.err(f"{entry.name}: {semi} pixel(s) semi-transparent(s), "
                    f"ex. {(int(xs[0]), int(ys[0]))}")

        flat = anim.reshape(-1, 4)
        flat = flat[flat[:, 3] == 255]
        if len(flat):
            rep.palette |= set(map(tuple, np.unique(flat, axis=0).tolist()))

        # contrôles par tuile
        for d in range(n_d):
            for f in range(n_f):
                where = f"{entry.name}[{C.DIRECTIONS[d] if n_d == 8 else 'All'},{f}]"
                a_t = anim[d * th:(d + 1) * th, f * tw:(f + 1) * tw]
                o_t = offs[d * th:(d + 1) * th, f * tw:(f + 1) * tw]
                s_t = sdw[d * th:(d + 1) * th, f * tw:(f + 1) * tw]

                has_center = _check_offsets_tile(o_t, rep, where)
                has_white = _check_shadow_tile(s_t, rep, where)
                empty = SH.covered_bounds(a_t) is None

                # une tuile totalement vide est tolérée (SpriteBot la saute)
                if empty and not has_center and not has_white:
                    continue
                if not has_center:
                    rep.err(f"{where}: aucun pixel vert (centre) "
                            f"dans -Offsets.png")
                if not has_white:
                    rep.err(f"{where}: aucun pixel blanc (ancre) "
                            f"dans -Shadow.png")

        rep.anims.append(entry.name)

    if len(rep.palette) > C.PALETTE_SOFT_LIMIT:
        rep.warn(f"{len(rep.palette)} couleurs non transparentes "
                 f"(> {C.PALETTE_SOFT_LIMIT}) : ajouter "
                 f"`--colors {len(rep.palette)}` à la soumission")

    return rep
