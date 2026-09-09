"""
Lecture / écriture des spritesheets SpriteCollab.

Un "AnimSheet" = un triplet de PNG de même taille exacte
  <Anim>-Anim.png      les pixels du Pokémon
  <Anim>-Offsets.png   1 px noir/rouge/vert/bleu par tuile (tête/mainG/centre/mainD)
  <Anim>-Shadow.png    blobs d'ombre + 1 px blanc par tuile (ancrage au sol)
plus une entrée <Anim> dans AnimData.xml.

La grille : colonnes = frames, lignes = directions (1 ou 8, jamais autre chose).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import config as C


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def load_rgba(path: str) -> np.ndarray:
    """Charge un PNG en HxWx4 uint8."""
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def save_rgba(arr: np.ndarray, path: str) -> None:
    """Écrit un HxWx4 uint8 en PNG RGBA (sans profil, sans métadonnée)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(path, optimize=True)


def blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def find_pixels(tile: np.ndarray, rgba: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """Coordonnées (x, y) des pixels valant exactement `rgba`."""
    m = np.all(tile == np.array(rgba, dtype=np.uint8), axis=-1)
    ys, xs = np.where(m)
    return list(zip(xs.tolist(), ys.tolist()))


def find_pixel(tile: np.ndarray, rgba: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
    px = find_pixels(tile, rgba)
    return px[0] if px else None


def covered_bounds(tile: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Boîte englobante des pixels non transparents : (x0, y0, x1, y1) exclusif."""
    ys, xs = np.where(tile[:, :, 3] > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Compose `src` sur `dst` en (x, y), alpha binaire, avec clipping."""
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    w = min(sw - sx0, dw - dx0)
    h = min(sh - sy0, dh - dy0)
    if w <= 0 or h <= 0:
        return
    region = src[sy0:sy0 + h, sx0:sx0 + w]
    mask = region[:, :, 3] > 0
    dst[dy0:dy0 + h, dx0:dx0 + w][mask] = region[mask]


# --------------------------------------------------------------------------
# Modèle AnimData.xml
# --------------------------------------------------------------------------

@dataclass
class AnimEntry:
    """Une balise <Anim> d'AnimData.xml."""
    name: str
    index: int = -1
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    durations: List[int] = field(default_factory=list)
    rush_frame: int = -1
    hit_frame: int = -1
    return_frame: int = -1
    copy_of: Optional[str] = None

    @property
    def is_ref(self) -> bool:
        return self.copy_of is not None

    @property
    def size(self) -> Tuple[int, int]:
        assert self.frame_width and self.frame_height
        return self.frame_width, self.frame_height


@dataclass
class AnimData:
    shadow_size: int = 1
    anims: List[AnimEntry] = field(default_factory=list)

    def get(self, name: str) -> Optional[AnimEntry]:
        low = name.lower()
        for a in self.anims:
            if a.name.lower() == low:
                return a
        return None

    def used_indexes(self) -> set:
        return {a.index for a in self.anims if a.index >= 0}

    def put(self, entry: AnimEntry) -> None:
        """Ajoute ou remplace une animation (par nom)."""
        for i, a in enumerate(self.anims):
            if a.name.lower() == entry.name.lower():
                self.anims[i] = entry
                return
        self.anims.append(entry)

    def sorted_anims(self) -> List[AnimEntry]:
        """Ordre du dépôt : par index, les <CopyOf> sans index à la fin."""
        keyed = [a for a in self.anims if a.index >= 0]
        rest = [a for a in self.anims if a.index < 0]
        keyed.sort(key=lambda a: a.index)
        return keyed + rest


def read_anim_data(path: str) -> AnimData:
    root = ET.parse(path).getroot()
    node = root.find("ShadowSize")
    data = AnimData(shadow_size=int(node.text) if node is not None and node.text else 1)

    anims_node = root.find("Anims")
    if anims_node is None:
        raise ValueError(f"{path}: balise <Anims> manquante")

    for an in anims_node.iter("Anim"):
        name_node = an.find("Name")
        if name_node is None or not name_node.text:
            raise ValueError(f"{path}: un <Anim> n'a pas de <Name>")
        entry = AnimEntry(name=name_node.text)

        idx = an.find("Index")
        if idx is not None and idx.text:
            entry.index = int(idx.text)

        cp = an.find("CopyOf")
        if cp is not None and cp.text:
            entry.copy_of = cp.text
            data.anims.append(entry)
            continue

        fw, fh = an.find("FrameWidth"), an.find("FrameHeight")
        if fw is None or not fw.text or fh is None or not fh.text:
            raise ValueError(f"{path}: FrameWidth/FrameHeight manquant pour {entry.name}")
        entry.frame_width, entry.frame_height = int(fw.text), int(fh.text)

        for tag, attr in (("RushFrame", "rush_frame"),
                          ("HitFrame", "hit_frame"),
                          ("ReturnFrame", "return_frame")):
            n = an.find(tag)
            if n is not None and n.text:
                setattr(entry, attr, int(n.text))

        durs = an.find("Durations")
        if durs is None:
            raise ValueError(f"{path}: <Durations> manquant pour {entry.name}")
        entry.durations = [int(d.text) for d in durs.iter("Duration") if d.text]

        data.anims.append(entry)

    return data


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Indentation par tabulations, à l'identique du dépôt SpriteCollab."""
    pad = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "\t"
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def write_anim_data(data: AnimData, path: str) -> None:
    """Sérialise AnimData.xml au format exact du dépôt (tabs, <?xml version="1.0"?>)."""
    root = ET.Element("AnimData")
    ET.SubElement(root, "ShadowSize").text = str(data.shadow_size)
    anims = ET.SubElement(root, "Anims")

    for a in data.sorted_anims():
        node = ET.SubElement(anims, "Anim")
        ET.SubElement(node, "Name").text = a.name
        if a.copy_of is not None:
            if a.index >= 0:
                ET.SubElement(node, "Index").text = str(a.index)
            ET.SubElement(node, "CopyOf").text = a.copy_of
            continue
        ET.SubElement(node, "Index").text = str(a.index)
        ET.SubElement(node, "FrameWidth").text = str(a.frame_width)
        ET.SubElement(node, "FrameHeight").text = str(a.frame_height)
        if a.rush_frame >= 0:
            ET.SubElement(node, "RushFrame").text = str(a.rush_frame)
        if a.hit_frame >= 0:
            ET.SubElement(node, "HitFrame").text = str(a.hit_frame)
        if a.return_frame >= 0:
            ET.SubElement(node, "ReturnFrame").text = str(a.return_frame)
        durs = ET.SubElement(node, "Durations")
        for d in a.durations:
            ET.SubElement(durs, "Duration").text = str(d)

    _indent(root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    xml = ET.tostring(root, encoding="unicode")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('<?xml version="1.0" ?>\n' + xml + "\n")


# --------------------------------------------------------------------------
# Triplet de feuilles
# --------------------------------------------------------------------------

@dataclass
class FrameOffsets:
    """Les 4 points d'ancrage d'une frame, en coordonnées tuile."""
    head: Tuple[int, int]
    lhand: Tuple[int, int]
    center: Tuple[int, int]
    rhand: Tuple[int, int]


@dataclass
class AnimSheet:
    """Un triplet Anim/Offsets/Shadow + ses métadonnées."""
    name: str
    entry: AnimEntry
    anim: np.ndarray
    offsets: np.ndarray
    shadow: np.ndarray

    @property
    def tile_w(self) -> int:
        return self.entry.frame_width  # type: ignore[return-value]

    @property
    def tile_h(self) -> int:
        return self.entry.frame_height  # type: ignore[return-value]

    @property
    def n_frames(self) -> int:
        return self.anim.shape[1] // self.tile_w

    @property
    def n_dirs(self) -> int:
        return self.anim.shape[0] // self.tile_h

    def tile(self, layer: str, frame: int, direction: int) -> np.ndarray:
        arr = getattr(self, layer)
        w, h = self.tile_w, self.tile_h
        return arr[direction * h:(direction + 1) * h, frame * w:(frame + 1) * w]

    def rel_center(self) -> Tuple[int, int]:
        """Le centre de tuile utilisé par SpriteBot pour les frames vides."""
        return (self.tile_w // 2 - C.DRAW_CENTER_X,
                self.tile_h // 2 - C.DRAW_CENTER_Y)

    def frame_offsets(self, frame: int, direction: int) -> Optional[FrameOffsets]:
        """
        Lit les 4 ancres d'une tuile, en appliquant la même logique que
        SpriteBot (utils.getOffsetFromRGB) : un pixel BLANC vaut pour les
        quatre ancres à la fois ; un pixel noir présent en plus l'emporte
        pour la tête.
        """
        t = self.tile("offsets", frame, direction)
        white = find_pixel(t, C.OFF_ALL)
        center = find_pixel(t, C.OFF_CENTER) or white
        if center is None:
            return None
        head = find_pixel(t, C.OFF_HEAD) or white or center
        lhand = find_pixel(t, C.OFF_LHAND) or white or center
        rhand = find_pixel(t, C.OFF_RHAND) or white or center
        return FrameOffsets(head=head, lhand=lhand, center=center, rhand=rhand)

    def shadow_center(self, frame: int, direction: int) -> Optional[Tuple[int, int]]:
        return find_pixel(self.tile("shadow", frame, direction), C.SDW_CENTER)


def read_sheet(folder: str, entry: AnimEntry) -> AnimSheet:
    base = os.path.join(folder, entry.name)
    return AnimSheet(
        name=entry.name,
        entry=entry,
        anim=load_rgba(base + "-Anim.png"),
        offsets=load_rgba(base + "-Offsets.png"),
        shadow=load_rgba(base + "-Shadow.png"),
    )


def write_sheet(sheet: AnimSheet, folder: str) -> None:
    base = os.path.join(folder, sheet.name)
    save_rgba(sheet.anim, base + "-Anim.png")
    save_rgba(sheet.offsets, base + "-Offsets.png")
    save_rgba(sheet.shadow, base + "-Shadow.png")


def sheet_files(folder: str, name: str) -> Dict[str, str]:
    base = os.path.join(folder, name)
    return {
        "anim": base + "-Anim.png",
        "offsets": base + "-Offsets.png",
        "shadow": base + "-Shadow.png",
    }
