"""
Export .ase (Aseprite) natif, sans dépendance ni binaire Aseprite.

But : livrer les sprites Dynamax dans un fichier directement ouvrable pour
retouche manuelle, avec les CALQUES séparés — c'est là qu'Aseprite aide
vraiment, puisqu'on peut corriger les nuages sans toucher au Pokémon, ou
inversement.

Calques produits :
    Nuages-Avant     les touffes qui passent devant la tête
    Pokemon          le sprite agrandi
    Nuages-Arriere   les touffes qui passent derrière
    Ombre            les blobs d'ombre (référence, à ne pas peindre)
    Offsets          les 4 pixels d'ancrage (référence)

Format implémenté d'après la spec officielle :
https://github.com/aseprite/aseprite/blob/main/docs/ase-file-specs.md
(en-tête 128 o, frames 0xF1FA, chunks Layer 0x2004 / Cel 0x2005 / Tags 0x2018,
couleur 32 bits RGBA, cels compressés zlib.)
"""
from __future__ import annotations

import struct
import zlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

ASE_MAGIC = 0xA5E0
FRAME_MAGIC = 0xF1FA
CHUNK_LAYER = 0x2004
CHUNK_CEL = 0x2005
CHUNK_TAGS = 0x2018

BLEND_NORMAL = 0


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _layer_chunk(name: str, visible: bool = True, opacity: int = 255) -> bytes:
    flags = 0
    if visible:
        flags |= 1          # visible
    flags |= 2              # éditable
    body = struct.pack(
        "<HHHHHHB3s",
        flags,
        0,                  # type : image normale
        0,                  # niveau d'imbrication
        0, 0,               # largeur/hauteur par défaut (ignorées)
        BLEND_NORMAL,
        opacity,
        b"\x00\x00\x00",
    ) + _string(name)
    return struct.pack("<IH", 6 + len(body), CHUNK_LAYER) + body


def _cel_chunk(layer_index: int, img: np.ndarray, x: int = 0, y: int = 0,
               opacity: int = 255) -> bytes:
    """Cel de type 2 (image compressée zlib), en RGBA 32 bits."""
    h, w = img.shape[:2]
    raw = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    packed = zlib.compress(raw, 9)
    body = struct.pack(
        "<HhhBHh5s",
        layer_index, x, y, opacity,
        2,                  # type de cel : image compressée
        0,                  # z-index (Aseprite >= 1.3 ; 0 = normal)
        b"\x00" * 5,
    ) + struct.pack("<HH", w, h) + packed
    return struct.pack("<IH", 6 + len(body), CHUNK_CEL) + body


def _tags_chunk(tags: Sequence[Tuple[str, int, int]]) -> bytes:
    """tags = [(nom, frame_debut, frame_fin), ...]"""
    body = struct.pack("<H8s", len(tags), b"\x00" * 8)
    for name, lo, hi in tags:
        body += struct.pack(
            "<HHB8sBBB B",
            lo, hi,
            0,                    # direction : avant
            b"\x00" * 8,
            0, 0, 0,              # couleur RGB (deprecated)
            0,
        ) + _string(name)
    return struct.pack("<IH", 6 + len(body), CHUNK_TAGS) + body


def _frame(chunks: List[bytes], duration_ms: int) -> bytes:
    payload = b"".join(chunks)
    header = struct.pack(
        "<IHHH2sI",
        16 + len(payload),
        FRAME_MAGIC,
        min(len(chunks), 0xFFFF),
        max(1, min(65535, duration_ms)),
        b"\x00\x00",
        len(chunks),
    )
    return header + payload


def write_ase(path: str, width: int, height: int,
              layers: Sequence[str],
              frames: Sequence[Sequence[Optional[np.ndarray]]],
              durations: Optional[Sequence[int]] = None,
              tags: Optional[Sequence[Tuple[str, int, int]]] = None) -> None:
    """
    Écrit un fichier .ase multi-calques.

    layers  : noms des calques, du FOND vers le PREMIER PLAN.
    frames  : frames[i][j] = image RGBA du calque j à la frame i (ou None).
    durations : durée de chaque frame en millisecondes.
    """
    n_frames = len(frames)
    if durations is None:
        durations = [100] * n_frames

    body = b""
    for i, frame_layers in enumerate(frames):
        chunks: List[bytes] = []
        if i == 0:
            for name in layers:
                chunks.append(_layer_chunk(name))
            if tags:
                chunks.append(_tags_chunk(tags))
        for j, img in enumerate(frame_layers):
            if img is None:
                continue
            if not np.any(img[:, :, 3]):
                continue          # calque vide : pas de cel
            chunks.append(_cel_chunk(j, img))
        body += _frame(chunks, int(durations[i]))

    header = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH84s",
        128 + len(body),      # taille du fichier
        ASE_MAGIC,
        n_frames,
        width, height,
        32,                   # profondeur : RGBA
        1,                    # flags : opacité de calque valide
        100,                  # vitesse (déprécié)
        0, 0,
        0,                    # index de la couleur transparente
        b"\x00" * 3,
        0,                    # nombre de couleurs
        1, 1,                 # ratio de pixel 1:1
        0, 0,                 # position de la grille
        16, 16,               # taille de la grille
        b"\x00" * 84,
    )
    with open(path, "wb") as fh:
        fh.write(header + body)


# --------------------------------------------------------------------------
# Passerelle avec les feuilles PMD
# --------------------------------------------------------------------------

def sheet_to_ase(path: str, anim: np.ndarray, offsets: np.ndarray,
                 shadow: np.ndarray, tile_w: int, tile_h: int,
                 durations: Sequence[int], direction: int = 0,
                 split_clouds: bool = True,
                 cloud_colors: Optional[Sequence[Tuple[int, int, int, int]]] = None
                 ) -> None:
    """
    Exporte UNE direction d'un triplet de feuilles en .ase animé.

    Si `split_clouds`, les pixels appartenant à la palette Dynamax sont
    isolés sur leur propre calque : on peut alors retoucher les nuages
    indépendamment du Pokémon dans Aseprite.
    """
    n_frames = anim.shape[1] // tile_w
    y0 = direction * tile_h

    layer_names = ["Offsets", "Ombre"]
    if split_clouds:
        layer_names += ["Nuages", "Pokemon"]
    else:
        layer_names += ["Pokemon"]

    cloud_set = None
    if split_clouds and cloud_colors:
        cloud_set = {tuple(c) for c in cloud_colors}

    frames: List[List[Optional[np.ndarray]]] = []
    for f in range(n_frames):
        x0 = f * tile_w
        a = anim[y0:y0 + tile_h, x0:x0 + tile_w].copy()
        o = offsets[y0:y0 + tile_h, x0:x0 + tile_w].copy()
        s = shadow[y0:y0 + tile_h, x0:x0 + tile_w].copy()

        if split_clouds:
            clouds = np.zeros_like(a)
            mon = a.copy()
            if cloud_set:
                flat = a.reshape(-1, 4)
                mask = np.zeros(len(flat), dtype=bool)
                for col in cloud_set:
                    mask |= np.all(flat == np.array(col, dtype=np.uint8), axis=1)
                mask = mask.reshape(a.shape[:2])
                clouds[mask] = a[mask]
                mon[mask] = 0
            frames.append([o, s, clouds, mon])
        else:
            frames.append([o, s, a])

    ms = [max(10, int(d * 1000 / 60)) for d in durations]
    write_ase(path, tile_w, tile_h, layer_names, frames, ms)
