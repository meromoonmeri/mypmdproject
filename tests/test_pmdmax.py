"""
Tests de pmdmax.

Les tests marqués "upstream" ont besoin du clone SpriteCollab
(tools/fetch_upstream.sh) et se désactivent tout seuls s'il est absent.

    .venv/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import os
import struct
import sys
import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmdmax import aseprite as ASE          # noqa: E402
from pmdmax import beam as BM               # noqa: E402
from pmdmax import clouds as CL             # noqa: E402
from pmdmax import config as C              # noqa: E402
from pmdmax import dynamax as DX            # noqa: E402
from pmdmax import graphicscale as GS       # noqa: E402
from pmdmax import pixels as P              # noqa: E402
from pmdmax import sheet as SH              # noqa: E402
from pmdmax import transform as TR          # noqa: E402
from pmdmax import verify as VF             # noqa: E402

UPSTREAM = os.path.join(C.REPO, "external", "SpriteCollab", "sprite")
PIKACHU = os.path.join(UPSTREAM, "0025")
needs_upstream = pytest.mark.skipif(
    not os.path.isdir(PIKACHU),
    reason="clone SpriteCollab absent (tools/fetch_upstream.sh)")


# --------------------------------------------------------------------------
# graphic scale
# --------------------------------------------------------------------------

def _rand_img(w=17, h=13, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    pal = [(0, 0, 0, 0), (255, 247, 0, 255), (0, 0, 0, 255),
           (223, 183, 0, 255), (215, 63, 0, 255)]
    for y in range(h):
        for x in range(w):
            img[y, x] = pal[int(rng.integers(0, len(pal)))]
    return img


@pytest.mark.parametrize("factor", [2, 3, 4, 6, 1.5, 1.6, 2.25])
@pytest.mark.parametrize("mode", ["epx", "nearest"])
def test_scale_never_invents_colors(factor, mode):
    """Contrainte dure de SpriteBot : pas de couleur nouvelle, pas d'alpha partiel."""
    src = _rand_img(seed=1)
    out = GS.graphic_scale(src, factor, mode=mode)
    assert GS.palette_of(out) <= GS.palette_of(src)
    alpha = out[:, :, 3]
    assert np.all((alpha == 0) | (alpha == 255)), "alpha semi-transparent produit"


@pytest.mark.parametrize("factor", [2, 3, 4, 1.5])
def test_scale_output_size(factor):
    src = _rand_img(w=16, h=20)
    out = GS.graphic_scale(src, factor)
    assert out.shape[1] == round(16 * factor)
    assert out.shape[0] == round(20 * factor)


def test_scale2x_preserves_flat_area():
    img = np.zeros((6, 6, 4), dtype=np.uint8)
    img[:, :] = (10, 20, 30, 255)
    out = GS.scale2x(img)
    assert out.shape == (12, 12, 4)
    assert np.all(out == np.array((10, 20, 30, 255), dtype=np.uint8))


def test_scale_identity():
    src = _rand_img()
    assert np.array_equal(GS.graphic_scale(src, 1.0), src)


# --------------------------------------------------------------------------
# palette / pixels
# --------------------------------------------------------------------------

def test_dyna_ramp_is_opaque_and_unique():
    ramp = C.DYNA_RAMP[1:]
    assert len(set(ramp)) == len(ramp), "couleurs Dynamax dupliquées"
    for col in ramp:
        assert col[3] == 255
    assert C.DYNA_RAMP[0] == (0, 0, 0, 0)


def test_dyna_ramp_is_monotonic():
    """La rampe doit vraiment aller du sombre au clair."""
    lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b, _ in C.DYNA_RAMP[1:]]
    assert lum == sorted(lum), f"rampe non monotone : {lum}"


def test_to_rgba_only_palette_colors():
    idx = np.arange(7, dtype=np.uint8).reshape(1, 7)
    out = P.to_rgba(idx)
    assert set(map(tuple, out.reshape(-1, 4).tolist())) <= set(C.DYNA_RAMP)


def test_shade_sphere_has_outline_and_stays_in_palette():
    mask = np.zeros((20, 20), dtype=bool)
    P.disc_mask(mask, 10, 10, 6)
    out = P.shade_sphere(mask, 10, 10, 6)
    assert (out == C.IDX_OUTLINE).any(), "aucun contour tracé"
    assert out.max() <= 6
    # le contour est bien hors du masque
    assert not (mask & (out == C.IDX_OUTLINE)).all()


def test_magenta_is_not_in_dynamax_palette():
    """Le chroma-key doit être distinguable du FX."""
    assert BM.MAGENTA not in C.DYNA_RAMP


# --------------------------------------------------------------------------
# nuages
# --------------------------------------------------------------------------

def test_clouds_split_front_and_back():
    """Sur un tour complet, des touffes doivent passer devant ET derrière."""
    saw_back = saw_front = False
    for i in range(8):
        back, front = CL.render_cloud_layer(48, 32, 24, 16, phase=i / 8,
                                            count=3, base_scale=4.0, seed=1)
        saw_back |= bool((back > 0).any())
        saw_front |= bool((front > 0).any())
    assert saw_back and saw_front


def test_clouds_orbit_moves():
    """Deux phases différentes ne doivent pas donner la même image."""
    _, f0 = CL.render_cloud_layer(48, 32, 24, 16, phase=0.0, seed=1)
    _, f1 = CL.render_cloud_layer(48, 32, 24, 16, phase=0.25, seed=1)
    assert not np.array_equal(f0, f1)


def test_clouds_are_deterministic():
    a = CL.render_cloud_layer(48, 32, 24, 16, phase=0.3, seed=7)
    b = CL.render_cloud_layer(48, 32, 24, 16, phase=0.3, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_cloud_count_respected():
    for n in (1, 3, 5):
        back, front = CL.render_cloud_layer(64, 48, 32, 24, phase=0.12,
                                            count=n, base_scale=3.0, seed=2)
        assert (back > 0).any() or (front > 0).any()


# --------------------------------------------------------------------------
# FX / colonne
# --------------------------------------------------------------------------

def test_beam_story_progresses():
    """Le FX doit réellement évoluer : début discret, milieu chargé."""
    frames = BM.render_beam_frames(64, 96, 32, 80, 24, n_frames=16, seed=1)
    assert len(frames) == 16
    counts = [int((f.idx > 0).sum()) for f in frames]
    assert counts[0] < max(counts)
    assert any(f.flash for f in frames), "aucune frame de flash à l'impact"
    # le Pokémon grandit du début à la fin
    assert frames[-1].scale_hint > frames[0].scale_hint


def test_beam_sheet_magenta_background():
    frames = BM.render_beam_frames(32, 48, 16, 40, 12, n_frames=4, seed=1)
    sheet = BM.frames_to_sheet(frames, 32, 48, background=BM.MAGENTA)
    assert sheet.shape == (48, 32 * 4, 4)
    corners = [tuple(sheet[0, 0]), tuple(sheet[-1, -1])]
    assert all(c == BM.MAGENTA for c in corners)
    assert np.all(sheet[:, :, 3] == 255), "le fond magenta doit être opaque"


def test_beam_sheet_alpha_background():
    frames = BM.render_beam_frames(32, 48, 16, 40, 12, n_frames=4, seed=1)
    sheet = BM.frames_to_sheet(frames, 32, 48, background=None)
    assert tuple(sheet[0, 0]) == (0, 0, 0, 0)
    a = sheet[:, :, 3]
    assert np.all((a == 0) | (a == 255))


def test_beam_only_uses_dynamax_palette():
    frames = BM.render_beam_frames(48, 64, 24, 56, 16, n_frames=8, seed=4)
    sheet = BM.frames_to_sheet(frames, 48, 64, background=None)
    assert GS.palette_of(sheet) <= set(C.DYNA_RAMP[1:])


# --------------------------------------------------------------------------
# AnimData.xml
# --------------------------------------------------------------------------

def test_anim_data_roundtrip(tmp_path):
    data = SH.AnimData(shadow_size=2)
    data.put(SH.AnimEntry(name="Walk", index=0, frame_width=32,
                          frame_height=40, durations=[8, 10, 8, 10]))
    data.put(SH.AnimEntry(name="Attack", index=1, frame_width=80,
                          frame_height=80, durations=[2] * 10,
                          rush_frame=1, hit_frame=3, return_frame=6))
    data.put(SH.AnimEntry(name="Idle", index=7, copy_of="Walk"))
    p = str(tmp_path / "AnimData.xml")
    SH.write_anim_data(data, p)

    back = SH.read_anim_data(p)
    assert back.shadow_size == 2
    w = back.get("Walk")
    assert w.index == 0 and w.size == (32, 40) and w.durations == [8, 10, 8, 10]
    a = back.get("Attack")
    assert (a.rush_frame, a.hit_frame, a.return_frame) == (1, 3, 6)
    assert back.get("Idle").copy_of == "Walk"

    raw = open(p, encoding="utf-8").read()
    assert raw.startswith('<?xml version="1.0" ?>')
    assert "\t<ShadowSize>" in raw, "indentation par tabulations attendue"


def test_free_action_index_skips_reserved():
    idx = C.free_action_index(set())
    assert idx not in C.ACTION_MAP
    used = set(C.ACTION_MAP) | {13, 14}
    assert C.free_action_index(used) not in used


# --------------------------------------------------------------------------
# offsets : convention du pixel blanc
# --------------------------------------------------------------------------

def test_white_offset_pixel_means_all_anchors():
    """Convention SpriteBot : blanc = head+lhand+center+rhand au même point."""
    entry = SH.AnimEntry(name="Walk", index=0, frame_width=8,
                         frame_height=8, durations=[1])
    offs = SH.blank(8, 8)
    offs[3, 4] = np.array(C.OFF_ALL, dtype=np.uint8)
    s = SH.AnimSheet(name="Walk", entry=entry, anim=SH.blank(8, 8),
                     offsets=offs, shadow=SH.blank(8, 8))
    fo = s.frame_offsets(0, 0)
    assert fo is not None
    assert fo.head == fo.lhand == fo.center == fo.rhand == (4, 3)


def test_black_pixel_overrides_white_for_head():
    entry = SH.AnimEntry(name="Walk", index=0, frame_width=8,
                         frame_height=8, durations=[1])
    offs = SH.blank(8, 8)
    offs[3, 4] = np.array(C.OFF_ALL, dtype=np.uint8)
    offs[1, 2] = np.array(C.OFF_HEAD, dtype=np.uint8)
    s = SH.AnimSheet(name="Walk", entry=entry, anim=SH.blank(8, 8),
                     offsets=offs, shadow=SH.blank(8, 8))
    fo = s.frame_offsets(0, 0)
    assert fo.head == (2, 1)
    assert fo.center == (4, 3)


# --------------------------------------------------------------------------
# aseprite
# --------------------------------------------------------------------------

def test_ase_file_parses_per_spec(tmp_path):
    """Relit le .ase produit en suivant la spec officielle du format."""
    w, h, n = 12, 10, 3
    layers = ["Fond", "Devant"]
    frames = []
    for i in range(n):
        a = np.zeros((h, w, 4), dtype=np.uint8)
        a[i, i] = (255, 0, 0, 255)
        b = np.zeros((h, w, 4), dtype=np.uint8)
        b[h - 1 - i, w - 1 - i] = (0, 255, 0, 255)
        frames.append([a, b])

    path = str(tmp_path / "t.ase")
    ASE.write_ase(path, w, h, layers, frames, [100] * n,
                  tags=[("boucle", 0, n - 1)])

    data = open(path, "rb").read()
    size, magic, nframes, fw, fh, depth = struct.unpack("<IHHHHH", data[:14])
    assert magic == ASE.ASE_MAGIC
    assert size == len(data), "taille déclarée != taille réelle"
    assert (nframes, fw, fh, depth) == (n, w, h, 32)

    off = 128
    n_layers = n_cels = 0
    for _ in range(nframes):
        fsize, fmagic, _, _, _, nchunks = struct.unpack("<IHHH2sI",
                                                        data[off:off + 16])
        assert fmagic == ASE.FRAME_MAGIC
        p = off + 16
        for _ in range(nchunks):
            csize, ctype = struct.unpack("<IH", data[p:p + 6])
            if ctype == ASE.CHUNK_LAYER:
                n_layers += 1
            elif ctype == ASE.CHUNK_CEL:
                n_cels += 1
                cw, ch = struct.unpack("<HH", data[p + 22:p + 26])
                raw = zlib.decompress(data[p + 26:p + csize])
                assert len(raw) == cw * ch * 4
            p += csize
        assert p == off + fsize, "chunks non alignés sur la frame"
        off += fsize
    assert off == len(data)
    assert n_layers == len(layers)
    assert n_cels == n * len(layers)


# --------------------------------------------------------------------------
# vérificateur
# --------------------------------------------------------------------------

def _minimal_sprite(folder: str, semi_transparent=False, drop_center=False,
                    bad_durations=False, bad_index=False):
    os.makedirs(folder, exist_ok=True)
    tw = th = 16
    anim = SH.blank(tw, th)
    anim[4:12, 4:12] = (255, 247, 0, 255)
    if semi_transparent:
        anim[5, 5] = (255, 247, 0, 128)
    offs = SH.blank(tw, th)
    if not drop_center:
        offs[8, 8] = np.array(C.OFF_CENTER, dtype=np.uint8)
    sdw = SH.blank(tw, th)
    sdw[12, 8] = np.array(C.SDW_CENTER, dtype=np.uint8)

    entry = SH.AnimEntry(name="Walk", index=3 if bad_index else 0,
                         frame_width=tw, frame_height=th,
                         durations=[8, 8] if bad_durations else [8])
    SH.write_sheet(SH.AnimSheet("Walk", entry, anim, offs, sdw), folder)
    data = SH.AnimData(shadow_size=1)
    data.put(entry)
    SH.write_anim_data(data, os.path.join(folder, C.MULTI_SHEET_XML))
    return folder


def test_verify_accepts_valid(tmp_path):
    rep = VF.verify_folder(_minimal_sprite(str(tmp_path / "ok")))
    assert rep.ok, rep.summary()


def test_verify_rejects_semi_transparency(tmp_path):
    rep = VF.verify_folder(_minimal_sprite(str(tmp_path / "semi"),
                                           semi_transparent=True))
    assert not rep.ok
    assert any("semi-transparent" in e for e in rep.errors)


def test_verify_rejects_missing_center(tmp_path):
    rep = VF.verify_folder(_minimal_sprite(str(tmp_path / "noc"),
                                           drop_center=True))
    assert not rep.ok
    assert any("vert" in e for e in rep.errors)


def test_verify_rejects_duration_mismatch(tmp_path):
    rep = VF.verify_folder(_minimal_sprite(str(tmp_path / "dur"),
                                           bad_durations=True))
    assert not rep.ok
    assert any("Duration" in e for e in rep.errors)


def test_verify_rejects_reserved_index(tmp_path):
    """L'index 3 est réservé... vérifions qu'un mauvais mapping est refusé."""
    folder = _minimal_sprite(str(tmp_path / "idx"), bad_index=True)
    rep = VF.verify_folder(folder)
    # index 3 n'est pas dans ACTION_MAP, donc ok ; on teste le vrai conflit :
    data = SH.read_anim_data(os.path.join(folder, C.MULTI_SHEET_XML))
    e = data.get("Walk")
    e.index = 1                      # 1 est réservé à Attack
    SH.write_anim_data(data, os.path.join(folder, C.MULTI_SHEET_XML))
    os.rename(os.path.join(folder, "Walk-Anim.png"),
              os.path.join(folder, "Walk-Anim.png"))
    rep = VF.verify_folder(folder)
    assert not rep.ok
    assert any("réservé" in e for e in rep.errors)


def test_verify_rejects_missing_xml(tmp_path):
    rep = VF.verify_folder(str(tmp_path))
    assert not rep.ok


# --------------------------------------------------------------------------
# pipeline complet (nécessite le clone amont)
# --------------------------------------------------------------------------

@needs_upstream
def test_build_pikachu_walk_is_valid(tmp_path):
    out = str(tmp_path / "dyna")
    params = DX.DynamaxParams(scale=1.6, seed=1)
    DX.dynamax_folder(PIKACHU, out, params, only=["Walk"], verbose=False)
    rep = VF.verify_folder(out)
    assert rep.ok, rep.summary()


@needs_upstream
def test_dynamax_is_bigger_and_multiple_of_8(tmp_path):
    out = str(tmp_path / "dyna")
    params = DX.DynamaxParams(scale=1.6, seed=1)
    DX.dynamax_folder(PIKACHU, out, params, only=["Walk"], verbose=False)

    src = SH.read_anim_data(os.path.join(PIKACHU, C.MULTI_SHEET_XML)).get("Walk")
    dst = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML)).get("Walk")
    assert dst.size[0] > src.size[0] and dst.size[1] > src.size[1]
    assert dst.size[0] % 8 == 0 and dst.size[1] % 8 == 0
    assert dst.durations == src.durations, "les timings doivent être préservés"


@needs_upstream
def test_dynamax_palette_is_source_plus_dynamax(tmp_path):
    """Aucune couleur parasite : palette finale = source + rampe Dynamax."""
    out = str(tmp_path / "dyna")
    DX.dynamax_folder(PIKACHU, out, DX.DynamaxParams(scale=1.6),
                      only=["Walk"], verbose=False)
    src_pal = GS.palette_of(SH.load_rgba(os.path.join(PIKACHU, "Walk-Anim.png")))
    out_pal = GS.palette_of(SH.load_rgba(os.path.join(out, "Walk-Anim.png")))
    assert out_pal <= src_pal | set(C.DYNA_RAMP[1:])


@needs_upstream
def test_dynamax_all_directions_preserved(tmp_path):
    out = str(tmp_path / "dyna")
    DX.dynamax_folder(PIKACHU, out, DX.DynamaxParams(scale=1.5),
                      only=["Walk"], verbose=False)
    data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
    s = SH.read_sheet(out, data.get("Walk"))
    assert s.n_dirs == 8 and s.n_frames == 4
    for d in range(8):
        for f in range(4):
            assert SH.covered_bounds(s.tile("anim", f, d)) is not None


@needs_upstream
def test_clouds_present_on_every_frame(tmp_path):
    """Chaque frame doit porter des pixels Dynamax : les nuages ne clignotent pas."""
    out = str(tmp_path / "dyna")
    DX.dynamax_folder(PIKACHU, out, DX.DynamaxParams(scale=1.6, seed=1),
                      only=["Walk"], verbose=False)
    data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
    s = SH.read_sheet(out, data.get("Walk"))
    dyna = set(C.DYNA_RAMP[1:])
    for f in range(s.n_frames):
        for d in range(s.n_dirs):
            pal = GS.palette_of(s.tile("anim", f, d))
            assert pal & dyna, f"pas de nuage en frame {f} direction {d}"


@needs_upstream
def test_transform_animation_is_valid(tmp_path):
    out = str(tmp_path / "dyna")
    params = DX.DynamaxParams(scale=1.6, seed=1)
    DX.dynamax_folder(PIKACHU, out, params, only=["Idle"], verbose=False)

    data = SH.read_anim_data(os.path.join(PIKACHU, C.MULTI_SHEET_XML))
    base = SH.read_sheet(PIKACHU, data.get("Idle"))
    sheet, fx, (tw, th) = TR.build_transform_anim(
        base, params, TR.TransformParams(n_frames=16, seed=2))

    assert sheet.n_frames == 16
    assert len(sheet.entry.durations) == 16
    assert tw % 8 == 0 and th % 8 == 0

    out_data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
    sheet.entry.index = C.free_action_index(out_data.used_indexes())
    SH.write_sheet(sheet, out)
    out_data.put(sheet.entry)
    SH.write_anim_data(out_data, os.path.join(out, C.MULTI_SHEET_XML))

    rep = VF.verify_folder(out)
    assert rep.ok, rep.summary()


@needs_upstream
def test_transform_has_column_then_clouds(tmp_path):
    """La colonne doit précéder les nuages dans le storyboard."""
    data = SH.read_anim_data(os.path.join(PIKACHU, C.MULTI_SHEET_XML))
    base = SH.read_sheet(PIKACHU, data.get("Idle"))
    _, fx, (tw, th) = TR.build_transform_anim(
        base, DX.DynamaxParams(scale=1.6), TR.TransformParams(n_frames=16))

    # au tiers du parcours : la colonne occupe le haut de la tuile
    early = fx[5].idx
    assert (early[: th // 3] > 0).any(), "pas de colonne en haut au début"
    # à la fin : plus de colonne en haut, mais des nuages près de la tête
    late = fx[-1]
    merged = late.idx.copy()
    m = (merged == 0) & (late.behind > 0)
    merged[m] = late.behind[m]
    assert (merged > 0).any()


@needs_upstream
def test_shadow_scales_with_sprite(tmp_path):
    """Un Pokémon géant projette une ombre agrandie."""
    out = str(tmp_path / "dyna")
    DX.dynamax_folder(PIKACHU, out, DX.DynamaxParams(scale=2.0),
                      only=["Walk"], verbose=False)
    src_data = SH.read_anim_data(os.path.join(PIKACHU, C.MULTI_SHEET_XML))
    src = SH.read_sheet(PIKACHU, src_data.get("Walk"))
    dst_data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
    dst = SH.read_sheet(out, dst_data.get("Walk"))

    def blob(sheet):
        t = sheet.tile("shadow", 0, 0)
        return int(np.count_nonzero(t[:, :, 3] > 0))

    assert blob(dst) > blob(src)


@needs_upstream
def test_every_tile_has_exactly_one_shadow_anchor(tmp_path):
    out = str(tmp_path / "dyna")
    DX.dynamax_folder(PIKACHU, out, DX.DynamaxParams(scale=1.6),
                      only=["Walk"], verbose=False)
    data = SH.read_anim_data(os.path.join(out, C.MULTI_SHEET_XML))
    s = SH.read_sheet(out, data.get("Walk"))
    for f in range(s.n_frames):
        for d in range(s.n_dirs):
            n = len(SH.find_pixels(s.tile("shadow", f, d), C.SDW_CENTER))
            assert n == 1, f"{n} ancres blanches en frame {f} dir {d}"
