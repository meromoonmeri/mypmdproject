"""Tests du pipeline de zones de boss.

Ce qui est vérifié ici, ce n'est pas « l'image est jolie » — c'est que les
contraintes exploitables en jeu tiennent : détourage sans halo, palette
bornée, calques superposables, boucle d'animation exacte.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from soulhalo import zoneassets as ZA
from soulhalo import zonecompose as ZC

RACINE = "zone_boss"


# --------------------------------------------------------------------------
# données
# --------------------------------------------------------------------------
def test_les_19_zones_sont_declarees():
    d = json.load(open(f"{RACINE}/zones.json", encoding="utf-8"))
    assert len(d["zones"]) == 19


def test_chaque_zone_a_six_couleurs_valides():
    d = json.load(open(f"{RACINE}/zones.json", encoding="utf-8"))
    for z in d["zones"]:
        assert len(z["palette"]) == 6, z["legendaire"]
        for h in z["palette"]:
            assert len(h) == 7 and h[0] == "#"
            int(h[1:], 16)              # lève si ce n'est pas de l'hexa


# --------------------------------------------------------------------------
# détourage
# --------------------------------------------------------------------------
def _planche_magenta():
    """Sujet gris au centre, fond magenta légèrement bruité.

    Le bruit est là exprès : le générateur ne rend jamais un aplat
    parfaitement uniforme, et un détourage par égalité exacte échouerait.
    """
    a = np.zeros((40, 40, 3), np.float32)
    a[..., 0] = 1.0
    a[..., 2] = 1.0
    rng = np.random.default_rng(0)
    a += rng.normal(0, 0.01, a.shape).astype(np.float32)
    a[12:28, 12:28] = 0.5
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")


def test_le_detourage_rend_le_fond_transparent():
    r = ZA.key_out(_planche_magenta(), ronge=0)
    assert r[2, 2, 3] == 0.0
    assert r[20, 20, 3] == 1.0


def test_le_detourage_ne_laisse_aucun_pixel_magenta_visible():
    """Le halo de bord est le défaut classique du chroma key."""
    r = ZA.key_out(_planche_magenta(), ronge=1)
    vis = r[..., 3] > 0.0
    rgb = r[..., :3]
    magenta = (rgb[..., 0] > 0.6) & (rgb[..., 1] < 0.35) & (rgb[..., 2] > 0.6)
    assert not (vis & magenta).any()


def test_le_detourage_du_noir_donne_un_alpha_progressif():
    """Une particule additive n'a pas de contour net : son alpha suit sa
    luminosité, sinon on lui coupe son dégradé."""
    a = np.zeros((16, 16, 3), np.float32)
    a[8, 8] = 1.0
    a[8, 9] = 0.5
    r = ZA.key_out_black(Image.fromarray((a * 255).astype(np.uint8), "RGB"))
    assert r[8, 8, 3] == pytest.approx(1.0, abs=0.02)
    assert 0.3 < r[8, 9, 3] < 0.7
    assert r[0, 0, 3] == 0.0


# --------------------------------------------------------------------------
# nettoyage pixel art
# --------------------------------------------------------------------------
def test_la_quantification_borne_la_palette():
    rng = np.random.default_rng(1)
    a = rng.random((32, 32, 4)).astype(np.float32)
    a[..., 3] = 1.0
    q = ZA.quantize_palette(a, n=8)
    cols = np.unique((q[..., :3] * 255).astype(np.uint8).reshape(-1, 3),
                     axis=0)
    assert len(cols) <= 8


def test_la_quantification_preserve_l_alpha():
    a = np.zeros((16, 16, 4), np.float32)
    a[..., :3] = 0.7
    a[4:12, 4:12, 3] = 1.0
    q = ZA.quantize_palette(a, n=4)
    assert np.array_equal(q[..., 3], a[..., 3])


def test_la_grille_reduit_bien_la_resolution():
    a = np.ones((64, 64, 4), np.float32)
    assert ZA.snap_grid(a, 4).shape[:2] == (16, 16)


# --------------------------------------------------------------------------
# découpe
# --------------------------------------------------------------------------
def test_la_decoupe_separe_des_objets_disjoints():
    a = np.zeros((40, 80, 4), np.float32)
    a[5:15, 5:15, 3] = 1.0
    a[20:35, 40:60, 3] = 1.0
    b = ZA.decoupe(a, min_px=10)
    assert len(b) == 2


def test_la_decoupe_recadre_sur_la_boite_exacte():
    a = np.zeros((40, 40, 4), np.float32)
    a[10:20, 12:30, 3] = 1.0
    (x0, y0, x1, y1), = ZA.decoupe(a, min_px=10)
    assert (x0, y0, x1, y1) == (12, 10, 30, 20)


def test_la_decoupe_ignore_le_bruit():
    """Deux pixels perdus ne sont pas un objet."""
    a = np.zeros((40, 40, 4), np.float32)
    a[5:20, 5:20, 3] = 1.0
    a[30, 30, 3] = 1.0
    assert len(ZA.decoupe(a, min_px=40)) == 1


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------
def test_le_blit_respecte_l_alpha():
    dst = np.zeros((10, 10, 4), np.float32)
    dst[..., :3] = 1.0
    dst[..., 3] = 1.0
    src = np.zeros((4, 4, 4), np.float32)          # noir opaque
    src[..., 3] = 1.0
    ZA.blit(dst, src, 2, 2)
    assert dst[3, 3, :3].max() == pytest.approx(0.0, abs=1e-5)
    assert dst[0, 0, :3].min() == pytest.approx(1.0, abs=1e-5)


def test_le_blit_hors_champ_ne_plante_pas():
    dst = np.zeros((10, 10, 4), np.float32)
    src = np.ones((4, 4, 4), np.float32)
    ZA.blit(dst, src, -20, -20)
    ZA.blit(dst, src, 50, 50)
    assert dst[..., 3].max() == 0.0


def test_le_pavage_est_periodique():
    t = np.random.default_rng(2).random((8, 8, 4)).astype(np.float32)
    f = ZA.tile_fill(t, 32, 32)
    assert np.array_equal(f[0:8, 0:8], f[8:16, 8:16])


# --------------------------------------------------------------------------
# calques
# --------------------------------------------------------------------------
def test_tous_les_calques_ont_la_taille_de_la_scene():
    """Condition pour pouvoir en décocher un sans rien recalculer."""
    pal = [(0.1, 0.1, 0.2)] * 6
    for c in (ZC.calque_ciel(pal), ZC.calque_arene(pal),
              ZC.calque_eclairage(pal), ZC.calque_bordure(), ZC.vide()):
        assert c.shape == (ZC.SCENE_H, ZC.SCENE_W, 4)


def test_le_ciel_est_opaque_et_le_reste_ne_l_est_pas():
    pal = [(0.1, 0.1, 0.2)] * 6
    assert ZC.calque_ciel(pal)[..., 3].min() == 1.0
    assert ZC.calque_bordure()[..., 3].min() == 0.0


def test_l_ambiance_ne_change_que_la_couleur():
    """Une ambiance est un réglage colorimétrique : l'alpha ne bouge pas."""
    pal = [(0.4, 0.3, 0.2)] * 6
    base = ZC.calque_ciel(pal)
    for amb in ZC.AMBIANCES:
        out = ZC.applique_ambiance(base, amb)
        assert np.array_equal(out[..., 3], base[..., 3])
    assert ZC.applique_ambiance(base, "sombre")[..., :3].mean() < \
        base[..., :3].mean()
    assert ZC.applique_ambiance(base, "cataclysme")[..., :3].mean() > \
        base[..., :3].mean()


def test_la_parallaxe_va_du_fond_au_premier_plan():
    p = [x[2] for x in ZC.CALQUES]
    assert p[0] < p[1] < 1.0
    assert max(p) > 1.0            # les FX passent devant


def test_un_seul_calque_est_anime():
    assert sum(1 for c, _, _ in ZC.CALQUES if c == "07_fx") == 1


# --------------------------------------------------------------------------
# bouclage
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(f"{RACINE}/Groudon/source/07_fx.png"),
                    reason="planche FX absente")
def test_la_boucle_des_particules_est_exacte():
    """La règle transversale du projet : phase 0 et phase 1 identiques au
    bit près, sinon la boucle saute d'une frame."""
    a = ZC.calque_fx(f"{RACINE}/Groudon", 5, 0.0)
    b = ZC.calque_fx(f"{RACINE}/Groudon", 5, 1.0)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not os.path.exists(f"{RACINE}/Groudon/source/07_fx.png"),
                    reason="planche FX absente")
def test_les_frames_intermediaires_different():
    """Une boucle exacte ne doit pas être une image fixe."""
    a = ZC.calque_fx(f"{RACINE}/Groudon", 5, 0.0)
    b = ZC.calque_fx(f"{RACINE}/Groudon", 5, 0.5)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
# livrables
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(f"{RACINE}/kit.json"),
                    reason="kit absent")
def test_le_kit_decrit_les_dix_calques():
    k = json.load(open(f"{RACINE}/kit.json", encoding="utf-8"))
    assert len(k["calques"]) == 10
    assert k["moteur"]["tuile"] == 24
    assert k["moteur"]["scene"] == [320, 240]


@pytest.mark.skipif(not os.path.isdir(f"{RACINE}/calques"),
                    reason="calques absents")
def test_les_calques_ecrits_sont_transparents_et_calibres():
    for zone in os.listdir(f"{RACINE}/calques"):
        for amb in os.listdir(f"{RACINE}/calques/{zone}"):
            d = f"{RACINE}/calques/{zone}/{amb}"
            for f in os.listdir(d):
                im = Image.open(os.path.join(d, f))
                assert im.mode == "RGBA", f
                assert im.size == (ZC.SCENE_W, ZC.SCENE_H), f
