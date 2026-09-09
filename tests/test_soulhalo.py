"""Tests de la parallaxe circulaire.

L'exigence centrale : l'animation doit boucler **exactement** et bouger
**progressivement**. Ces deux propriétés sont vérifiées au pixel près.
"""
import numpy as np
import pytest

from soulhalo import field as F
from soulhalo.halo import HaloParams, HaloRenderer, SoulParams, default_layers

PLATE = "research/plates/plate_nebula.png"
W, H = 240, 135


@pytest.fixture(scope="module")
def r():
    return HaloRenderer(PLATE, w=W, h=H)


# --- champ ----------------------------------------------------------------
def test_smoothstep_accepte_bornes_decroissantes():
    """Une vignette s'exprime avec e1 < e0 ; le resultat doit s'inverser."""
    x = np.array([0.0, 0.5, 1.0], np.float32)
    up = F.smoothstep(0.0, 1.0, x)
    down = F.smoothstep(1.0, 0.0, x)
    assert up[0] == pytest.approx(0.0) and up[-1] == pytest.approx(1.0)
    assert down[0] == pytest.approx(1.0) and down[-1] == pytest.approx(0.0)


def test_polar_centre_et_bords():
    rn, th = F.polar(64, 64)
    assert rn[32, 32] < 0.05
    assert rn[0, 32] == pytest.approx(1.0, abs=0.05)
    assert -np.pi <= th.min() and th.max() <= np.pi


def test_strip_est_cyclique_en_theta():
    """La bande polaire doit se refermer : derniere colonne ~ premiere."""
    s = F.polar_strip(PLATE, n_theta=256, n_rad=64)
    assert s.shape == (64, 256, 3)
    wrap = np.abs(s[:, -1] - s[:, 0]).mean()
    interne = np.abs(s[:, 128] - s[:, 0]).mean()
    assert wrap < interne, "la bande doit etre continue au raccord"


def test_blur_conserve_la_moyenne():
    a = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert F.blur(a, 3.0).mean() == pytest.approx(a.mean(), abs=0.02)


# --- bouclage -------------------------------------------------------------
def test_boucle_exacte(r):
    """phase 1.0 doit redonner phase 0.0 au bit pres."""
    assert np.array_equal(r.render(0.0), r.render(1.0))


def test_phase_repliee(r):
    assert np.array_equal(r.render(0.25), r.render(1.25))


def test_ame_boucle(r):
    assert r.soul_xy(0.0) == pytest.approx(r.soul_xy(1.0), abs=1e-6)


def test_toutes_les_vitesses_sont_entieres():
    """Une vitesse non entiere casserait le raccord de boucle."""
    for L in default_layers():
        assert float(L.speed).is_integer()


# --- mouvement ------------------------------------------------------------
def test_animation_progressive(r):
    """Aucun a-coup : le plus grand ecart reste proche de la moyenne."""
    n = 16
    fr = [r.render(i / n) for i in range(n)]
    d = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
    assert min(d) > 0, "l'image doit bouger"
    assert max(d) < 2.4 * (sum(d) / len(d)), "saut brutal detecte"


def test_ame_parcourt_son_orbite(r):
    pts = [r.soul_xy(i / 8) for i in range(8)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert max(xs) - min(xs) > 0.5, "l'ame doit voyager horizontalement"
    assert max(ys) - min(ys) > 0.3, "et verticalement"


def test_orbite_est_elliptique(r):
    """L'ellipse donne la perspective : l'amplitude verticale est moindre."""
    pts = [r.soul_xy(i / 32) for i in range(32)]
    assert (max(p[1] for p in pts) - min(p[1] for p in pts)) < \
           (max(p[0] for p in pts) - min(p[0] for p in pts))


def test_parallaxe_les_couches_ne_bougent_pas_ensemble():
    """Le coeur meme de la parallaxe : des vitesses distinctes et opposees."""
    sp = [L.speed for L in default_layers()]
    assert len(set(sp)) == len(sp), "chaque couche a sa propre vitesse"
    assert any(s > 0 for s in sp) and any(s < 0 for s in sp), "sens opposes"
    assert all(abs(a) < abs(b) for a, b in zip(sp, sp[1:])), \
        "la vitesse doit croitre vers le centre"


# --- rendu ----------------------------------------------------------------
def test_sortie_valide(r):
    a = r.render(0.3)
    assert a.shape == (H, W, 3)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert not np.isnan(a).any()


def test_image_coloree(r):
    """Le halo doit rester colore, pas virer au blanc."""
    a = r.render(0.0)
    mx, mn = a.max(2), a.min(2)
    vis = mx > 0.08
    sat = ((mx - mn) / np.maximum(mx, 1e-6))[vis]
    assert sat.mean() > 0.20, "halo delave"


def test_fond_noir_sur_les_bords(r):
    a = r.render(0.0)
    assert a[0, 0].mean() < 0.10 and a[-1, -1].mean() < 0.10


def test_ame_est_le_point_le_plus_brillant(r):
    """La sphere doit dominer : c'est le sujet de la scene."""
    a = r.render(0.0)
    lum = a.mean(2)
    yy, xx = np.unravel_index(int(np.argmax(lum)), lum.shape)
    sx, sy = r.soul_xy(0.0)
    half = min(H, W) / 2.0
    ex = (W - 1) / 2.0 + sx * half
    ey = (H - 1) / 2.0 + sy * half
    assert np.hypot(xx - ex, yy - ey) < 0.20 * half


def test_deterministe(r):
    assert np.array_equal(r.render(0.42), r.render(0.42))


def test_grain_ne_scintille_pas():
    """Le grain est fixe : deux rendus consecutifs partagent le meme bruit."""
    r2 = HaloRenderer(PLATE, w=W, h=H)
    r3 = HaloRenderer(PLATE, w=W, h=H)
    assert np.array_equal(r2.grain, r3.grain)


def test_parametres_personnalises():
    p = HaloParams(soul=SoulParams(orbit_rx=0.2, orbit_ry=0.2, radius=0.05),
                   saturation=1.0, grain=0.0)
    rr = HaloRenderer(PLATE, w=120, h=120, params=p)
    assert np.array_equal(rr.render(0.0), rr.render(1.0))


# --- arc-en-ciel anime dans la parallaxe ----------------------------------
RAINBOW_PLATE = "research/plates/plate_rainbow_rings.png"


def test_hue_rotate_tour_entier_est_identite():
    """Indispensable au bouclage : sans identite exacte a 1 tour, la
    derniere frame ne se raccorde plus a la premiere."""
    rng = np.random.default_rng(0)
    x = rng.random((32, 32, 3)).astype(np.float32)
    for turns in (0.0, 1.0, -1.0, 2.0, -3.0):
        assert np.allclose(F.hue_rotate(x, turns), x, atol=1e-5), turns


def test_hue_rotate_preserve_la_luminance():
    rng = np.random.default_rng(1)
    x = rng.random((32, 32, 3)).astype(np.float32)
    w = np.array([0.299, 0.587, 0.114], np.float32)
    y = F.hue_rotate(x, 0.37)
    assert np.abs((y * w).sum(2) - (x * w).sum(2)).max() < 1e-4


def test_hue_rotate_change_bien_la_teinte():
    x = np.full((8, 8, 3), 0.0, np.float32)
    x[..., 0] = 0.9
    assert np.abs(F.hue_rotate(x, 1 / 3.0) - x).mean() > 0.05


def test_rainbow_boucle_exactement():
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                     params=HaloParams(rainbow=1, plate_offset=0.30))
    assert np.array_equal(r.render(0.0), r.render(1.0))


def test_rainbow_cycles_entiers():
    assert float(HaloParams().rainbow).is_integer()
    assert float(HaloParams(rainbow=1).rainbow).is_integer()


def test_rainbow_eteint_par_defaut():
    """L'arc-en-ciel ne doit pas s'imposer aux rendus existants : sur une
    plaque peinte il peut rendre une couronne aussi vive que l'ame."""
    assert HaloParams().rainbow == 0


def test_rainbow_fait_defiler_la_teinte():
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                     params=HaloParams(rainbow=1, plate_offset=0.30))
    assert np.abs(r.render(0.0) - r.render(1 / 3.0)).mean() > 0.02


def test_rainbow_desactivable():
    p = HaloParams(rainbow=0, plate_offset=0.30)
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H, params=p)
    assert np.array_equal(r.render(0.0), r.render(1.0))


def test_rainbow_etale_le_spectre_entre_couronnes():
    """L'arc-en-ciel doit etre DANS la parallaxe : des couronnes de rayons
    differents doivent porter des teintes differentes."""
    import colorsys
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                     params=HaloParams(rainbow=1, plate_offset=0.30))
    a = r.render(0.0)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    hues = []
    for lo in (0.15, 0.45, 0.75):
        c = a[(rr >= lo) & (rr < lo + 0.2)].mean(0)
        hues.append(colorsys.rgb_to_hsv(*c)[0] * 360)
    spread = max(hues) - min(hues)
    assert spread > 40, f"spectre trop resserre : {spread:.0f} deg"


def test_rainbow_reste_sature():
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                     params=HaloParams(rainbow=1, plate_offset=0.30))
    a = r.render(0.0)
    mx, mn = a.max(2), a.min(2)
    vis = mx > 0.05
    assert float(((mx - mn) / np.maximum(mx, 1e-6))[vis].mean()) > 0.35


def test_rainbow_sans_a_coup():
    r = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                     params=HaloParams(rainbow=1, plate_offset=0.30))
    n = 12
    fr = [r.render(i / n) for i in range(n)]
    d = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
    assert max(d) / (sum(d) / len(d)) < 1.6


def test_plate_offset_evite_le_coeur_sombre():
    """Sans decalage, toutes les couronnes lisent le coeur sombre de la
    plaque en anneaux et la couleur s'effondre."""
    base = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                        params=HaloParams(rainbow=1, plate_offset=0.0)).render(0.0)
    off = HaloRenderer(RAINBOW_PLATE, w=W, h=H,
                       params=HaloParams(rainbow=1, plate_offset=0.30)).render(0.0)

    def sat(a):
        mx, mn = a.max(2), a.min(2)
        v = mx > 0.05
        return float(((mx - mn) / np.maximum(mx, 1e-6))[v].mean())
    assert sat(off) > sat(base)
