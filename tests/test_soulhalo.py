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
