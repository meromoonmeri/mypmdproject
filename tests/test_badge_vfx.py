"""VFX d'apparition du badge : jointure, pureté, lecture de la séquence."""
import numpy as np
import pytest

from soulhalo.badge_vfx import BadgeReveal, BurstParams, _bell, _ss

W, H = 160, 120           # petit mais pas minuscule : une resolution trop
                          # basse a deja menti sur une mesure de halo


@pytest.fixture(scope="module")
def r():
    return BadgeReveal(w=W, h=H, scale=1, nature="jolly", nom="Evoli")


# --- la jointure : le point critique ---------------------------------------
def test_la_fin_du_burst_est_exactement_le_debut_de_la_boucle(r):
    """Un fondu qui CONVERGE vers la boucle sans l'atteindre laisse un saut
    d'une frame, invisible en GIF mais bien visible en jeu."""
    assert np.array_equal(r.burst(1.0), r.idle(0.0))


def test_la_boucle_du_badge_boucle(r):
    assert np.array_equal(r.idle(0.0), r.idle(1.0))


def test_le_raccord_est_progressif(r):
    """La derniere marche avant la boucle doit etre petite : sinon la
    jointure est exacte mais precedee d'un a-coup."""
    a, b = r.burst(0.97), r.burst(1.0)
    assert float(np.abs(a - b).mean()) < 0.05


# --- pureté ----------------------------------------------------------------
def test_burst_est_pur(r):
    """Un rendu a effet de bord rendrait la sequence non rejouable."""
    assert np.array_equal(r.burst(0.5), r.burst(0.5))
    assert np.array_equal(r.burst(0.5), r.burst(0.5))


def test_idle_est_pur(r):
    assert np.array_equal(r.idle(0.33), r.idle(0.33))


def test_u_est_borne(r):
    """u hors [0,1] ne doit pas exploser : le jeu peut deborder d'une frame."""
    assert np.array_equal(r.burst(-0.5), r.burst(0.0))
    assert np.array_equal(r.burst(1.7), r.burst(1.0))


# --- la séquence se lit-elle ? ---------------------------------------------
def test_la_ruee_precede_le_badge(r):
    """Au debut il n'y a pas encore de badge : l'ecran est sombre."""
    assert r.burst(0.0).mean() < 0.10


def test_le_flash_est_le_sommet_lumineux(r):
    """L'impact doit dominer toute la sequence, sinon il ne se voit pas."""
    p = r.p
    lums = {u: float(r.burst(u).mean()) for u in
            (0.0, 0.2, 0.4, 0.55, p.flash_at, 0.85, 1.0)}
    assert lums[p.flash_at] == max(lums.values())
    assert lums[p.flash_at] > 0.75          # un vrai voile blanc


def test_l_energie_retombe_apres_l_impact(r):
    """Le badge se pose dans le calme, pas dans le bruit."""
    assert r.burst(0.90).mean() < r.burst(r.p.flash_at).mean() * 0.6


def test_le_badge_est_pose_a_la_fin(r):
    """Le centre doit porter le badge, donc etre plus clair que les bords."""
    f = r.burst(1.0)
    h, w = f.shape[:2]
    centre = f[h // 2 - 12:h // 2 + 12, w // 2 - 20:w // 2 + 20].mean()
    bord = np.concatenate([f[:6].ravel(), f[-6:].ravel()]).mean()
    assert centre > bord


# --- couleur de la nature ---------------------------------------------------
def test_chaque_nature_donne_une_couleur_distincte(r):
    """Piege deja rencontre : empiler teinte + accent + lueur sature les
    trois canaux et toutes les natures redeviennent blanches."""
    vus = {}
    for nat in ("hardy", "jolly", "calm", "impish", "lonely"):
        r.set_nature(nat, "Evoli")
        f = r.idle(0.0)
        h, w = f.shape[:2]
        vus[nat] = f[h // 2 - 15:h // 2 + 15,
                     w // 2 - 25:w // 2 + 25].reshape(-1, 3).mean(0)
    r.set_nature("jolly", "Evoli")
    ks = list(vus)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            assert float(np.abs(vus[a] - vus[b]).max()) > 0.10, f"{a} vs {b}"


def test_nature_inconnue_refusee(r):
    with pytest.raises(KeyError):
        r.set_nature("nawak")


# --- UI DX ------------------------------------------------------------------
def test_le_bandeau_dx_apparait_a_la_fin(r):
    """Le bandeau ne doit pas etre la pendant la ruee : il se pose apres."""
    bas_debut = r.burst(0.2)[-40:]
    bas_fin = r.burst(1.0)[-40:]
    assert bas_fin.std() > bas_debut.std()


def test_sans_bandeau_l_ecran_reste_propre():
    """L'UI est optionnelle : le VFX doit pouvoir servir seul."""
    r2 = BadgeReveal(w=W, h=H, scale=1, nature="calm",
                     params=BurstParams(panel=False))
    assert np.array_equal(r2.burst(1.0), r2.idle(0.0))


# --- agrandissement ---------------------------------------------------------
def test_agrandissement_entier():
    r2 = BadgeReveal(w=W, h=H, scale=3, nature="jolly")
    im = r2.burst_image(1.0)
    assert im.size == (W * 3, H * 3)


# --- helpers ----------------------------------------------------------------
def test_smoothstep_scalaire_borne():
    assert _ss(0.2, 0.8, 0.0) == 0.0
    assert _ss(0.2, 0.8, 1.0) == 1.0
    assert _ss(0.5, 0.5, 0.9) == 1.0        # bornes egales : pas de division
    assert 0.0 < _ss(0.0, 1.0, 0.5) < 1.0


def test_cloche_centree():
    assert _bell(0.5, 0.5, 0.1) == pytest.approx(1.0)
    assert _bell(0.0, 0.5, 0.1) < 0.01
