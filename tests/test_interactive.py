"""Souris, fond de reve, carrousel : l'entree joueur ne casse jamais la boucle."""
import numpy as np
import pytest

from soulhalo.interactive import (Carousel, DreamBackdrop, LookState,
                                  load_types)

# Resolution de test volontairement proche de la production. A 240x136 le
# sprite agrandi (128 px) occupe plus de la moitie de la largeur : la
# vignette et l'ecretage absorbent alors le halo, et les mesures ne
# refletent plus ce que le joueur voit.
W, H = 480, 270


@pytest.fixture(scope="module")
def d():
    return DreamBackdrop(w=W, h=H)


@pytest.fixture(scope="module")
def T():
    return load_types()


@pytest.fixture(scope="module")
def c(T):
    return Carousel(T["roster"], T, w=W, h=H)


# --- la souris ------------------------------------------------------------
def test_souris_lissee():
    """Le pointeur saute, la camera non : un seul pas ne doit pas suffire."""
    L = LookState()
    L.update(1.0, 1.0)
    assert L.x < 0.5, "la camera suit trop brutalement"
    for _ in range(60):
        L.update(1.0, 1.0)
    assert L.x > 0.95, "la camera n'atteint jamais la cible"


def test_souris_bornee():
    L = LookState()
    for _ in range(80):
        L.update(5.0, -5.0)
    assert -1.001 <= L.x <= 1.001 and -1.001 <= L.y <= 1.001


def test_offset_borne_par_max_tilt():
    L = LookState(max_tilt=0.3)
    for _ in range(80):
        L.update(1.0, 1.0)
    ox, oy = L.offset
    assert abs(ox) <= 0.3001 and abs(oy) <= 0.3001


# --- le fond de reve ------------------------------------------------------
def test_fond_boucle_exactement(d):
    assert np.array_equal(d.render(0.0), d.render(1.0))


def test_fond_boucle_meme_avec_la_souris(d):
    """C'est le point capital : bouger la souris ne fait pas avancer le temps."""
    L = LookState()
    for _ in range(40):
        L.update(0.8, -0.5)
    assert np.array_equal(d.render(0.0, L), d.render(1.0, L))


def test_la_souris_change_le_point_de_vue(d):
    L = LookState()
    for _ in range(40):
        L.update(1.0, 0.6)
    assert np.abs(d.render(0.0) - d.render(0.0, L)).mean() > 0.01


def test_parallaxe_les_plans_proches_bougent_plus(d):
    """Sans gradient de deplacement, il n'y a pas de relief."""
    L1, L2 = LookState(), LookState()
    for _ in range(40):
        L1.update(0.3, 0.0)
        L2.update(1.0, 0.0)
    a = np.abs(d.render(0.0) - d.render(0.0, L1)).mean()
    b = np.abs(d.render(0.0) - d.render(0.0, L2)).mean()
    assert b > a


def test_le_spectre_defile(d):
    assert np.abs(d.render(0.0) - d.render(1 / 3.0)).mean() > 0.02


def test_spectre_en_cycles_entiers():
    from soulhalo.interactive import DreamParams
    p = DreamParams()
    assert float(p.spectrum_cycles).is_integer()
    assert float(p.breathe).is_integer()
    for _, _, sp in p.rings:
        assert float(sp).is_integer()


def test_fond_sature(d):
    a = d.render(0.0)
    mx, mn = a.max(2), a.min(2)
    v = mx > 0.05
    assert float(((mx - mn) / np.maximum(mx, 1e-6))[v].mean()) > 0.5


def test_fond_sans_a_coup(d):
    n = 12
    fr = [d.render(i / n) for i in range(n)]
    dd = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
    assert max(dd) / (sum(dd) / len(dd)) < 1.7


# --- le carrousel ---------------------------------------------------------
def test_sprites_charges(c, T):
    for mon in T["roster"]:
        assert c.sprite(mon["dex"], 0) is not None, mon["dex"]


def test_sprites_transparents(c):
    a = np.asarray(c.sprite("0004", 0))
    assert a.shape[2] == 4 and a[..., 3].min() == 0


def test_animation_suit_les_durees_du_xml(c):
    """PMD n'anime pas a cadence fixe : chaque frame a sa duree propre."""
    dur = c.idle_frames("0004")
    assert dur == [12, 8, 8, 8]
    assert c.frame_at("0004", 0) == 0
    assert c.frame_at("0004", 11) == 0      # encore la 1re a t=11
    assert c.frame_at("0004", 12) == 1      # bascule a t=12
    assert c.frame_at("0004", 36) == 0      # boucle sur le total


def test_animation_boucle_sur_le_total(c):
    total = sum(c.idle_frames("0025"))
    assert c.frame_at("0025", 0) == c.frame_at("0025", total)


def test_carrousel_defile(c, d):
    bg = d.render(0.0)
    a = c.render(bg, 3.0, 0.0)
    b = c.render(bg, 4.0, 0.0)
    assert np.abs(a - b).mean() > 0.005


def test_survol_ajoute_un_halo(c, d):
    bg = d.render(0.0)
    base = c.render(bg, 3.0, 0.0)
    hov = c.render(bg, 3.0, 0.0, hover=3)
    assert np.abs(hov - base).mean() > 0.002
    assert hov.sum() > base.sum(), "le halo doit AJOUTER de la lumiere"


def test_selection_ajoute_une_aura(c, d):
    bg = d.render(0.0)
    base = c.render(bg, 3.0, 0.0)
    sel = c.render(bg, 3.0, 0.0, selected=3)
    assert np.abs(sel - base).mean() > 0.004


def test_aura_prend_la_couleur_du_type(c, d, T):
    """Pikachu = Electrik : l'aura doit tirer vers le jaune, pas le bleu."""
    bg = d.render(0.0)
    base = c.render(bg, 3.0, 0.0)
    sel = c.render(bg, 3.0, 0.0, selected=3)
    h, w = base.shape[:2]
    reg = (slice(h // 2 - 24, h // 2 + 24), slice(w // 2 - 32, w // 2 + 32))
    diff = (sel[reg] - base[reg]).reshape(-1, 3).mean(0)
    assert diff[0] > diff[2] and diff[1] > diff[2], diff


def test_types_differents_donnent_auras_differentes(c, d):
    bg = d.render(0.0)
    a = c.render(bg, 3.0, 0.0, selected=3)      # Pikachu, electrik
    b = c.render(bg, 3.0, 0.0, selected=2)      # Carapuce, eau
    assert np.abs(a - b).mean() > 0.002


def test_le_carrousel_ne_casse_pas_la_boucle(c, d):
    a = c.render(d.render(0.0), 3.0, 0.0, hover=3, selected=2, tick=0)
    b = c.render(d.render(1.0), 3.0, 1.0, hover=3, selected=2, tick=0)
    assert np.array_equal(a, b)


def test_carrousel_ne_modifie_pas_le_fond(c, d):
    bg = d.render(0.0)
    ref = bg.copy()
    c.render(bg, 3.0, 0.0, hover=1, selected=2)
    assert np.array_equal(bg, ref), "render doit travailler sur une copie"


def test_bornes_du_roster(c, d):
    """Aux extremites, il n'y a pas de voisin : ne pas deborder."""
    bg = d.render(0.0)
    for pos in (0.0, len(c.roster) - 1.0):
        out = c.render(bg, pos, 0.0)
        assert np.isfinite(out).all()


def test_les_18_types_existent(T):
    assert len(T["types"]) == 18
    for t in ("grass", "fire", "water", "electric", "fighting"):
        assert t in T["types"]
