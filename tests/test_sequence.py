"""La sequence d'intro : chaque etat boucle, chaque transition raccorde."""
import numpy as np
import pytest

from soulhalo.sequence import (StateHalo, StatePortal, StateSky, StateVoid,
                               Transition)

W, H = 240, 136


@pytest.fixture(scope="module")
def states():
    return [StateVoid(W, H), StateSky(W, H), StateHalo(W, H), StatePortal(W, H)]


def test_chaque_etat_boucle_exactement(states):
    """Le joueur peut rester indefiniment : la boucle doit etre parfaite."""
    for s in states:
        assert np.array_equal(s.render(0.0), s.render(1.0)), s.name


def test_chaque_etat_est_anime(states):
    """Un etat fige donnerait l'impression que le jeu a plante."""
    for s in states:
        assert not np.array_equal(s.render(0.0), s.render(0.25)), s.name


def test_aucun_a_coup_dans_les_boucles(states):
    for s in states:
        n = 12
        fr = [s.render(i / n) for i in range(n)]
        d = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
        assert max(d) / (sum(d) / len(d)) < 1.7, s.name


def test_progression_lumineuse():
    """Le scenario va du noir vers la lumiere : la sequence doit s'eclaircir."""
    lums = [s(W, H).render(0.0).mean()
            for s in (StateVoid, StateSky, StateHalo)]
    assert lums[0] < lums[1] < lums[2], lums


def test_vide_est_bien_noir():
    a = StateVoid(W, H).render(0.0)
    assert a.mean() < 0.06
    # ... mais la sphere doit briller au centre
    cy, cx = H // 2, W // 2
    k = max(3, min(H, W) // 14)
    assert a[cy - k:cy + k, cx - k:cx + k].mean() > 5 * a.mean()


def test_ciel_est_bleu_nuit():
    a = StateSky(W, H).render(0.0)
    r, g, b = a.mean((0, 1))
    assert b > r, "le ciel doit tirer vers le bleu"
    assert a.mean() < 0.35, "le ciel doit rester sombre"


def test_halo_est_un_arc_en_ciel():
    """Le spectre doit vraiment etre etale, pas deux teintes voisines."""
    import colorsys
    a = StateHalo(W, H).render(0.0)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    hues = []
    for lo in (0.15, 0.45, 0.75, 1.05):
        m = (rr >= lo) & (rr < lo + 0.22)
        if m.sum():
            hues.append(colorsys.rgb_to_hsv(*a[m].mean(0))[0] * 360)
    ecarts = [abs(b - x) for x, b in zip(hues, hues[1:])]
    assert max(ecarts) > 30, f"spectre trop resserre : {hues}"


def test_halo_est_sature():
    a = StateHalo(W, H).render(0.0)
    mx, mn = a.max(2), a.min(2)
    v = mx > 0.05
    assert float(((mx - mn) / np.maximum(mx, 1e-6))[v].mean()) > 0.5


def test_la_sphere_est_partout_au_centre(states):
    """L'ame est le fil de la sequence : elle doit briller dans chaque etat.

    On echantillonne un petit disque a l'echelle du NOYAU (rayon ~0.05 de
    demi-hauteur). Une boite plus large englobe le puits sombre qui entoure
    la sphere dans l'etat halo et donne un faux negatif.
    """
    for s in states:
        a = s.render(0.0).mean(2)
        cy, cx = H // 2, W // 2
        k = max(2, int(0.05 * min(H, W) / 2))
        centre = a[cy - k:cy + k + 1, cx - k:cx + k + 1].mean()
        assert centre > a.mean(), (s.name, centre, a.mean())
        # Critere robuste : le pixel le PLUS BRILLANT de l'image doit se
        # trouver sur la sphere. Comparer des moyennes ne convient pas -
        # le portail est un tunnel lumineux, sa moyenne est deja haute, et
        # le halo entoure la sphere d'un puits sombre.
        yy, xx = np.unravel_index(int(np.argmax(a)), a.shape)
        assert np.hypot(xx - cx, yy - cy) <= 0.06 * min(H, W), s.name


def test_transition_part_et_arrive_sur_les_boucles(states):
    """Sans ce raccord, le joueur verrait un saut a chaque changement."""
    s1, s2 = states[0], states[1]
    t = Transition(s1, s2, 12)
    assert np.allclose(t.render(0), s1.render(0.0), atol=1e-6)
    assert np.allclose(t.render(11), s2.render(0.0), atol=1e-6)


def test_transition_est_progressive(states):
    t = Transition(states[1], states[2], 16)
    fr = [t.render(i) for i in range(16)]
    d = [float(np.abs(fr[i + 1] - fr[i]).mean()) for i in range(15)]
    assert max(d) / (sum(d) / len(d)) < 2.6


def test_flash_blanc_culmine_au_milieu(states):
    """Le flash doit monter puis redescendre, pas rester allume."""
    t = Transition(states[1], states[2], 21, flash=0.9, flash_at=0.5)
    lum = [t.render(i).mean() for i in range(21)]
    assert lum[10] > lum[0] and lum[10] > lum[-1]


def test_sans_flash_pas_de_voile(states):
    t = Transition(states[0], states[1], 9, flash=0.0)
    mid = t.render(4)
    ref = 0.5 * (states[0].render(0.0) + states[1].render(0.0))
    assert np.abs(mid - ref).mean() < 0.12


def test_deterministe():
    assert np.array_equal(StateHalo(W, H).render(0.3),
                          StateHalo(W, H).render(0.3))
