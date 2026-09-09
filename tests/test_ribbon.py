"""Le ruban de resultat : boucle, maintien, badge, natures."""
import colorsys

import numpy as np
import pytest
from PIL import Image

from soulhalo.ribbon import (RibbonRenderer, first_frame, load_natures,
                             paste_reveal)

W, H = 240, 136


@pytest.fixture(scope="module")
def r():
    return RibbonRenderer(w=W, h=H, nature="jolly")


def test_idle_boucle_exactement(r):
    """Le joueur peut rester indefiniment sur le resultat."""
    assert np.array_equal(r.idle(0.0), r.idle(1.0))


def test_idle_est_anime(r):
    assert not np.array_equal(r.idle(0.0), r.idle(0.25))


def test_idle_sans_a_coup(r):
    n = 12
    fr = [r.idle(i / n) for i in range(n)]
    d = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
    assert max(d) / (sum(d) / len(d)) < 1.7


def test_le_maintien_intensifie(r):
    """Maintenir le bouton doit se voir : la lumiere monte avec t."""
    lum = [r.charge(t, 0.0).mean() for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b > a for a, b in zip(lum, lum[1:])), lum


def test_relacher_revient_a_l_idle(r):
    """Relacher ne doit rien casser : t=0 rejoint exactement la boucle."""
    assert np.array_equal(r.charge(0.0, 0.3), r.idle(0.3))


def test_le_maintien_ne_casse_pas_la_boucle(r):
    """`t` est independant de `phase` : le fond continue de boucler."""
    for t in (0.3, 0.7, 1.0):
        assert np.array_equal(r.charge(t, 0.0), r.charge(t, 1.0))


def test_badge_present(r):
    """L'oeuf et les deux ailes : c'est la signature PMD du ruban."""
    assert r.badge is not None, "badge non charge"
    assert r.badge_a.max() > 0.9, "masque du badge vide"


def test_badge_occupe_le_centre(r):
    """Sans badge, le centre du ruban serait vide."""
    a = r.idle(0.0).mean(2)
    cy, cx = H // 2, W // 2
    k = min(H, W) // 8
    assert a[cy - k:cy + k, cx - k:cx + k].mean() > a.mean()


def test_badge_a_des_ailes():
    """Le badge doit etre LARGE (oeuf + deux ailes), pas un simple disque."""
    r2 = RibbonRenderer(w=W, h=H)
    bh, bw = r2.badge.shape[:2]
    assert bw > bh * 1.25, f"badge trop carre pour porter deux ailes : {bw}x{bh}"


def test_chaque_nature_a_sa_couleur():
    """La couleur du ruban doit suivre la nature, sans saturer en blanc."""
    nat = load_natures()
    r2 = RibbonRenderer(w=W, h=H)
    for nid in ("brave", "jolly", "docile", "calm", "impish"):
        r2.set_nature(nid)
        a = r2.charge(0.55, 0.0)
        cy, cx = H // 2, W // 2
        c = a[cy - 10:cy + 10, cx - 22:cx + 22].reshape(-1, 3).mean(0)
        hue = colorsys.rgb_to_hsv(*np.clip(c, 0, 1))[0] * 360
        tgt = nat[nid]["hue"]
        ecart = min(abs(hue - tgt), 360 - abs(hue - tgt))
        assert ecart < 50, (nid, hue, tgt)


def test_les_natures_sont_distinctes():
    r2 = RibbonRenderer(w=W, h=H)
    r2.set_nature("brave")
    a = r2.charge(0.6, 0.0)
    r2.set_nature("calm")
    b = r2.charge(0.6, 0.0)
    assert np.abs(a - b).mean() > 0.03


def test_quirky_est_instable():
    """Nature bizarre : la teinte doit parcourir le spectre."""
    r2 = RibbonRenderer(w=W, h=H, nature="quirky")
    assert r2.multicolore
    assert np.abs(r2.idle(0.0) - r2.idle(1 / 3.0)).mean() > 0.01
    assert np.array_equal(r2.idle(0.0), r2.idle(1.0))


def test_nature_inconnue_rejetee(r):
    with pytest.raises(KeyError):
        r.set_nature("inexistante")


def test_les_13_natures_existent():
    nat = load_natures()
    assert len(nat) == 13
    for n in ("hardy", "brave", "jolly", "docile", "calm", "quirky"):
        assert n in nat


def test_revelation_progressive(r):
    """On ne montre jamais le Pokemon d'un coup : silhouette puis couleur."""
    bg = r.charge(1.0, 0.0)
    sp = Image.new("RGBA", (32, 40), (200, 60, 40, 255))
    a0 = paste_reveal(bg, None, sp, 0.0)
    a5 = paste_reveal(bg, None, sp, 0.5)
    a1 = paste_reveal(bg, None, sp, 1.0)
    assert np.array_equal(a0, bg), "u=0 ne doit rien afficher"
    assert np.abs(a5 - bg).mean() > 0.001
    assert np.abs(a1 - bg).mean() > np.abs(a5 - bg).mean()


def test_revelation_ne_modifie_pas_le_fond(r):
    bg = r.idle(0.0)
    ref = bg.copy()
    paste_reveal(bg, None, Image.new("RGBA", (16, 16), (255, 0, 0, 255)), 1.0)
    assert np.array_equal(bg, ref), "paste_reveal doit rendre une copie"


def test_lecture_sprite_spritecollab():
    """La taille de case vient d'AnimData.xml, jamais devinee."""
    f = first_frame("personality_test/sprites/0004/Idle-Anim.png",
                    "personality_test/sprites/0004/AnimData.xml")
    assert f is not None
    assert f.size == (32, 40)
    assert f.mode == "RGBA"


def test_sprite_conserve_la_transparence():
    f = first_frame("personality_test/sprites/0007/Idle-Anim.png",
                    "personality_test/sprites/0007/AnimData.xml")
    a = np.asarray(f)
    assert a[..., 3].min() == 0, "le sprite doit avoir un fond transparent"


def test_anim_absente_rend_none():
    assert first_frame("personality_test/sprites/0004/Idle-Anim.png",
                       "personality_test/sprites/0004/AnimData.xml",
                       anim="AnimationQuiNExistePas") is None
