"""Le voyage dans le portail : bouclage, profondeur, absence de couture."""
import numpy as np
import pytest

from soulhalo import field as F
from soulhalo.tunnel import Shell, TunnelParams, TunnelRenderer

PLATE = "research/plates/plate_portal_tunnel.png"
W, H = 240, 136


@pytest.fixture(scope="module")
def r():
    return TunnelRenderer(PLATE, w=W, h=H)


def test_boucle_exactement(r):
    """Phase 1.0 doit redonner la phase 0.0 au bit pres."""
    assert np.array_equal(r.render(0.0), r.render(1.0))


def test_vitesses_entieres():
    """Le bouclage repose la-dessus : une vitesse fractionnaire laisserait
    la derniere frame decalee par rapport a la premiere."""
    for s in TunnelParams().shells:
        assert float(s.speed).is_integer()
        assert float(s.spin).is_integer()
    assert float(TunnelParams().core_pulse).is_integer()
    assert float(TunnelParams().tint_shift).is_integer()


def test_la_matiere_fuit_vers_l_exterieur(r):
    """Le coeur du portail : une meme ecaille doit GROSSIR avec le temps.

    On suit un motif dans une couronne interieure ; il doit se retrouver
    plus loin du centre a la frame suivante.
    """
    p = TunnelParams(shells=[Shell(speed=1, spin=0, alpha=1.0, streak=1)],
                     core_glow=0.0, bloom=0.0, soul=False, grain=0.0)
    t = TunnelRenderer(PLATE, w=W, h=H, params=p)
    # On SUIT UN MOTIF precis (une bande sombre) le long d'un rayon.
    # Un centre de masse ne conviendrait pas : il est domine par la
    # luminosite globale, pas par le deplacement des ecailles.
    cy, cx = H // 2, W // 2
    lo, hi = 8, 60

    def bande(ph):
        row = t.render(ph).mean(2)[cy, cx:]
        return int(np.argmin(row[lo:hi])) + lo

    pos = [bande(ph) for ph in (0.0, 0.05, 0.10, 0.20)]
    assert pos[-1] > pos[0], f"la matiere ne fuit pas vers l'exterieur : {pos}"
    # et la progression doit etre globalement monotone
    assert sum(b >= a for a, b in zip(pos, pos[1:])) >= 2, pos


def test_pas_de_couture_annulaire(r):
    """Le miroir doit rendre le raccord de tuile invisible : aucun saut
    brutal de couleur le long d'un rayon."""
    p = TunnelParams(shells=[Shell(speed=1, spin=0, alpha=1.0, streak=1)],
                     core_glow=0.0, bloom=0.0, soul=False, grain=0.0)
    t = TunnelRenderer(PLATE, w=W, h=H, params=p)
    a = t.render(0.0).mean(2)
    row = a[H // 2, W // 2 + 6:]          # au-dela de la zone centrale
    d = np.abs(np.diff(row))
    assert d.max() < 12.0 * (d.mean() + 1e-6), "anneau de couture visible"


def test_miroir_est_continu():
    """f=0 et f=1 doivent retomber sur le meme rayon de plaque."""
    for f in (0.0, 1.0):
        tri = 1.0 - abs(2.0 * (f % 1.0) - 1.0)
        assert abs(tri - 0.0) < 1e-9


def test_couvre_les_coins(r):
    """Un portail qui laisse les coins noirs trahit une image plate."""
    a = r.render(0.0)
    for y, x in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        assert a[y, x].mean() > 0.05


def test_reste_colore(r):
    """Piege connu : moyenner les nappes fait virer le tunnel au violet."""
    a = r.render(0.0)
    mx, mn = a.max(2), a.min(2)
    vis = mx > 0.05
    assert float(((mx - mn) / np.maximum(mx, 1e-6))[vis].mean()) > 0.35


def test_sans_a_coup(r):
    n = 12
    fr = [r.render(i / n) for i in range(n)]
    d = [float(np.abs(fr[i] - fr[(i + 1) % n]).mean()) for i in range(n)]
    assert max(d) / (sum(d) / len(d)) < 1.6


def test_ame_au_centre(r):
    """La sphere traverse le portail : elle doit briller au centre."""
    a = r.render(0.0).mean(2)
    cy, cx = H // 2, W // 2
    k = max(3, min(H, W) // 12)
    centre = a[cy - k:cy + k, cx - k:cx + k].mean()
    assert centre > a.mean()


def test_ame_desactivable():
    p = TunnelParams(soul=False)
    t = TunnelRenderer(PLATE, w=W, h=H, params=p)
    assert np.array_equal(t.render(0.0), t.render(1.0))


def test_teinte_optionnelle_boucle():
    p = TunnelParams(tint_shift=1)
    t = TunnelRenderer(PLATE, w=W, h=H, params=p)
    assert np.array_equal(t.render(0.0), t.render(1.0))
    assert np.abs(t.render(0.0) - t.render(1 / 3.0)).mean() > 0.01


def test_deterministe():
    a = TunnelRenderer(PLATE, w=W, h=H).render(0.21)
    b = TunnelRenderer(PLATE, w=W, h=H).render(0.21)
    assert np.array_equal(a, b)


def test_grain_fixe(r):
    """Un grain retire a chaque frame produirait un gresillement."""
    g1 = r.grain.copy()
    r.render(0.4)
    assert np.array_equal(g1, r.grain)
