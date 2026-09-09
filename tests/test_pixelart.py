"""Couche pixel art : tramage, palette bornée, relief 2.5D."""
import numpy as np

from soulhalo import pixelart as PX
from soulhalo.dxui import MenuFrame, new_screen, upscale


def _rampe(h=32, w=64):
    """Dégradé horizontal lisse, le pire cas pour les bandes."""
    g = np.linspace(0.0, 1.0, w, dtype=np.float32)
    return np.repeat(np.repeat(g[None, :, None], h, 0), 3, 2).copy()


# --- tramage ----------------------------------------------------------------
def test_le_tramage_borne_le_nombre_de_valeurs():
    a = PX.dither(_rampe(), levels=8)
    assert len(np.unique(np.round(a * 255))) <= 8 * 2


def test_le_tramage_casse_les_bandes():
    """Sans trame, quantifier cree des colonnes uniformes. La trame doit
    introduire de la variation VERTICALE dans chaque colonne."""
    r = _rampe()
    plat = np.round(r * 7) / 7
    trame = PX.dither(r, levels=8)
    assert trame.std(axis=0).mean() > plat.std(axis=0).mean()


def test_le_tramage_conserve_la_moyenne():
    """Une trame ordonnee ne doit pas eclaircir ni assombrir l'image."""
    r = _rampe()
    assert abs(float(PX.dither(r, levels=10).mean() - r.mean())) < 0.02


def test_le_tramage_est_deterministe():
    """Une trame aleatoire gresillerait d'une frame a l'autre."""
    r = _rampe()
    assert np.array_equal(PX.dither(r, levels=8), PX.dither(r, levels=8))


def test_tramage_desactive():
    r = _rampe()
    assert np.array_equal(PX.dither(r, levels=1), np.clip(r, 0, 1))


def test_la_matrice_de_bayer_est_bien_formee():
    b = PX.bayer(8, 8, 4)
    assert b.shape == (8, 8)
    assert 0.0 <= b.min() and b.max() < 1.0
    assert len(np.unique(PX.BAYER4)) == 16      # 16 seuils distincts


# --- palette ----------------------------------------------------------------
def test_la_quantification_borne_les_couleurs():
    pal = PX.ramp((0.6, 0.45, 0.23), n=5)
    out = PX.quantize(_rampe(), pal)
    couleurs = {tuple(c) for c in out.reshape(-1, 3)}
    assert len(couleurs) <= 5


def test_la_rampe_va_du_sombre_au_clair():
    r = PX.ramp((0.5, 0.4, 0.2), n=5)
    lum = r.mean(1)
    assert np.all(np.diff(lum) > 0)


# --- relief 2.5D ------------------------------------------------------------
def test_le_relief_eclaire_en_haut_et_ombre_en_bas():
    """C'est la convention 2.5D : lumiere en haut-gauche."""
    m = np.zeros((20, 20), np.float32)
    m[5:15, 5:15] = 1.0
    base = np.full((20, 20, 3), 0.5, np.float32)
    out = PX.emboss(base, m, light=0.3, shade=0.3)
    assert out[5, 8].mean() > 0.5           # arete haute eclairee
    assert out[14, 8].mean() < 0.5          # arete basse ombree


def test_le_relief_ne_touche_pas_l_interieur():
    m = np.zeros((20, 20), np.float32)
    m[4:16, 4:16] = 1.0
    base = np.full((20, 20, 3), 0.5, np.float32)
    out = PX.emboss(base, m, light=0.3, shade=0.3)
    assert np.allclose(out[10, 10], 0.5), "le coeur du panneau reste plat"


def test_l_ombre_portee_est_decalee_et_hors_forme():
    m = np.zeros((24, 24), np.float32)
    m[6:16, 6:16] = 1.0
    base = np.ones((24, 24, 3), np.float32)
    out = PX.drop_shadow(base, m, dx=2, dy=2, opacity=0.5)
    assert out[17, 10].mean() < 1.0         # ombre en bas
    assert out[10, 10].mean() == 1.0        # rien SOUS la forme
    assert out[3, 3].mean() == 1.0          # rien en haut-gauche


def test_l_ombre_portee_est_tramee():
    """Une ombre pleine parait lourde a cette echelle."""
    m = np.zeros((24, 24), np.float32)
    m[4:14, 4:14] = 1.0
    out = PX.drop_shadow(np.ones((24, 24, 3), np.float32), m, opacity=0.6)
    zone = out[14:17, 6:14]
    assert len(np.unique(np.round(zone * 100))) > 1     # alterne


def test_le_contour_cerne_la_forme():
    m = np.zeros((16, 16), np.float32)
    m[6:10, 6:10] = 1.0
    out = PX.outline(np.ones((16, 16, 3), np.float32), m, color=(0, 0, 0))
    assert out[5, 7].sum() == 0.0           # cerne juste au-dessus
    assert out[7, 7].sum() == 3.0           # interieur intact


# --- integration avec l'UI --------------------------------------------------
def test_les_panneaux_dx_ont_du_relief():
    buf = new_screen(60, 40)
    buf[:] = 0.25
    MenuFrame().panel(buf, 8, 8, 44, 24)
    haut = buf[9, 12:48].mean()
    bas = buf[30, 12:48].mean()
    assert haut > bas, "le panneau doit etre eclaire par le haut"


def test_les_panneaux_dx_portent_une_ombre():
    a = new_screen(60, 40)
    a[:] = 0.5
    b = a.copy()
    MenuFrame().panel(a, 8, 8, 40, 20)
    # sous le bord bas du panneau, hors de sa surface
    assert a[29, 20:40].mean() < b[29, 20:40].mean()


def test_le_tramage_survit_a_l_agrandissement():
    """La trame doit etre posee AVANT l'agrandissement : appliquee apres,
    elle ferait des points de 3x3 pixels au lieu d'une trame fine."""
    buf = new_screen(40, 30)
    buf[:] = 0.3
    MenuFrame().panel(buf, 4, 4, 32, 22)
    gros = upscale(buf, 3)
    # un bloc 3x3 issu de l'agrandissement est uniforme
    bloc = gros[15:18, 15:18]
    assert np.allclose(bloc, bloc[0, 0])
