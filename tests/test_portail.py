"""Caméra du portail pendant les questions.

La règle du projet est ici mise à l'épreuve sous sa forme la plus
exposée : le joueur a une entrée continue (la souris) au milieu d'une
séquence qui doit boucler et ne progresser que sur validation.

    ANIMATION = BOUCLE INFINIE
    REGARD    = LIBRE, SANS EFFET SUR LA PROGRESSION
    DIALOGUE  = SEULE PROGRESSION
"""
import numpy as np
import pytest

from soulhalo.dxui import MenuStyle, pmdo_style
from soulhalo.portalcam import Camera, PortalView, SoulPath, _rampe
from soulhalo.quiz import QuizState
from soulhalo.quizscreen import QuizScreen


# --------------------------------------------------------------------------
# bouclage
# --------------------------------------------------------------------------
def test_le_portail_boucle_au_bit_pres():
    v = PortalView(96, 72, pixel=False)
    c = Camera()
    assert np.array_equal(v.render(0.0, c), v.render(1.0, c))


def test_le_portail_boucle_aussi_camera_tournee():
    """Le bouclage ne doit pas dépendre de l'angle de vue."""
    v = PortalView(96, 72, pixel=False)
    c = Camera()
    for _ in range(40):
        c.update(0.8, -0.6)
    assert np.array_equal(v.render(0.0, c), v.render(1.0, c))


def test_le_portail_bouge_vraiment():
    """Une boucle exacte ne doit pas être une image fixe."""
    v = PortalView(96, 72, pixel=False)
    c = Camera()
    assert not np.array_equal(v.render(0.0, c), v.render(0.5, c))


def test_la_trajectoire_de_l_ame_se_referme():
    p = SoulPath()
    a = np.array(p.position(0.0))
    b = np.array(p.position(1.0))
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------
# la caméra oriente, elle ne fait pas avancer
# --------------------------------------------------------------------------
def test_render_est_pur():
    """`render` ne doit muter ni la caméra ni quoi que ce soit."""
    v = PortalView(64, 48, pixel=False)
    c = Camera()
    c.update(0.7, -0.2)
    avant = (c.yaw, c.pitch)
    v.render(0.4, c)
    assert (c.yaw, c.pitch) == avant


def test_tourner_la_tete_change_la_vue():
    v = PortalView(96, 72, pixel=False)
    g, d = Camera(), Camera()
    for _ in range(40):
        g.update(-1.0, 0.0)
        d.update(1.0, 0.0)
    a, b = v.render(0.25, g), v.render(0.25, d)
    assert np.abs(a - b).mean() > 0.01


def test_le_regard_est_borne():
    """Le joueur regarde autour de lui, il ne fait pas de tonneau."""
    c = Camera()
    for _ in range(200):
        c.update(5.0, -5.0)          # entrée volontairement hors bornes
    yaw, pitch = c.rad
    assert abs(yaw) <= c.max_yaw * 2 * np.pi + 1e-6
    assert abs(pitch) <= c.max_pitch * 2 * np.pi + 1e-6


def test_la_camera_est_lissee():
    """Un saut du pointeur ne doit pas se traduire par un saut de vue."""
    c = Camera()
    c.update(1.0, 1.0)
    assert abs(c.yaw) < 0.5          # une seule frame : chemin partiel
    for _ in range(60):
        c.update(1.0, 1.0)
    assert c.yaw == pytest.approx(1.0, abs=0.02)


def test_le_portail_reste_en_vue_a_fond_de_course():
    """Aux amplitudes maximales, la bouche du tunnel doit rester visible.

    Défaut mesuré à l'image lors du premier réglage : à grande amplitude
    on ne voyait plus qu'un mur oblique.
    """
    v = PortalView(96, 72, pixel=False)
    for mx, my in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
        c = Camera()
        for _ in range(60):
            c.update(mx, my)
        img = v.render(0.3, c)
        # le point de fuite est un pic lumineux : il doit être dans le cadre
        assert img.max() > 0.55, (mx, my)


# --------------------------------------------------------------------------
# la sphère-âme
# --------------------------------------------------------------------------
def test_l_ame_est_visible_pendant_toute_la_boucle():
    v = PortalView(160, 120, pixel=False)
    c = Camera()
    assert all(v.visible(p / 12.0, c) for p in range(12))


def test_l_ame_se_voit_sous_plusieurs_angles():
    """Le point de l'exigence : la sphère doit se déplacer à l'écran quand
    on tourne la tête, sinon c'est un décor plat."""
    v = PortalView(160, 120, pixel=False)
    g, d = Camera(), Camera()
    for _ in range(60):
        g.update(-1.0, 0.0)
        d.update(1.0, 0.0)
    pg = v._projette(v.path.position(0.2), g)
    pd = v._projette(v.path.position(0.2), d)
    assert pg is not None and pd is not None
    assert abs(pg[0] - pd[0]) > 8      # décalage horizontal net


def test_l_ame_grossit_quand_elle_approche():
    v = PortalView(160, 120, pixel=False)
    p = v.path
    proche = min((p.position(i / 24.0)[2] for i in range(24)))
    loin = max((p.position(i / 24.0)[2] for i in range(24)))
    assert proche < loin


def test_le_spectre_du_halo_n_est_pas_un_arc_en_ciel():
    """Bleu nuit -> cyan -> turquoise -> violet -> magenta -> blanc.
    Ni rouge pur ni jaune : ce n'est pas un arc-en-ciel."""
    t = np.linspace(0, 1, 64, endpoint=False)
    c = _rampe(t)
    rouge = (c[:, 0] > 0.75) & (c[:, 1] < 0.35) & (c[:, 2] < 0.35)
    jaune = (c[:, 0] > 0.75) & (c[:, 1] > 0.70) & (c[:, 2] < 0.35)
    assert not rouge.any()
    assert not jaune.any()


def test_le_spectre_boucle():
    assert np.allclose(_rampe(np.array([0.0])), _rampe(np.array([1.0])))


# --------------------------------------------------------------------------
# intégration dans l'écran de question
# --------------------------------------------------------------------------
def test_l_ecran_de_question_boucle():
    s = QuizScreen(seed=3, pixel=False)
    assert np.array_equal(s.render(0.0), s.render(1.0))


def test_bouger_la_souris_ne_repond_pas_a_la_question():
    """LE test de la règle : une entrée continue au milieu du dialogue ne
    doit jamais faire progresser le test."""
    s = QuizScreen(seed=3, pixel=False)
    n0, q0 = s.quiz.numero, s.quiz.question.texte
    for i in range(120):
        s.step(mouse=(np.sin(i / 7.0), np.cos(i / 5.0)))
        s.render(i / 120.0)
    assert s.quiz.numero == n0
    assert s.quiz.question.texte == q0


def test_seule_la_validation_avance():
    s = QuizScreen(seed=3, pixel=False)
    n0 = s.quiz.numero
    s.valider()
    assert s.quiz.numero == n0 + 1


def test_le_stick_pilote_la_meme_camera():
    s = QuizScreen(seed=3, pixel=False)
    for _ in range(40):
        s.look_at(mouse=(1.0, 0.0))
    assert s.cam.yaw > 0.5


def test_l_ecran_utilise_l_ui_pmdo_par_defaut():
    """Pendant le test, l'UI est celle du moteur, pas la transposition DX."""
    s = QuizScreen(seed=3, pixel=False)
    assert s.frame.s.fill_hi == pmdo_style().fill_hi
    assert s.frame.s.hatch == 0.0            # PMDO n'a pas de hachures
    assert s.frame.s.fill_hi != MenuStyle().fill_hi


def test_le_texte_contraste_avec_son_cerne():
    """Défaut constaté à l'image : texte clair cerné de clair, illisible.
    Le cerne doit toujours s'opposer à la couleur du texte."""
    for st in (pmdo_style(), MenuStyle()):
        for texte, cerne in ((st.text, st.text_outline),
                             (st.text_on_cursor, st.cursor)):
            lt = float(np.mean(texte))
            lc = float(np.mean(cerne))
            assert abs(lt - lc) > 0.25, (texte, cerne)


def test_aucun_score_n_apparait_avec_le_portail():
    """Le fond a changé, la règle ne bouge pas : jamais de points."""
    s = QuizScreen(seed=3, pixel=False)
    public = [a for a in dir(s) if not a.startswith("_")]
    for interdit in ("poids", "score", "points", "total_poids", "scores"):
        assert not any(interdit in a.lower() for a in public)
