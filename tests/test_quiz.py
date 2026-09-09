"""Test de personnalité : progression, pondération, confidentialité, écran."""
import json

import numpy as np
import pytest

from soulhalo.quiz import QuizState, load_natures, load_questions
from soulhalo.quizscreen import QuizScreen, wrap


@pytest.fixture(scope="module")
def data():
    return load_questions()


# --- la regle absolue : jamais de points ------------------------------------
def test_aucun_point_dans_l_api_publique():
    """Regle du projet : le joueur ne voit JAMAIS de score.

    Rien de public ne doit exposer les totaux - ni attribut, ni methode.
    """
    q = QuizState(seed=3)
    interdits = ("poids", "score", "points", "total_poids", "scores")
    for a in dir(q):
        if a.startswith("_"):
            continue
        assert a not in interdits, f"l'API expose {a}"


def test_le_resultat_ne_contient_aucun_chiffre_de_score():
    q = QuizState(seed=3)
    while not q.fini:
        q.answer(0)
    r = q.result_data()
    assert set(r) == {"nature", "fr", "rgb", "accent"}


def test_l_ecran_n_affiche_aucun_score():
    """Le bandeau montre « QUESTION 3 / 8 », un repere - pas un score."""
    s = QuizScreen(seed=3)
    s.valider()
    s.valider()
    q = s.quiz
    assert q.numero == 3 and q.total == 8
    # le repere ne depend d'aucun poids : il ne bouge pas selon les reponses
    a = QuizScreen(seed=3)
    b = QuizScreen(seed=3)
    a.valider()
    b.move(1)
    b.valider()
    assert a.quiz.numero == b.quiz.numero


# --- progression ------------------------------------------------------------
def test_seule_la_reponse_fait_avancer():
    """Aucun timer : rendre mille frames ne doit rien changer."""
    s = QuizScreen(seed=5)
    avant = s.quiz.numero
    for i in range(50):
        s.render(i / 50.0, tick=i)
        s.step()
    assert s.quiz.numero == avant
    s.valider()
    assert s.quiz.numero == avant + 1


def test_le_test_se_termine():
    q = QuizState(seed=7)
    n = 0
    while not q.fini and n < 100:
        q.answer(0)
        n += 1
    assert q.fini and n == q.total


def test_resultat_refuse_avant_la_fin():
    q = QuizState(seed=7)
    with pytest.raises(RuntimeError):
        q.result()


def test_retour_en_arriere_annule_la_ponderation():
    """Sans annulation des poids, reculer puis repondre autrement
    cumulerait les DEUX reponses."""
    a = QuizState(seed=11)
    a.answer(0)
    a.back()
    a.answer(1)
    while not a.fini:
        a.answer(0)

    b = QuizState(seed=11)
    b.answer(1)
    while not b.fini:
        b.answer(0)
    assert a.result() == b.result()


def test_back_au_debut_ne_plante_pas():
    q = QuizState(seed=11)
    q.back()
    assert q.numero == 1


def test_choix_hors_bornes_borne():
    q = QuizState(seed=11)
    q.answer(99)
    assert q.numero == 2


# --- tirage et donnees ------------------------------------------------------
def test_tirage_sans_remise():
    q = QuizState(seed=13)
    ids = [x.id for x in q.questions]
    assert len(ids) == len(set(ids))


def test_meme_graine_meme_test():
    a = QuizState(seed=17)
    b = QuizState(seed=17)
    assert [x.id for x in a.questions] == [x.id for x in b.questions]


def test_graines_differentes_donnent_des_tests_differents():
    vus = {tuple(x.id for x in QuizState(seed=s).questions) for s in range(12)}
    assert len(vus) > 1


def test_chaque_question_a_au_moins_deux_reponses(data):
    for q in data["questions"]:
        assert len(q["reponses"]) >= 2
        for r in q["reponses"]:
            assert r["label"] and r["poids"]


def test_les_poids_visent_des_natures_connues(data):
    connues = {n["id"] for n in load_natures()}
    for q in data["questions"]:
        for r in q["reponses"]:
            for nat in r["poids"]:
                assert nat in connues, f"{q['id']} : nature inconnue {nat}"


# --- equilibrage ------------------------------------------------------------
def test_les_13_natures_sont_atteignables():
    """Une nature qu'aucun parcours ne donne serait du contenu mort."""
    rng = np.random.default_rng(0)
    vues = set()
    for _ in range(2500):
        q = QuizState(seed=int(rng.integers(1_000_000)))
        while not q.fini:
            q.answer(int(rng.integers(q.question.n)))
        vues.add(q.result())
    assert len(vues) == 13, f"jamais atteintes : {set(load_natures_ids()) - vues}"


def load_natures_ids():
    return [n["id"] for n in load_natures()]


def test_la_distribution_est_equilibree():
    """Piege corrige : 'calm' servait de reponse prudente par defaut dans
    11 questions sur 18 et sortait 21 fois plus souvent que 'impish'."""
    rng = np.random.default_rng(1)
    from collections import Counter
    c = Counter()
    N = 2500
    for _ in range(N):
        q = QuizState(seed=int(rng.integers(1_000_000)))
        while not q.fini:
            q.answer(int(rng.integers(q.question.n)))
        c[q.result()] += 1
    assert max(c.values()) / min(c.values()) < 4.0


def test_les_poids_sont_repartis(data):
    """Aucune nature ne doit ecraser les autres dans les donnees."""
    from collections import Counter
    tot = Counter()
    for q in data["questions"]:
        for r in q["reponses"]:
            for nat, w in r["poids"].items():
                tot[nat] += w
    assert max(tot.values()) / min(tot.values()) < 2.0


# --- rendu ------------------------------------------------------------------
def test_le_rendu_est_pur():
    s = QuizScreen(seed=19)
    assert np.array_equal(s.render(0.3, tick=4), s.render(0.3, tick=4))


def test_le_fond_boucle():
    s = QuizScreen(seed=19)
    assert np.array_equal(s.render(0.0, tick=0), s.render(1.0, tick=0))


def test_chaque_question_est_un_etat_independant():
    """Deux questions differentes doivent rendre deux images differentes."""
    s = QuizScreen(seed=19)
    a = s.render(0.2, tick=0)
    s.valider()
    b = s.render(0.2, tick=0)
    assert not np.array_equal(a, b)


def test_le_curseur_est_borne():
    s = QuizScreen(seed=19)
    s.move(-5)
    assert s.choix == 0
    s.move(99)
    assert s.choix == s.quiz.question.n - 1


def test_decoupe_du_texte():
    lignes = wrap("un deux trois quatre cinq six sept huit neuf dix", 60)
    assert len(lignes) > 1
    from soulhalo.dxui import text_width
    for ln in lignes:
        assert text_width(ln) <= 60


def test_mot_plus_long_que_la_boite():
    """Un mot seul trop long ne doit pas boucler a l'infini."""
    lignes = wrap("anticonstitutionnellement", 20)
    assert len(lignes) == 1
