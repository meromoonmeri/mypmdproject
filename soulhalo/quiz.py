"""Moteur du test de personnalité : état, pondération, résultat.

RÈGLE ABSOLUE — le joueur ne voit JAMAIS de points.

Les poids existent, ils décident de la nature, mais rien de l'interface
publique ne les expose : pas de score, pas de barre de progression
chiffrée, pas de « nature en tête ». `QuizState` garde ses totaux dans un
attribut privé et n'offre aucune méthode pour les lire avant la fin. C'est
un choix de conception, pas une omission — dans PMD le test est une
conversation, pas un questionnaire noté.

La progression dépend UNIQUEMENT de la réponse du joueur : `answer()`
avance d'une question, rien d'autre. Aucun timer n'existe ici.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def load_questions(path="personality_test/questions.json"):
    with open(_p(path), encoding="utf-8") as f:
        return json.load(f)


def load_natures(path="personality_test/natures.json"):
    with open(_p(path), encoding="utf-8") as f:
        return json.load(f)["natures"]


@dataclass
class Question:
    id: str
    texte: str
    reponses: list

    @property
    def n(self):
        return len(self.reponses)


class QuizState:
    """Déroulé du test. Une question à la fois, avancée par le joueur.

    `seed` rend le tirage reproductible : deux parties avec la même graine
    posent les mêmes questions, ce qui permet de tester le moteur. En jeu,
    on laisse la graine à None pour tirer au hasard.
    """

    def __init__(self, data=None, natures=None, seed=None, n_questions=None):
        d = data if data is not None else load_questions()
        self.natures = natures if natures is not None else load_natures()
        self.ids_natures = [n["id"] for n in self.natures]
        toutes = [Question(q["id"], q["texte"], q["reponses"])
                  for q in d["questions"]]
        k = int(n_questions or d.get("tirage", 8))
        k = max(1, min(k, len(toutes)))
        rng = np.random.default_rng(seed)
        # tirage SANS remise : on ne repose jamais deux fois la même
        idx = rng.permutation(len(toutes))[:k]
        self.questions = [toutes[i] for i in sorted(idx)]
        self.index = 0
        self.reponses = []                  # historique des choix
        # Totaux INTERNES. Volontairement prive : rien dans l'API publique
        # ne les expose avant la fin du test.
        self._poids = {n: 0.0 for n in self.ids_natures}

    # -- déroulé -----------------------------------------------------------
    @property
    def question(self):
        """Question courante, ou None si le test est fini."""
        if self.fini:
            return None
        return self.questions[self.index]

    @property
    def fini(self):
        return self.index >= len(self.questions)

    @property
    def total(self):
        return len(self.questions)

    @property
    def numero(self):
        """Numéro affichable : « question 3 sur 8 ». Ce n'est pas un score."""
        return min(self.index + 1, self.total)

    def answer(self, choix: int):
        """Enregistre une réponse et avance. SEUL moyen de progresser."""
        if self.fini:
            return self
        q = self.questions[self.index]
        c = int(np.clip(choix, 0, q.n - 1))
        for nat, w in q.reponses[c]["poids"].items():
            if nat in self._poids:
                self._poids[nat] += float(w)
        self.reponses.append((q.id, c))
        self.index += 1
        return self

    def back(self):
        """Revient sur la question précédente et annule son effet.

        Sans annulation des poids, un joueur qui recule puis répond
        autrement cumulerait les deux réponses.
        """
        if self.index <= 0:
            return self
        self.index -= 1
        qid, c = self.reponses.pop()
        q = next(q for q in self.questions if q.id == qid)
        for nat, w in q.reponses[c]["poids"].items():
            if nat in self._poids:
                self._poids[nat] -= float(w)
        return self

    # -- résultat ----------------------------------------------------------
    def result(self):
        """Nature obtenue. Disponible seulement une fois le test fini.

        En cas d'égalité, on départage par l'ordre des natures plutôt que
        par un tirage : deux parcours identiques doivent donner le même
        résultat, sinon le test n'est pas reproductible.
        """
        if not self.fini:
            raise RuntimeError("le test n'est pas termine")
        best = max(self.ids_natures, key=lambda n: (self._poids[n],
                                                    -self.ids_natures.index(n)))
        return best

    def result_data(self):
        """Le résultat complet, tel que l'interface l'affichera.

        Ne contient AUCUN point : seulement la nature et sa couleur.
        """
        nat = self.result()
        n = next(x for x in self.natures if x["id"] == nat)
        return {"nature": nat, "fr": n.get("fr", nat), "rgb": n["rgb"],
                "accent": n["accent"]}
