"""UI DX : constantes du moteur PMDO, vignettes, molette, roster complet."""
import numpy as np
import pytest

from soulhalo import dxui
from soulhalo.dxui import (LINE_HEIGHT, PORTRAIT_SIZE, TITLE_OFFSET,
                           VERT_SPACE, ChoiceList, MenuFrame, PokemonCard,
                           SpriteBank, WheelState, draw_text, load_roster,
                           new_screen, text_width, upscale)
from soulhalo.selector import SelectorScreen


@pytest.fixture(scope="module")
def R():
    return load_roster()


@pytest.fixture(scope="module")
def bank():
    return SpriteBank()


# --- constantes du moteur -------------------------------------------------
def test_constantes_pmdo():
    """Relevees dans RogueCollab/RogueEssence : ne pas les inventer."""
    assert VERT_SPACE == 14
    assert LINE_HEIGHT == 12
    assert TITLE_OFFSET == 16
    assert PORTRAIT_SIZE == 40


def test_ecran_logique_pmdo():
    assert (dxui.SCREEN_W, dxui.SCREEN_H) == (320, 240)


def test_hauteur_de_liste_suit_la_formule_moteur():
    """BaseSettingsMenu : n * VERT_SPACE + bordures + ContentOffset."""
    cl = ChoiceList()
    assert cl.height(3, titled=True) == 3 * VERT_SPACE + 4 + TITLE_OFFSET
    assert cl.height(3, titled=False) == 3 * VERT_SPACE + 4


# --- roster ---------------------------------------------------------------
def test_les_42_starters_halcyon(R):
    assert len(R["roster"]) == 42


def test_chaque_starter_a_sprite_et_portrait(R, bank):
    for mon in R["roster"]:
        assert bank.sprite(mon["dex"]) is not None, mon["dex"]
        assert bank.portrait(mon["dex"]) is not None, mon["dex"]


def test_portraits_a_la_taille_du_moteur(R, bank):
    for mon in R["roster"][:12]:
        assert bank.portrait(mon["dex"]).size == (PORTRAIT_SIZE, PORTRAIT_SIZE)


def test_chaque_starter_a_talents_et_type(R):
    for mon in R["roster"]:
        assert mon["type"] in R["types"], mon
        assert len(mon["talents"]) >= 1, mon


def test_formes_alternatives_presentes(R):
    """Goupix d'Alola, Miaouss d'Alola, Zorua de Hisui ont leur propre entree."""
    dex = {m["dex"] for m in R["roster"]}
    assert {"0037-1", "0052-1", "0570-1"} <= dex, sorted(dex)


# --- sprites --------------------------------------------------------------
def test_animation_suit_les_durees_reelles(bank):
    d = bank.durations("0004")
    assert d == [12, 8, 8, 8]
    assert bank.frame_index("0004", 11) == 0
    assert bank.frame_index("0004", 12) == 1
    assert bank.frame_index("0004", 36) == 0


def test_sprites_transparents(bank):
    a = np.asarray(bank.sprite("0004"))
    assert a.shape[2] == 4 and a[..., 3].min() == 0


def test_carte_assez_grande_pour_le_plus_grand_sprite(R, bank):
    """Bug corrige : Pikachu (40x56) debordait d'une carte de 64 px."""
    p = PokemonCard(bank).p
    mx = max(bank.sprite(m["dex"]).height for m in R["roster"])
    besoin = p.gap + PORTRAIT_SIZE + p.gap + 2 + mx
    assert p.h >= besoin, (p.h, besoin)
    mw = max(bank.sprite(m["dex"]).width for m in R["roster"])
    assert p.w >= mw


# --- dessin ---------------------------------------------------------------
def test_cadre_dessine():
    buf = new_screen(80, 60)
    MenuFrame().draw(buf, 4, 4, 60, 40, title="Test")
    assert buf.sum() > 0
    assert buf[0, 0].sum() == 0, "ne doit pas deborder du cadre"


def test_texte_dessine():
    buf = new_screen(80, 20)
    draw_text(buf, "ABC", 2, 2)
    assert buf.sum() > 0


def test_texte_largeur_coherente():
    assert text_width("AB") == 2 * (dxui.GLYPH_W + 1) - 1


def test_texte_hors_ecran_ne_plante_pas():
    buf = new_screen(20, 12)
    draw_text(buf, "TRES LONG TEXTE", 15, 8)
    assert np.isfinite(buf).all()


def test_upscale_est_entier_et_net():
    a = np.zeros((2, 2, 3), np.float32)
    a[0, 0] = 1.0
    b = upscale(a, 3)
    assert b.shape == (6, 6, 3)
    # plus proche voisin : que des 0 ou des 1, aucune valeur interpolee
    assert set(np.unique(b).tolist()) <= {0.0, 1.0}


def test_vignette_dessine_portrait_et_sprite(R, bank):
    buf = new_screen(80, 120)
    card = PokemonCard(bank)
    card.draw(buf, R["roster"][0]["dex"], 4, 4, tick=0)
    haut = buf[6:44].sum()      # zone du portrait
    bas = buf[48:100].sum()     # zone du sprite
    assert haut > 0 and bas > 0


# --- molette --------------------------------------------------------------
def test_molette_a_de_l_inertie():
    w = WheelState(n=42)
    w.scroll(1)
    v0 = w.vel
    w.update()
    assert w.pos > 0
    assert 0 < w.vel < v0, "la vitesse doit decroitre, pas s'arreter net"


def test_molette_s_arrete():
    w = WheelState(n=42)
    w.scroll(1)
    for _ in range(200):
        w.update()
    assert w.vel == 0.0


def test_molette_bornee_au_roster():
    w = WheelState(n=42)
    for _ in range(50):
        w.scroll(5)
        w.update()
    assert w.pos <= 41
    for _ in range(200):
        w.scroll(-5)
        w.update()
    assert w.pos >= 0


def test_position_continue():
    w = WheelState(n=42)
    w.scroll(1)
    w.update()
    assert w.pos != round(w.pos) or w.pos == 0


# --- ecran complet --------------------------------------------------------
def test_selecteur_boucle():
    s = SelectorScreen()
    assert np.array_equal(s.render(0.0), s.render(1.0))


def test_la_molette_ne_fait_pas_avancer_le_temps():
    """Regle du projet : l'entree joueur ne touche jamais a la boucle."""
    s = SelectorScreen()
    s.wheel.scroll(3)
    for _ in range(4):
        s.step()
    assert np.array_equal(s.render(0.0), s.render(1.0))


def test_le_rendu_ne_modifie_pas_l_etat():
    """Un rendu a effet de bord casserait le bouclage : deux appels a la
    meme phase doivent rendre exactement la meme image."""
    s = SelectorScreen()
    s.wheel.scroll(3)
    s.step()
    pos, vel = s.wheel.pos, s.wheel.vel
    a = s.render(0.25)
    assert (s.wheel.pos, s.wheel.vel) == (pos, vel)
    assert np.array_equal(a, s.render(0.25))


def test_paillettes_seulement_en_mouvement():
    s = SelectorScreen()
    buf = new_screen(320, 240)
    immobile = s._sparkles(buf.copy(), 0.0, 0.0)
    bouge = s._sparkles(buf.copy(), 0.0, 1.0)
    assert np.array_equal(immobile, buf)
    assert bouge.sum() > buf.sum()


def test_paillettes_deterministes():
    """Graine fixe : sinon les paillettes gresilleraient d'une frame a l'autre."""
    s = SelectorScreen()
    buf = new_screen(320, 240)
    assert np.array_equal(s._sparkles(buf.copy(), 0.3, 1.0),
                          s._sparkles(buf.copy(), 0.3, 1.0))


# --- choix du talent et de la nature --------------------------------------
def test_les_13_natures_sont_chargees():
    """Une liste ecrite en dur en avait oublie 5 ; elles viennent du JSON."""
    s = SelectorScreen()
    assert len(s.natures) == 13


def test_toutes_les_natures_sont_atteignables():
    """Defaut corrige : la liste etait decoupee a [:4], donc 9 natures
    etaient impossibles a selectionner."""
    s = SelectorScreen()
    vues = set()
    for _ in range(40):
        vues.add(s.selection()["nature"])
        s.move_nature(1)
    assert len(vues) == 13
    assert s.nature_idx == 12          # borne haute, pas de bouclage sauvage
    for _ in range(40):
        s.move_nature(-1)
    assert s.nature_idx == 0


def test_fenetre_de_liste_garde_le_curseur_visible():
    cl = ChoiceList()
    for i in range(13):
        top = cl.window(13, i, 4)
        assert 0 <= top <= 13 - 4
        assert top <= i < top + 4      # le curseur est DANS la fenetre


def test_liste_courte_ne_defile_pas():
    assert ChoiceList.window(2, 1, 4) == 0


def test_le_talent_suit_l_espece():
    """Zorua n'a qu'un talent : le curseur ne doit pas rester a 1."""
    s = SelectorScreen()
    multi = next(i for i, m in enumerate(s.roster) if len(m["talents"]) > 1)
    solo = next(i for i, m in enumerate(s.roster) if len(m["talents"]) == 1)
    s.wheel.pos = float(multi)
    s.move_talent(1)
    assert s.talent_idx == 1
    s.wheel.pos = float(solo)
    s.step()
    assert s.talent_idx == 0
    assert s.selection()["talent"] == s.roster[solo]["talents"][0]


def test_selection_complete():
    s = SelectorScreen()
    sel = s.selection()
    for k in ("dex", "slug", "nom", "type", "talent", "nature", "nature_rgb"):
        assert k in sel
    assert sel["talent"] in s.roster[s.wheel.index]["talents"]


def test_choisir_ne_fait_pas_avancer_le_temps():
    """Regle du projet : l'entree joueur ne touche jamais a la boucle."""
    s = SelectorScreen()
    s.move_nature(7)
    s.move_talent(1)
    assert np.array_equal(s.render(0.0), s.render(1.0))


def test_roster_sans_orphelin(R):
    """Le dossier de sprites doit correspondre EXACTEMENT au roster."""
    import os
    dirs = {d for d in os.listdir("personality_test/sprites")
            if not d.startswith(".")}
    assert dirs == {m["dex"] for m in R["roster"]}


def test_la_fonte_couvre_tout_le_texte_affiche(R):
    """Le jeu est en francais : sans accents, « Naif » s'affichait « NA F ».
    Aucun libelle du jeu ne doit tomber sur un glyphe manquant."""
    import json
    from soulhalo.dxui import _GLYPHS
    txt = set()
    for e in R["roster"]:
        txt |= set(e["nom"].upper()) | set("".join(e["talents"]).upper())
    for t in R["types"].values():
        txt |= set(t["fr"].upper())
    with open("personality_test/natures.json", encoding="utf-8") as f:
        for n in json.load(f)["natures"]:
            txt |= set(n["fr"].upper())
    manquants = sorted(c for c in txt if c not in _GLYPHS)
    assert manquants == [], f"glyphes manquants : {manquants}"


def test_glyphes_bien_formes():
    """Chaque glyphe doit tenir exactement dans la grille 5x7."""
    from soulhalo.dxui import GLYPH_H, GLYPH_W, _GLYPHS
    for ch, g in _GLYPHS.items():
        assert len(g) == GLYPH_H, ch
        assert all(len(r) == GLYPH_W and set(r) <= {"0", "1"} for r in g), ch
