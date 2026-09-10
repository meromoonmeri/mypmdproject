"""
UI façon PMD DX, calée sur les paramètres réels du moteur PMDO.

CONSTANTES DU MOTEUR (relevées dans RogueCollab/RogueEssence)
--------------------------------------------------------------
    MenuBase.VERT_SPACE        = 14   espacement vertical d'une ligne
    MenuBase.LINE_HEIGHT       = 12   hauteur d'une ligne de texte
    TitledStripMenu.TITLE_OFFSET = 16 décalage sous le titre
    GraphicsManager.PortraitSize = 40 portraits carrés 40x40

L'écran de référence de PMDO est 320x240. Tout est donc dessiné dans cette
grille logique puis agrandi d'un facteur ENTIER, au plus proche voisin :
c'est ce que fait le moteur, et c'est la seule façon de ne pas baver le
pixel art.

CE QUE FOURNIT CE MODULE
------------------------
`MenuFrame`   cadre de menu DX (bordure, fond translucide, titre) ;
`PokemonCard` la vignette : SPRITE encadré, PORTRAIT au-dessus ;
`ChoiceList`  liste de choix (talent / nature) avec curseur ;
`WheelState`  défilement à la molette, avec inertie et paillettes.

Rien ici n'anime le temps : ces éléments se dessinent par-dessus un fond
qui, lui, continue de boucler.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import field as F
from . import pixelart as PX

TAU = F.TAU
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- constantes reprises telles quelles du moteur -------------------------
VERT_SPACE = 14
LINE_HEIGHT = 12
TITLE_OFFSET = 16
PORTRAIT_SIZE = 40
SCREEN_W, SCREEN_H = 320, 240      # écran logique de référence PMDO


def _p(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def load_roster(path="personality_test/types.json"):
    return json.load(open(_p(path), encoding="utf-8"))


# --------------------------------------------------------------------------
# dessin de base, en pixels logiques
# --------------------------------------------------------------------------
def _blend(dst, src, alpha, x, y):
    """Compose `src` (h,w,3) avec `alpha` (h,w,1) à la position (x,y)."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    s = src[y1 - y:y2 - y, x1 - x:x2 - x]
    a = alpha[y1 - y:y2 - y, x1 - x:x2 - x]
    dst[y1:y2, x1:x2] = dst[y1:y2, x1:x2] * (1.0 - a) + s * a


def _round_mask(h, w, r):
    """Masque plein à coins arrondis. DX n'a aucun angle droit."""
    m = np.ones((h, w), np.float32)
    r = int(min(r, h // 2, w // 2))
    if r <= 0:
        return m
    yy, xx = np.indices((r, r))
    c = (np.hypot(r - 1 - yy, r - 1 - xx) <= r - 0.35).astype(np.float32)
    m[:r, :r] = c
    m[:r, w - r:] = c[:, ::-1]
    m[h - r:, :r] = c[::-1, :]
    m[h - r:, w - r:] = c[::-1, ::-1]
    return m


def _erode(m):
    """Érosion 4-voisins : sert à extraire les liserés du masque."""
    e = m.copy()
    e[1:] = np.minimum(e[1:], m[:-1])
    e[:-1] = np.minimum(e[:-1], m[1:])
    e[:, 1:] = np.minimum(e[:, 1:], m[:, :-1])
    e[:, :-1] = np.minimum(e[:, :-1], m[:, 1:])
    return e


@dataclass(frozen=True)
class MenuStyle:
    """Palette RELEVEE sur des captures de Rescue Team DX.

    Les valeurs sont des medianes mesurees sur le panneau « Overview »,
    pas des couleurs choisies a l'oeil : le parchemin de DX est CHAUD
    (0.58, 0.45, 0.23) et son second panneau turquoise (0.40, 0.55, 0.58).
    La palette bleu nuit precedente etait celle d'Explorers of Sky - donc
    du PMD classique, pas du DX.
    """
    # parchemin
    fill: tuple = (0.584, 0.449, 0.227)
    fill_hi: tuple = (0.714, 0.588, 0.361)
    fill_lo: tuple = (0.443, 0.318, 0.153)
    fill_alpha: float = 0.97
    # bois du cadre
    border_out: tuple = (0.235, 0.129, 0.055)     # trait exterieur sombre
    border_in: tuple = (0.804, 0.671, 0.443)      # bevel clair interieur
    wood: tuple = (0.604, 0.435, 0.286)
    # bandeau sombre des libelles
    band: tuple = (0.486, 0.298, 0.149)
    # panneau secondaire turquoise
    teal: tuple = (0.396, 0.545, 0.584)
    teal_lo: tuple = (0.259, 0.388, 0.427)
    # textes
    title: tuple = (1.000, 0.984, 0.945)
    text: tuple = (0.204, 0.110, 0.047)
    text_on_band: tuple = (0.969, 0.929, 0.847)
    cursor: tuple = (1.000, 0.843, 0.322)
    dim: tuple = (0.435, 0.318, 0.176)
    # Cerne du texte courant. Distinct de `border_in` : ce dernier est le
    # bevel du cadre, et il se trouve etre clair. Sur un style a TEXTE
    # CLAIR (PMDO), cerner du clair avec du clair rend le texte illisible.
    # Le cerne doit toujours contraster avec la couleur du texte.
    text_outline: tuple = (0.804, 0.671, 0.443)
    # Texte de la ligne SELECTIONNEE, posee sur le bandeau de curseur jaune
    # vif. Il lui faut sa propre couleur : reprendre `text` marchait en
    # style DX (texte sombre) mais donnait du clair sur jaune en PMDO.
    text_on_cursor: tuple = (0.204, 0.110, 0.047)
    # texture
    hatch: float = 0.06
    radius: int = 3
    # relief 2.5D
    bevel: float = 0.22          # arête éclairée en haut-gauche
    shade: float = 0.20          # arête d'ombre en bas-droite
    shadow: float = 0.42         # ombre portée
    shadow_dx: int = 2
    shadow_dy: int = 2
    dither_levels: int = 16      # paliers par canal (0 = pas de tramage)


def pmdo_style():
    """Style de fenêtre PAR DÉFAUT de PMDO / RogueEssence.

    Le moteur ne peint pas le parchemin de DX : sa fenêtre standard est un
    bleu nuit dense, cerné d'un liseré clair, sans hachure ni relief
    marqué. C'est la fenêtre que voit un joueur qui lance un mod PMDO sans
    y toucher — donc celle du test de personnalité.

    On ne touche PAS au moteur de rendu : `MenuFrame` reste le même, seul
    le jeu de couleurs change. Le style DX reste disponible pour le reste
    du jeu (badge, révélation).
    """
    return MenuStyle(
        fill=(0.129, 0.161, 0.353),
        fill_hi=(0.169, 0.204, 0.427),
        fill_lo=(0.086, 0.106, 0.247),
        fill_alpha=0.96,
        border_out=(0.043, 0.055, 0.129),     # trait extérieur sombre
        border_in=(0.898, 0.918, 0.976),      # liseré clair de RogueEssence
        wood=(0.196, 0.239, 0.478),
        band=(0.106, 0.133, 0.302),
        teal=(0.216, 0.325, 0.494),
        teal_lo=(0.129, 0.208, 0.345),
        title=(1.000, 0.984, 0.945),
        text=(0.965, 0.973, 0.996),           # texte CLAIR sur fond sombre
        text_on_band=(0.898, 0.918, 0.976),
        cursor=(1.000, 0.843, 0.322),
        dim=(0.616, 0.655, 0.784),
        text_outline=(0.043, 0.055, 0.129),   # cerne SOMBRE sous texte clair
        text_on_cursor=(0.043, 0.055, 0.129),  # sombre sur le curseur jaune
        hatch=0.0,                            # PMDO n'a pas de hachures
        radius=2,                             # coins plus francs que DX
        bevel=0.10,
        shade=0.10,
        shadow=0.34,
        dither_levels=16,
    )


class MenuFrame:
    """Panneau DX transpose en pixel art, sur la grille du moteur.

    DX peint ses panneaux en haute resolution ; PMDO travaille en 320x240.
    On garde donc la GRILLE du moteur et on transpose la direction
    artistique : parchemin chaud degrade, hachures diagonales, liseré bois
    sombre double d'un bevel clair, coins arrondis.

    Difference structurelle avec la version precedente : dans DX le titre
    n'est PAS un separateur trace a l'interieur du cadre, c'est une PLAQUE
    bombee posee par-dessus le bord haut, qui deborde vers le haut.
    """

    def __init__(self, style: MenuStyle | None = None):
        self.s = style or MenuStyle()

    # -- corps -------------------------------------------------------------
    def panel(self, buf, x, y, w, h, color=None, lo=None, radius=None,
              alpha=None, hatch=True):
        """Pose un panneau : degrade, hachures, bevel clair, contour sombre."""
        s = self.s
        H, W = buf.shape[:2]
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 2 or h <= 2:
            return
        r = s.radius if radius is None else int(radius)
        a = s.fill_alpha if alpha is None else float(alpha)
        top = np.array(s.fill_hi if color is None else color, np.float32)
        bot = np.array(s.fill_lo if lo is None else lo, np.float32)

        m = _round_mask(h, w, r)
        # degrade vertical : DX eclaire ses panneaux par le haut
        g = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        col = top[None, None, :] * (1.0 - g) + bot[None, None, :] * g
        col = np.repeat(col, w, axis=1)

        if hatch and s.hatch > 0:
            yy, xx = np.indices((h, w))
            # hachures diagonales, plus marquees vers le bas-droite
            ramp = ((yy / max(1, h - 1)) * 0.6 + (xx / max(1, w - 1)) * 0.4)
            lines = (((xx + yy) % 4) == 0).astype(np.float32)
            col = col * (1.0 - s.hatch * lines * ramp)[..., None]

        inner = _erode(m)
        bevel = np.clip(m - inner, 0.0, 1.0)          # 1 px de contour
        core = _erode(inner)
        light = np.clip(inner - core, 0.0, 1.0)       # 1 px juste dedans
        col = col * (1.0 - bevel[..., None]) + \
            np.array(s.border_out, np.float32) * bevel[..., None]
        col = col * (1.0 - light[..., None] * 0.55) + \
            np.array(s.border_in, np.float32) * (light[..., None] * 0.55)

        # relief 2.5D : le panneau est ECLAIRE en haut-gauche et ombre en
        # bas-droite. C'est ce qui le fait paraitre pose SUR l'ecran plutot
        # que peint dedans.
        if s.bevel > 0 or s.shade > 0:
            col = PX.emboss(col, inner, light=s.bevel, shade=s.shade)
        # tramage : les degrades doivent etre trames AVANT l'agrandissement,
        # sinon on obtient des aplats en bandes.
        if s.dither_levels:
            col = PX.dither(col, levels=s.dither_levels, strength=0.85)

        # decoupe a l'ecran
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        # ombre portee : posee sur le FOND, hors de la forme
        if s.shadow > 0:
            sm = np.zeros((H, W), np.float32)
            sm[y1:y2, x1:x2] = m[y1 - y:y2 - y, x1 - x:x2 - x]
            buf[:] = PX.drop_shadow(buf, sm, dx=s.shadow_dx, dy=s.shadow_dy,
                                    opacity=s.shadow)
        sub = col[y1 - y:y2 - y, x1 - x:x2 - x]
        sa = (m[y1 - y:y2 - y, x1 - x:x2 - x] * a)[..., None]
        buf[y1:y2, x1:x2] = buf[y1:y2, x1:x2] * (1.0 - sa) + sub * sa

    # -- plaque de titre ---------------------------------------------------
    def title_plate(self, buf, x, y, w, text):
        """Plaque bombee posee SUR le bord haut, comme dans DX."""
        s = self.s
        tw = text_width(text)
        pw = int(min(max(tw + 18, 52), max(20, w - 6)))
        px = int(x + (w - pw) // 2)
        ph = 13
        py = int(y - 5)
        self.panel(buf, px, py, pw, ph, color=s.wood,
                   lo=(0.404, 0.259, 0.137), radius=4, hatch=False)
        draw_text(buf, text, px + (pw - tw) // 2, py + 3, s.title,
                  outline=s.border_out)
        return py + ph

    def draw(self, buf, x, y, w, h, title=None, divider=True, color=None,
             lo=None, radius=None, alpha=None):
        """Panneau complet. `divider` n'a plus d'objet : DX pose une plaque."""
        self.panel(buf, x, y, w, h, color=color, lo=lo, radius=radius,
                   alpha=alpha)
        if title is not None:
            self.title_plate(buf, x, y, w, str(title).upper())


# --------------------------------------------------------------------------
# sprites
# --------------------------------------------------------------------------
class SpriteBank:
    """Charge sprites et portraits SpriteCollab, avec leurs vraies durées."""

    def __init__(self, root="personality_test/sprites"):
        self.root = _p(root)
        self._sheet = {}
        self._dur = {}
        self._por = {}

    def durations(self, dex):
        if dex in self._dur:
            return self._dur[dex]
        xml = os.path.join(self.root, dex, "AnimData.xml")
        out = []
        if os.path.exists(xml):
            root = ET.parse(xml).getroot()
            node = next((a for a in root.iter("Anim")
                         if a.findtext("Name") == "Idle"), None)
            if node is not None and node.findtext("FrameWidth"):
                out = [int(d.text) for d in node.iter("Duration")]
                self._sheet[dex] = (int(node.findtext("FrameWidth")),
                                    int(node.findtext("FrameHeight")))
        self._dur[dex] = out
        return out

    def frame_index(self, dex, tick):
        """Index de frame d'après les DURÉES réelles, pas une cadence fixe."""
        dur = self.durations(dex)
        if not dur:
            return 0
        t = int(tick) % sum(dur)
        acc = 0
        for i, d in enumerate(dur):
            acc += d
            if t < acc:
                return i
        return 0

    def sprite(self, dex, tick=0, direction=0):
        self.durations(dex)
        if dex not in self._sheet:
            return None
        fw, fh = self._sheet[dex]
        path = os.path.join(self.root, dex, "Idle-Anim.png")
        if not os.path.exists(path):
            return None
        im = Image.open(path).convert("RGBA")
        i = self.frame_index(dex, tick)
        ncol = max(1, im.width // fw)
        i %= ncol
        row = min(direction, max(0, im.height // fh - 1))
        return im.crop((i * fw, row * fh, i * fw + fw, row * fh + fh))

    def portrait(self, dex):
        """Portrait 40x40. SpriteCollab les livre sur fond OPAQUE : on le
        garde tel quel et on l'ENCADRE, comme DX, plutôt que de détourer un
        fond qui fait partie de l'illustration."""
        if dex in self._por:
            return self._por[dex]
        path = os.path.join(self.root, dex, "portrait.png")
        im = Image.open(path).convert("RGBA") if os.path.exists(path) else None
        self._por[dex] = im
        return im


# --------------------------------------------------------------------------
# la vignette : sprite encadré + portrait au-dessus
# --------------------------------------------------------------------------
@dataclass
class CardParams:
    # Taille MESUREE sur le jeu de sprites reel : le plus grand fait 40x56
    # (Pikachu). Une carte de 64 px de haut le laissait deborder sous le
    # cadre. Hauteur = gap + portrait(40) + gap + liseré + 56.
    w: int = 48
    h: int = 102
    portrait_scale: int = 1     # le portrait est déjà à 40 px
    sprite_scale: int = 1
    gap: int = 2


class PokemonCard:
    """Vignette DX : le PORTRAIT au-dessus, le SPRITE encadré en dessous."""

    def __init__(self, bank: SpriteBank, frame: MenuFrame | None = None,
                 params: CardParams | None = None):
        self.bank = bank
        self.frame = frame or MenuFrame()
        self.p = params or CardParams()

    def draw(self, buf, dex, x, y, tick=0, selected=False, hover=0.0,
             tint=None):
        p = self.p
        s = self.frame.s
        # la carte selectionnee passe sur bois clair, comme un onglet actif
        self.frame.draw(buf, x, y, p.w, p.h,
                        color=s.wood if selected else None,
                        lo=s.fill_lo if selected else None)

        # portrait, encadré : SpriteCollab le livre sur fond opaque, donc on
        # l'assume comme une vignette, exactement comme DX.
        por = self.bank.portrait(dex)
        if por is not None:
            a = np.asarray(por, np.float32) / 255.0
            px = x + (p.w - por.width) // 2
            py = y + p.gap + 1
            _blend(buf, a[..., :3], a[..., 3:], px, py)
            # liseré du portrait : bois sombre, dore si selectionne
            c = np.array(s.cursor if selected else s.border_out, np.float32)
            H, W = buf.shape[:2]
            if 0 <= py - 1 < H:
                buf[py - 1, max(0, px - 1):min(W, px + por.width + 1)] = c
            if 0 <= py + por.height < H:
                buf[py + por.height, max(0, px - 1):min(W, px + por.width + 1)] = c
            if 0 <= px - 1 < W:
                buf[max(0, py - 1):min(H, py + por.height + 1), px - 1] = c
            if 0 <= px + por.width < W:
                buf[max(0, py - 1):min(H, py + por.height + 1), px + por.width] = c

        # sprite, sous le portrait
        sp = self.bank.sprite(dex, tick)
        if sp is not None:
            a = np.asarray(sp, np.float32) / 255.0
            rgb, al = a[..., :3], a[..., 3:]
            if tint is not None and hover > 0:
                rgb = rgb * (1 - 0.35 * hover) + \
                    np.asarray(tint, np.float32) * (0.35 * hover)
            sx = x + (p.w - sp.width) // 2
            sy = y + p.gap + PORTRAIT_SIZE + p.gap + 2
            _blend(buf, rgb, al, sx, sy)


# --------------------------------------------------------------------------
# liste de choix : talent, nature
# --------------------------------------------------------------------------
class ChoiceList:
    """Liste de choix DX, espacée de VERT_SPACE comme dans le moteur."""

    def __init__(self, frame: MenuFrame | None = None, font=None):
        self.frame = frame or MenuFrame()
        self.font = font

    def height(self, n, titled=True):
        """Hauteur exacte, formule de BaseSettingsMenu du moteur :
        n * VERT_SPACE + bordures + ContentOffset."""
        return n * VERT_SPACE + 4 + (TITLE_OFFSET if titled else 0)

    @staticmethod
    def window(n, index, max_rows):
        """Premier indice visible pour que `index` reste dans le cadre.

        Sans ça, une liste plus longue que la fenêtre rend ses derniers
        éléments INATTEIGNABLES : c'était le cas des 13 natures, dont 9
        étaient tronquées par un simple découpage `[:4]`.
        """
        if max_rows is None or n <= max_rows:
            return 0
        # on garde le curseur au centre tant que les bords le permettent
        top = index - max_rows // 2
        return int(np.clip(top, 0, n - max_rows))

    def draw(self, buf, x, y, w, items, index=0, title=None, tick=0,
             max_rows=None):
        n = len(items)
        top = self.window(n, index, max_rows)
        shown = items[top:top + max_rows] if max_rows else list(items)
        h = self.height(len(shown), title is not None)
        self.frame.draw(buf, x, y, w, h, title=title)
        s = self.frame.s
        y0 = y + 2 + (TITLE_OFFSET if title else 2)
        H, W = buf.shape[:2]
        for i, it in enumerate(shown):
            yy = y0 + i * VERT_SPACE
            sel = (top + i == index)
            if sel:
                # bandeau de curseur : dans DX la ligne active est posee sur
                # une plaque, pas seulement coloree. Periode ENTIERE (30
                # ticks) pour que le clignotement ne derive pas.
                # Le plancher reste haut (0.88) : a 0.72 la plaque doree
                # tombait au niveau du parchemin a mi-clignotement et la
                # ligne active devenait indistinguable des autres.
                blink = 0.94 + 0.06 * float(np.cos(TAU * (tick % 30) / 30.0))
                self.frame.panel(buf, x + 3, yy - 2, w - 6, LINE_HEIGHT,
                                 color=tuple(c * blink for c in s.cursor),
                                 lo=tuple(c * blink * 0.72 for c in s.cursor),
                                 radius=2, hatch=False)
            else:
                # les lignes inactives reposent sur le bandeau sombre
                x1, x2 = max(0, x + 3), min(W, x + w - 3)
                y1, y2 = max(0, yy - 2), min(H, yy + LINE_HEIGHT - 2)
                if x2 > x1 and y2 > y1:
                    b = np.array(s.band, np.float32)
                    buf[y1:y2, x1:x2] = buf[y1:y2, x1:x2] * 0.62 + b * 0.38
            lbl = str(it).replace("_", " ")
            draw_text(buf, lbl, x + 7, yy + 1,
                      s.text_on_cursor if sel else s.text_on_band,
                      outline=None if sel else s.border_out)
        # chevrons : signaler qu'il reste des entrées hors du cadre
        if max_rows and n > max_rows:
            if top > 0:
                draw_text(buf, "-", x + w - 11, y + 3, s.title,
                          outline=s.border_out)
            if top + max_rows < n:
                draw_text(buf, "-", x + w - 11, y + h - 10, s.title,
                          outline=s.border_out)
        return h


# --------------------------------------------------------------------------
# molette
# --------------------------------------------------------------------------
@dataclass
class WheelState:
    """Défilement à la molette, avec inertie.

    `pos` est continu : le carrousel peut s'arrêter entre deux crans. La
    vitesse décroît géométriquement, ce qui donne le glissé de DX au lieu
    d'un saut sec.
    """
    pos: float = 0.0
    vel: float = 0.0
    n: int = 1
    friction: float = 0.82
    step: float = 0.55          # crans par coup de molette
    max_vel: float = 1.8

    def scroll(self, notches: float):
        self.vel = float(np.clip(self.vel + notches * self.step,
                                 -self.max_vel, self.max_vel))
        return self

    def update(self):
        self.pos += self.vel
        self.vel *= self.friction
        if abs(self.vel) < 1e-3:
            self.vel = 0.0
        self.pos = float(np.clip(self.pos, 0.0, max(0, self.n - 1)))
        return self

    @property
    def index(self):
        return int(round(self.pos))

    @property
    def moving(self):
        return abs(self.vel) > 1e-3


# --------------------------------------------------------------------------
# agrandissement entier
# --------------------------------------------------------------------------
def upscale(buf, factor: int):
    """Agrandit d'un facteur ENTIER au plus proche voisin.

    Le moteur travaille en 320x240 puis agrandit ainsi. Un facteur
    fractionnaire ou une interpolation lisse rendrait le pixel art flou.
    """
    a = np.clip(buf, 0.0, 1.0)
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


_GLYPHS = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "11110", "10001", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "11110", "10000", "10000", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01111"],
    "H": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "11100", "10100", "10010", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "01110", "00001", "00001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
    "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    # accents francais : sans eux « Naïf » et « Presse » s'affichaient
    # troues (« NA F »). La grille 5x7 n'a pas de place au-dessus des
    # capitales, donc l'accent mord sur la premiere ligne du glyphe.
    "É": ["00010", "11111", "10000", "11110", "10000", "10000", "11111"],
    "È": ["01000", "11111", "10000", "11110", "10000", "10000", "11111"],
    "Ê": ["00100", "11111", "10000", "11110", "10000", "10000", "11111"],
    "Ï": ["01010", "11111", "00100", "00100", "00100", "00100", "11111"],
    "Î": ["00100", "11111", "00100", "00100", "00100", "00100", "11111"],
    "À": ["01000", "01110", "10001", "11111", "10001", "10001", "10001"],
    "Â": ["00100", "01110", "10001", "11111", "10001", "10001", "10001"],
    "Ô": ["00100", "01110", "10001", "10001", "10001", "10001", "01110"],
    "Û": ["00100", "10001", "10001", "10001", "10001", "10001", "01110"],
    "Ù": ["01000", "10001", "10001", "10001", "10001", "10001", "01110"],
    "Ç": ["01111", "10000", "10000", "10000", "10000", "01111", "00100"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "'": ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}
GLYPH_W, GLYPH_H = 5, 7


def draw_text(buf, text, x, y, color=(1.0, 1.0, 1.0), spacing=1,
              outline=None):
    """Ecrit du texte en police bitmap 5x7.

    Les menus PMDO reposent sur une fonte pixel : une fonte vectorielle
    agrandie casserait l'alignement sur la grille logique.

    `outline` cerne les glyphes d'un liseré 1 px. DX cerne tous ses textes
    de sombre : sans ce contour, du texte clair sur parchemin clair devient
    illisible.
    """
    H, W = buf.shape[:2]
    col = np.asarray(color, np.float32)
    x, y = int(x), int(y)

    # on collecte d'abord les pixels, pour poser le contour dessous
    pix = []
    cx = x
    for ch in str(text).upper():
        g = _GLYPHS.get(ch)
        if g is not None:
            for ry, row in enumerate(g):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        pix.append((cx + rx, y + ry))
        cx += GLYPH_W + spacing

    if outline is not None and pix:
        oc = np.asarray(outline, np.float32)
        on = set()
        full = set(pix)
        for px, py in pix:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    p = (px + dx, py + dy)
                    if p not in full:
                        on.add(p)
        for px, py in on:
            if 0 <= px < W and 0 <= py < H:
                buf[py, px] = oc

    for px, py in pix:
        if 0 <= px < W and 0 <= py < H:
            buf[py, px] = col
    return cx - x


def text_width(text, spacing=1):
    return max(0, len(str(text)) * (GLYPH_W + spacing) - spacing)


def new_screen(w=SCREEN_W, h=SCREEN_H):
    return np.zeros((h, w, 3), np.float32)
