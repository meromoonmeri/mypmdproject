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


@dataclass(frozen=True)
class MenuStyle:
    """Palette du cadre, dans l'esprit des menus PMD (bleu nuit + liseré)."""
    fill: tuple = (0.07, 0.10, 0.28)
    fill_alpha: float = 0.86
    border_out: tuple = (0.98, 0.98, 1.00)
    border_in: tuple = (0.38, 0.52, 0.92)
    title: tuple = (1.00, 0.94, 0.62)
    text: tuple = (0.96, 0.97, 1.00)
    cursor: tuple = (1.00, 0.86, 0.30)
    dim: tuple = (0.62, 0.66, 0.78)


class MenuFrame:
    """Cadre de menu DX : fond translucide, double liseré, séparateur.

    La bordure fait 1 px logique, comme dans le moteur : à l'échelle 3x ou
    4x elle reste nette parce que l'agrandissement est entier.
    """

    def __init__(self, style: MenuStyle | None = None):
        self.s = style or MenuStyle()

    def draw(self, buf, x, y, w, h, title=None, divider=True):
        s = self.s
        H, W = buf.shape[:2]
        x, y = int(x), int(y)
        w, h = int(w), int(h)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x or y2 <= y:
            return
        # fond
        reg = buf[y:y2, x:x2]
        buf[y:y2, x:x2] = reg * (1 - s.fill_alpha) + \
            np.array(s.fill, np.float32) * s.fill_alpha
        # liseré externe puis interne
        for k, col in ((0, s.border_out), (1, s.border_in)):
            c = np.array(col, np.float32)
            if y + k < y2:
                buf[y + k, x + k:x2 - k] = c
            if y2 - 1 - k > y:
                buf[y2 - 1 - k, x + k:x2 - k] = c
            if x + k < x2:
                buf[y + k:y2 - k, x + k] = c
            if x2 - 1 - k > x:
                buf[y + k:y2 - k, x2 - 1 - k] = c
        # séparateur sous le titre, à LINE_HEIGHT comme TitledStripMenu
        if title is not None and divider:
            yy = y + 2 + LINE_HEIGHT
            if y < yy < y2 - 1:
                buf[yy, x + 3:x2 - 3] = np.array(s.border_in, np.float32)


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
        self.frame.draw(buf, x, y, p.w, p.h, title=None, divider=False)

        # portrait, encadré : SpriteCollab le livre sur fond opaque, donc on
        # l'assume comme une vignette, exactement comme DX.
        por = self.bank.portrait(dex)
        if por is not None:
            a = np.asarray(por, np.float32) / 255.0
            px = x + (p.w - por.width) // 2
            py = y + p.gap + 1
            _blend(buf, a[..., :3], a[..., 3:], px, py)
            # liseré du portrait
            c = np.array(s.border_in if not selected else s.cursor, np.float32)
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

    def draw(self, buf, x, y, w, items, index=0, title=None, tick=0):
        n = len(items)
        h = self.height(n, title is not None)
        self.frame.draw(buf, x, y, w, h, title=title)
        s = self.frame.s
        if title:
            draw_text(buf, title, x + 4, y + 3, s.title)
        y0 = y + 2 + (TITLE_OFFSET if title else 2)
        for i, it in enumerate(items):
            yy = y0 + i * VERT_SPACE
            if i == index:
                # curseur clignotant, période ENTIÈRE : pas de dérive
                blink = 0.65 + 0.35 * float(np.cos(TAU * (tick % 30) / 30.0))
                H, W = buf.shape[:2]
                x1, x2 = max(0, x + 3), min(W, x + w - 3)
                y1, y2 = max(0, yy - 1), min(H, yy + LINE_HEIGHT - 1)
                if x2 > x1 and y2 > y1:
                    c = np.array(s.cursor, np.float32) * blink
                    buf[y1:y2, x1:x2] = buf[y1:y2, x1:x2] * 0.45 + c * 0.55
            lbl = str(it).replace("_", " ")
            draw_text(buf, lbl, x + 6, yy + 1,
                      s.fill if i == index else s.text)
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
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "'": ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "?": ["01110", "10001", "00010", "00100", "00100", "00000", "00100"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}
GLYPH_W, GLYPH_H = 5, 7


def draw_text(buf, text, x, y, color=(1.0, 1.0, 1.0), spacing=1):
    """Ecrit du texte en police bitmap 5x7.

    Les menus PMDO reposent sur une fonte pixel : une fonte vectorielle
    agrandie casserait l'alignement sur la grille logique.
    """
    H, W = buf.shape[:2]
    col = np.asarray(color, np.float32)
    cx = int(x)
    for ch in str(text).upper():
        g = _GLYPHS.get(ch)
        if g is None:
            cx += GLYPH_W + spacing
            continue
        for ry, row in enumerate(g):
            yy = int(y) + ry
            if not (0 <= yy < H):
                continue
            for rx, bit in enumerate(row):
                if bit == "1":
                    xx = cx + rx
                    if 0 <= xx < W:
                        buf[yy, xx] = col
        cx += GLYPH_W + spacing
    return cx - int(x)


def text_width(text, spacing=1):
    return max(0, len(str(text)) * (GLYPH_W + spacing) - spacing)


def new_screen(w=SCREEN_W, h=SCREEN_H):
    return np.zeros((h, w, 3), np.float32)
