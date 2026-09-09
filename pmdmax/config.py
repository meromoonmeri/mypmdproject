"""
Constantes du format SpriteCollab / SpriteBot + palette Dynamax.

Toutes les valeurs "strictes" viennent de :
  - PMDCollab/SpriteBot : Constants.py, SpriteUtils.py, utils.py
  - PMDCollab/SpriteCollab : sprite_config.json

Ne rien modifier ici sans vérifier la source amont : le vérificateur
(pmdmax/verify.py) applique exactement les mêmes règles que le bot
de soumission.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# Constantes issues de SpriteBot/SpriteUtils.py
# --------------------------------------------------------------------------

#: Décalage du "centre de dessin" d'une tuile (SpriteUtils.DRAW_CENTER_X/Y).
#: rel_center = (w // 2 - DRAW_CENTER_X, h // 2 - DRAW_CENTER_Y)
DRAW_CENTER_X = 0
DRAW_CENTER_Y = -4

#: Taille max d'un fichier dans le zip de soumission (SpriteUtils.ZIP_SIZE_LIMIT)
ZIP_SIZE_LIMIT = 5_000_000

#: Nom du descripteur d'animations (Constants.MULTI_SHEET_XML)
MULTI_SHEET_XML = "AnimData.xml"

#: Les 8 directions, dans l'ordre des lignes de la spritesheet (Constants.DIRECTIONS)
DIRECTIONS = [
    "Down", "DownRight", "Right", "UpRight",
    "Up", "UpLeft", "Left", "DownLeft",
]

#: Nombre max de couleurs non transparentes avant que le bot ne réclame `--colors`
PALETTE_SOFT_LIMIT = 15

#: Les tuiles doivent être des multiples de 8 (utils.roundUpToMult(x, 8))
TILE_MULTIPLE = 8

# --- Couleurs réservées des feuilles techniques ---------------------------
# -Offsets.png : un pixel de chaque, opaque, par tuile.
OFF_HEAD = (0, 0, 0, 255)        # noir   : tête
OFF_LHAND = (255, 0, 0, 255)     # rouge  : main gauche
OFF_CENTER = (0, 255, 0, 255)    # vert   : centre du corps
OFF_RHAND = (0, 0, 255, 255)     # bleu   : main droite

#: BLANC dans -Offsets.png = raccourci officiel signifiant « les quatre
#: ancres sont au même pixel » (cf. SpriteBot/utils.py getOffsetFromRGB :
#: un pixel blanc renseigne d'un coup results[0..3]). Le noir peut coexister
#: avec le blanc et prend alors le pas pour la tête.
OFF_ALL = (255, 255, 255, 255)

# -Shadow.png : blobs d'ombre + un unique pixel blanc = point d'ancrage au sol.
SDW_CENTER = (255, 255, 255, 255)  # blanc : centre de l'ombre (1 seul par tuile)
SDW_SMALL = (0, 255, 0, 255)       # vert  : visible quel que soit ShadowSize
SDW_MED = (255, 0, 0, 255)         # rouge : visible si ShadowSize >= 1
SDW_LARGE = (0, 0, 255, 255)       # bleu  : visible si ShadowSize >= 2

TRANSPARENT = (0, 0, 0, 0)


# --------------------------------------------------------------------------
# sprite_config.json (liste officielle des actions)
# --------------------------------------------------------------------------

def _load_sprite_config() -> dict:
    """Charge sprite_config.json depuis le clone amont, sinon la copie locale."""
    candidates = [
        os.path.join(REPO, "external", "SpriteCollab", "sprite_config.json"),
        os.path.join(REPO, "data", "sprite_config.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(
        "sprite_config.json introuvable. Lance tools/fetch_upstream.sh "
        "ou restaure data/sprite_config.json."
    )


_CFG = _load_sprite_config()

#: Noms d'action valides. Un <Name> hors de cette liste = refus du bot.
ACTIONS: List[str] = _CFG["actions"]

#: Index WAN imposés pour certaines animations (clé = index, valeur = nom).
ACTION_MAP: Dict[int, str] = {int(k): v for k, v in _CFG["action_map"].items()}

#: Paliers de complétion : [0] = minimum requis, [2] = "fully featured".
COMPLETION_ACTIONS: List[List[int]] = _CFG["completion_actions"]

PORTRAIT_SIZE = _CFG["portrait_size"]
PORTRAIT_TILE_X = _CFG["portrait_tile_x"]
PORTRAIT_TILE_Y = _CFG["portrait_tile_y"]
EMOTIONS: List[str] = _CFG["emotions"]

#: Index WAN -> nom, pour les animations obligatoires.
REQUIRED_ACTION_INDEXES = dict(ACTION_MAP)


def free_action_index(used: set) -> int:
    """Renvoie le plus petit index WAN libre qui ne viole pas ACTION_MAP."""
    idx = 0
    while True:
        if idx not in used and idx not in ACTION_MAP:
            return idx
        idx += 1


# --------------------------------------------------------------------------
# Palette Dynamax
# --------------------------------------------------------------------------
# Extraite par quantification des planches de référence générées par le
# générateur d'images (refs/dynamax_clouds_ref.png, refs/dynamax_beam_ref.png)
# puis calée sur une rampe 6 tons lisible en 16 bits.
#
# 6 couleurs seulement : un sprite PMD typique tient en 7-12 couleurs, on reste
# donc sous la barre des 15 couleurs du bot pour la plupart des Pokémon.

DYNA_OUTLINE = (43, 0, 16, 255)     # contour / ombre portée des nuages
DYNA_DARK = (112, 5, 38, 255)       # face inférieure des nuages
DYNA_MID = (186, 21, 62, 255)       # corps du nuage (cramoisi)
DYNA_BRIGHT = (233, 45, 96, 255)    # face éclairée
DYNA_RIM = (252, 110, 160, 255)     # liseré supérieur / aura
DYNA_CORE = (255, 208, 226, 255)    # cœur des éclairs et de la colonne

#: Rampe ordonnée du plus sombre au plus clair. Les modules de rendu
#: n'utilisent QUE ces indices (1..6), 0 = transparent.
DYNA_RAMP = [
    TRANSPARENT,
    DYNA_OUTLINE,
    DYNA_DARK,
    DYNA_MID,
    DYNA_BRIGHT,
    DYNA_RIM,
    DYNA_CORE,
]

IDX_EMPTY = 0
IDX_OUTLINE = 1
IDX_DARK = 2
IDX_MID = 3
IDX_BRIGHT = 4
IDX_RIM = 5
IDX_CORE = 6
