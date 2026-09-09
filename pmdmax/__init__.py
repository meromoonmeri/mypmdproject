"""
pmdmax — génération de sprites Dynamax au format strict PMDCollab/SpriteCollab.

Modules :
  config       constantes du format (SpriteBot) + palette Dynamax
  pixels       primitives de dessin pixel-art (palette indexée, alpha binaire)
  graphicscale agrandissement EPX/Scale2x/3x sans création de couleur
  sheet        lecture/écriture des triplets Anim/Offsets/Shadow + AnimData.xml
  clouds       nuages Dynamax tournoyants
  beam         colonne d'énergie + éclairs spiralés (FX sur fond magenta)
  dynamax      pipeline : sprite normal -> sprite Dynamax
  transform    animation de transformation multi-frames
  verify       validateur strict (mêmes règles que le bot de soumission)
  aseprite     export .ase multi-calques pour retouche manuelle
"""

__version__ = "1.0.0"
