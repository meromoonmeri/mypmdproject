# pmdmax — sprites Dynamax au format PMDCollab/SpriteCollab

Générateur de sprites **Dynamax** pour *Pokémon Mystery Dungeon*, au format
**strict** du dépôt [PMDCollab/SpriteCollab](https://github.com/PMDCollab/SpriteCollab).

À partir d'un sprite existant, l'outil produit :

1. **le Pokémon agrandi** avec **les nuages rouges qui tournoient au-dessus
   de sa tête** et **l'aura de fluide rouge ondulant qui épouse sa
   silhouette**, appliqués à **toutes ses animations** (les 8 directions et
   toutes les frames) ;
2. **une animation de transformation en plusieurs frames**, à part : une
   **colonne d'énergie rouge opaque** lui tombe dessus, puis **plusieurs
   éclairs rouges montent en spirale** et condensent les nuages ;
3. **les frames du rayon sur fond magenta** (`#FF00FF`), prêtes à être
   animées/retouchées, comme demandé.

Tout ce qui sort de l'outil passe un **validateur qui rejoue les règles du
bot de soumission** (SpriteBot) : dimensions, offsets, ombres, palette,
semi-transparence, index d'animation.

---

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install Pillow numpy pytest

# récupère SpriteCollab (clone partiel : ~10 Mo/Pokémon au lieu de 1,9 Go)
./tools/fetch_upstream.sh 0025 0006 0094 0133
```

## Utilisation

```bash
# sprite Dynamax complet + animation de transformation + planches FX
.venv/bin/python -m pmdmax build 0025 --scale 1.6

# uniquement les frames du rayon, sur fond magenta
.venv/bin/python -m pmdmax fx --frames 16 --out output/fx

# validation stricte (les mêmes contrôles que le bot)
.venv/bin/python -m pmdmax verify output/0025-dynamax

# aperçus : GIF à la vitesse du jeu, planche de contact
.venv/bin/python -m pmdmax preview output/0025-dynamax --anims Walk
.venv/bin/python -m pmdmax preview output/0025-dynamax --contact

# export Aseprite (.ase) avec calques séparés
.venv/bin/python -m pmdmax ase output/0025-dynamax Walk

# galerie visuelle dans le navigateur
.venv/bin/python tools/gallery.py --port 3000
```

### Options utiles de `build`

| Option | Effet | Défaut |
|---|---|---|
| `--scale` | agrandissement du Pokémon | `1.6` |
| `--scale-mode` | `epx` (lissé) ou `nearest` (gros pixels) | `epx` |
| `--clouds` | nombre de nuages (3 = 3 tours de Dynamax) | `3` |
| `--cloud-scale` / `--cloud-gap` | taille des touffes / hauteur au-dessus du crâne | auto / `6` |
| `--orbit-rx` | rayon de l'orbite | auto |
| `--revolutions` | tours de nuages par boucle d'animation | `1.0` |
| `--no-bolts` | supprime les arcs électriques | — |
| `--no-aura` | désactive l'aura de fluide | — |
| `--aura-reach` | portée de l'aura en pixels | auto |
| `--aura-strength` | intensité de l'aura (0..1) | `1.0` |
| `--aura-cycles` | ondulations par boucle d'animation | `1.0` |
| `--anims Walk,Idle` | limite le traitement (test rapide) | toutes |
| `--transform-frames` | frames de la transformation | `16` |
| `--seed` | graine aléatoire (reproductible) | `1` |

---

## Le format strict, et comment il est respecté

Chaque animation est un **triplet de PNG de dimensions identiques** plus une
entrée dans `AnimData.xml` :

| Fichier | Contenu |
|---|---|
| `<Anim>-Anim.png` | les pixels du Pokémon |
| `<Anim>-Offsets.png` | 1 px **noir** (tête), **rouge** (main G), **vert** (centre), **bleu** (main D) par tuile |
| `<Anim>-Shadow.png` | blobs d'ombre (**vert**/**rouge**/**bleu** selon `ShadowSize`) + 1 px **blanc** = ancrage au sol |

Les règles appliquées, toutes vérifiées par `pmdmax verify` :

- grille = **colonnes : frames**, **lignes : directions** (1 ou 8, jamais autre chose) ;
- feuille divisible par `FrameWidth`/`FrameHeight`, tuiles **multiples de 8** ;
- `len(<Durations>) == nombre de frames` ; `Rush/Hit/ReturnFrame` dans les bornes ;
- `ShadowSize` ∈ 0..2 ; noms d'action ∈ `sprite_config.json` ;
- **index réservés** respectés (0 Walk, 1 Attack, 5 Sleep, 6 Hurt, 7 Idle,
  8 Swing, 9 Double, 10 Hop, 11 Charge, 12 Rotate) — la transformation est
  donc rangée dans un slot `Special*` libre ;
- **aucun pixel semi-transparent** (l'alpha vaut 0 ou 255, jamais entre) ;
- **un seul** pixel vert par tuile d'offsets, **un seul** pixel blanc par
  tuile d'ombre ;
- comptage de palette : au-delà de 15 couleurs, l'outil rappelle qu'il faut
  soumettre avec `--colors N`.

> **Détail du format souvent oublié** — un pixel **blanc** dans
> `-Offsets.png` n'est pas une erreur : c'est le raccourci officiel signifiant
> « les quatre ancres sont au même point » (cf. `SpriteBot/utils.py`,
> `getOffsetFromRGB`). Plusieurs sprites du dépôt s'en servent (Gengar, par
> exemple). `pmdmax` le lit **et** le réécrit correctement.

### Pourquoi la palette ne dérape jamais

L'agrandissement utilise **EPX / Scale2x / Scale3x** (`pmdmax/graphicscale.py`),
qui ne font que **recopier des pixels existants**. Aucun filtre bilinéaire :
la palette de sortie est un **sous-ensemble exact** de la palette d'entrée et
l'alpha reste binaire. Les facteurs non entiers montent en EPX jusqu'à un
entier composable puis redescendent en *nearest*.

Résultat mesuré sur Pikachu : **11 couleurs d'origine + 6 couleurs Dynamax
= 17**, sans une seule couleur parasite. C'est vérifié par un test.

---

## Direction artistique

La palette Dynamax (6 tons) a été **extraite par quantification de planches
de référence produites par le générateur d'images** (`refs/`), puis calée sur
une rampe lisible en 16 bits :

| | | |
|---|---|---|
| `#2B0010` contour | `#700526` face inférieure | `#BA153E` corps |
| `#E92D60` face éclairée | `#FC6EA0` liseré | `#FFD0E2` cœur des éclairs |

Les nuages suivent le modèle des jeux (Sword/Shield, Pokémon GO) : **des
touffes distinctes** qui orbitent sur une ellipse écrasée par la perspective,
et **non** une nébuleuse unique. Celles qui passent **derrière** la tête sont
plus petites et plus sombres, celles qui passent **devant** sont plus grosses
et plus claires — la couronne est donc composée en **deux passes**, sous et
sur le sprite, ce qui donne la vraie sensation de rotation.

### Fluidité de l'animation

Trois règles, vérifiées par des tests :

- **rien n'est tiré au hasard par frame.** La forme des lobes d'une touffe
  dépend de son identité, pas du temps : sans ça, chaque nuage grésille.
- **toutes les fréquences sont entières en phase.** La frame N se raccorde
  donc exactement à la frame 0 (`test_clouds_loop_exactly`,
  `test_aura_animates_and_loops` comparent phase 1.0 et phase 0.0 au pixel
  près).
- **les événements durent.** Un arc électrique s'étale sur plusieurs frames
  avec un zigzag figé, au lieu de clignoter aléatoirement.

Deux tests mesurent le nombre de pixels qui changent entre frames voisines et
échouent si un saut brutal apparaît.

### L'aura de fluide rouge

Elle n'est pas plaquée : elle est **dérivée du sprite lui-même**, après
agrandissement. Pour chaque frame et chaque direction, on calcule la
**distance euclidienne** à la silhouette réelle, puis on module la portée par
un champ d'ondes exprimé dans le repère **angulaire** de la forme.

Conséquence : les oreilles de Pikachu, les ailes de Dracaufeu ou la rondeur
d'Ectoplasma produisent automatiquement une aura qui épouse leur contour,
sans le moindre réglage manuel. Le rendu se fait en trois zones — gaine rose
fine collée au trait, lueur, puis langues de fluide qui s'étirent vers le
haut et se dissolvent en tramage.

Les planches de référence par Pokémon (`refs/aura_*.png`) ont été produites
en repassant les **vrais sprites** dans le générateur d'images ; elles ont
servi à caler la direction artistique du module.

## L'animation de transformation

16 frames, minutées comme dans le jeu (montée lente, impact très bref) :

| Frames | Étape |
|---|---|
| 1-2 | le sol s'illumine, des étincelles montent |
| 3-6 | **la colonne d'énergie rouge opaque** tombe (chute accélérée) |
| 7-8 | **impact** : flash blanc, onde de choc, le Pokémon commence à grandir |
| 9-12 | **les éclairs rouges montent en spirale** autour de lui |
| 13-15 | les spirales **condensent les nuages** au-dessus de sa tête |
| 16 | tout se stabilise : taille Dynamax, nuages en rotation |

Elle est écrite comme une animation SpriteCollab normale (dans un slot
`Special*`), **et** exportée à part en planches :

```
output/<mon>-dynamax/fx/
  Dynamax-FX-magenta.png      fond #FF00FF, prêt pour le chroma-key
  Dynamax-FX-magenta-x4.png   idem, agrandi x4 pour inspection
  Dynamax-FX-alpha.png        fond transparent, pour composer directement
```

## Aseprite

`pmdmax ase` écrit un vrai fichier `.ase` (spec officielle : en-tête 128 o,
chunks `Layer`/`Cel`/`Tags`, cels RGBA compressés zlib) — **sans binaire
Aseprite requis** — avec les calques séparés :

```
Nuages    ← retouchables sans toucher au Pokémon
Pokemon
Ombre     ← référence
Offsets   ← référence
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v     # 64 tests
```

Ils couvrent notamment : l'invariance de palette de tous les facteurs
d'échelle, l'absence de semi-transparence, l'aller-retour `AnimData.xml`,
la convention du pixel blanc d'offsets, la relecture du `.ase` selon la
spec, le storyboard du FX, et le rejet effectif des sprites non conformes.

## Structure

```
pmdmax/
  config.py        constantes du format + palette Dynamax
  pixels.py        primitives pixel-art (index, ombrage volumétrique)
  graphicscale.py  EPX/Scale2x/Scale3x, sans création de couleur
  sheet.py         triplets Anim/Offsets/Shadow + AnimData.xml
  clouds.py        nuages tournoyants
  aura.py          aura de fluide ondulant, calquée sur la silhouette
  beam.py          colonne d'énergie + éclairs spiralés
  dynamax.py       pipeline sprite normal -> Dynamax
  transform.py     animation de transformation
  verify.py        validateur strict (règles SpriteBot)
  aseprite.py      export .ase multi-calques
  preview.py       GIF et planches de contact
  cli.py           ligne de commande
tools/
  fetch_upstream.sh  clone partiel de SpriteCollab
  gallery.py         galerie HTML
tests/               64 tests
```

## Licence et usage

Les sprites d'origine viennent de SpriteCollab et sont sous
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) : usage
**non commercial**, **crédit** aux auteurs (voir `credits.txt`, recopié dans
chaque dossier de sortie).

⚠️ Dynamax est une **forme non officielle** dans l'univers PMD. La politique
de SpriteCollab **n'accepte pas** les Pokémon/formes non officiels dans le
dépôt principal : ces sprites sont donc destinés à un **usage personnel, un
romhack ou un fork**, pas à une soumission sur le dépôt officiel. Le format
est néanmoins respecté à la lettre pour rester utilisable par les outils de
l'écosystème (SkyTemple, etc.).
