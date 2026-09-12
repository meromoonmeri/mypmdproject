# Structures Pokémon Donjon Mystère

Douze planches originales de bâtiments et de stands inspirées de l’architecture de **Pokémon Donjon Mystère**.

## Format commun

- PNG RGBA avec fond transparent
- Canevas : **338 × 267 px**
- Caméra frontale orthographique
- Composition modulaire inspirée de la planche Arcanine :
  - deux variantes de façade en haut à gauche ;
  - une structure assemblée en bas à gauche ;
  - un tapis ou une bannière en haut à droite ;
  - un comptoir séparé ;
  - trois petits accessoires séparés.
- Aucun texte et aucun personnage vivant devant le bâtiment

## Fichiers

1. `01_spinda_cafe.png` — Café Spinda
2. `02_kecleon_shop.png` — Marché Kecleon
3. `03_kangaskhan_storage.png` — Réserve Kangourex
4. `04_duskull_bank.png` — Banque Skelénox
5. `05_electivire_link_shop.png` — Stand de capacités Élekable
6. `06_xatu_appraisal.png` — Expertise Xatu
7. `07_chansey_day_care.png` — Garderie Leveinard
8. `08_marowak_dojo.png` — Dojo Ossatueur
9. `09_pelipper_post_office.png` — Poste Bekipan
10. `10_wigglytuff_guild_outpost.png` — Avant-poste de la Guilde Grodoudou
11. `11_wobbuffet_stand.png` — Stand Qulbutoké : structure et disposition Arcanin conservées à l’identique, avec uniquement le masque et la palette remplacés
12. `12_eevee_shop.png` — Boutique Évoli (refaite) : tête Évoli symétrique, grandes oreilles triangulaires, col de fourrure crème en lobes arrondis et queue à bout crème ; les yeux reprennent le masque Arcanin à l’identique — amande inclinée cerclée de caramel, **barreaux verticaux sombres sur fond ambré et barre crème en appui, comme une fenêtre à meneaux**, sans pupille ni reflet

Aperçus :

- `apercu_10_structures.jpg` — collection initiale
- `apercu_11_structures.jpg` — collection incluant le stand Qulbutoké
- `apercu_12_structures.jpg` — collection incluant la première version de la boutique Évoli
- `apercu_13_structures.jpg` — collection avec la boutique Évoli refaite

## Méthode de la hutte Évoli refaite

La planche a été refaite avec un générateur d’images à partir de la planche Arcanin (structure, modules et emplacements conservés), puis retravaillée pour coller au gabarit du dépôt :

1. fond uni détouré par inondation depuis les bords → alpha dur `0 / 255`, aucun halo ;
2. mise à l’échelle du cadre complet en **338 × 267**, sans déformation : le contenu tombe en `x 29-330`, `y 3-257`, soit aux mêmes marges que la planche Arcanin (`x 30-327`, `y 3-253`) ;
3. réduction à **32 couleurs** (médiane-cut, sans trame) puis enregistrement en PNG RGBA transparent ;
4. itérations ciblées sur les yeux uniquement : disques ronds → hachures diagonales → **fenêtre à barreaux** reprise du gros plan des yeux de la hutte Arcanin (crop de référence utilisé comme image guide) ;
5. le tout est rejouable avec `scripts/finition_planche.py <planche_generee.png> <sortie.png>`, qui enchaîne les trois premières étapes.

La v1 (masque peint par remap de palette) reste consultable dans l’historique Git, ainsi que `apercu_12_structures.jpg`.

## Références étudiées

- Bulbapedia — Treasure Town : https://bulbapedia.bulbagarden.net/wiki/Treasure_Town
- Bulbapedia — Pokémon Square : https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_Square
- The Spriters Resource — Spinda Café : https://www.spriters-resource.com/ds_dsi/pokemonmysterydungeonexplorersofsky/asset/27765/
- The Spriters Resource — Treasure Town : https://www.spriters-resource.com/ds_dsi/pokemonmysterydungeonexplorersoftimedarkness/asset/5979/
- The Spriters Resource — Pelipper Post Office : https://www.spriters-resource.com/game_boy_advance/pokemonmysterydungeonredrescueteam/asset/5328/
- GeoisEvil / DeviantArt — recherches de tilesets et planche Arcanine : https://www.deviantart.com/geoisevil/art/New-Tiles-Wip-502890888
- Manuel Nintendo de Pokémon Mystery Dungeon: Red Rescue Team : https://www.nintendo.com/eu/media/downloads/games_8/emanuals/game_boy_advance_8/Manual_GameBoyAdvance_PokemonMysteryDungeonRedRescueTeam_EN.pdf

Ces images sont des créations fan art originales et ne sont pas des sprites extraits des jeux.
