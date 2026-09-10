# Base d’équipe de secours — style Pokémon Donjon Mystère

Création fan art originale inspirée de l’ambiance chaleureuse et organique des bases de *Pokémon Donjon Mystère : Équipe de Secours Rouge et Bleue*.

La base reste volontairement générique : aucun Pokémon n’est imposé comme mascotte. La disposition et le mobilier reprennent le langage visuel de *Rescue Team* sans reproduire exactement une carte du jeu.

## Structure extérieure

- `exterior.png`
- Planche modulaire transparente : **338 × 267 px**
- Tilesheet entièrement isolé : deux états de façade, une base assemblée, une bannière, un comptoir, un tapis, une boîte aux lettres et trois accessoires séparés
- Aucun élément ne se touche et aucun sol, chemin ou décor extérieur n’est attaché aux sprites
- Architecture arrondie en bois, paille et feuillage, sans toit humain triangulaire

## Intérieur

Tous les fichiers intérieurs possèdent un canevas transparent de **512 × 384 px** et s’alignent directement. La salle adopte une **vue zénithale/top-down**, proche de la présentation des cartes intérieures de *Rescue Team*. Ses murs en paille dorée, ses nervures en bois brun et son sol verdoyant reprennent directement les matériaux et les couleurs de la maison extérieure.

Ordre d’empilement, du bas vers le haut :

1. `interior_layer_01_architecture.png` — anneau mural en paille dorée, nervures en bois assorties, sol végétal en planches et entrée basse
2. `interior_layer_02_furniture.png` — couchages, table de préparation, coffre, étagère et provisions
3. `interior_layer_03_vines_and_flag.png` — lianes décoratives et drapeau simple de l’équipe de secours
4. `interior_layer_04_foreground.png` — tapis d’entrée vu du dessus

Autres fichiers :

- `interior_composite.png` — assemblage aplati exact des quatre calques
- `rescue_team_base_interior.ora` — document OpenRaster avec les quatre calques, compatible avec Krita et GIMP
- `apercu_base_rescue_team.jpg` — aperçu de l’extérieur et de l’intérieur assemblé

## Références visuelles

- Rescue Team Base : https://mysterydungeonwiki.com/wiki/Rescue_Team:Rescue_Team_Base
- Cartes de *Pokémon Mystery Dungeon: Red Rescue Team* : https://archives.bulbagarden.net/wiki/Category:Pok%C3%A9mon_Mystery_Dungeon_Rescue_Team_maps
- Ressources de *Rescue Team DX* : https://models.spriters-resource.com/nintendo_switch/pokemonmysterydungeonrescueteamdx/

Ces images sont des créations fan art originales et ne sont pas des sprites extraits des jeux.
