# Séquence d'intro — de l'écran noir au portail

**Principe : ANIMATION = BOUCLE INFINIE, DIALOGUE = PROGRESSION.**
Rien n'avance tout seul. Chaque état boucle indéfiniment ; c'est l'appui du
joueur sur CONTINUER qui joue la transition vers l'état suivant.

## États (boucle infinie)

| Dossier | Contenu | Frames |
|---|---|---|
| `01_vide/` | Écran noir, la sphère-âme s'illumine et respire | 36 |
| `02_ciel/` | Bleu nuit étoilé, 3 nappes d'étoiles en parallaxe | 36 |
| `03_halo/` | Halo de spectre arc-en-ciel, 6 couronnes | 36 |
| `04_portail/` | Voyage à l'intérieur du tunnel d'énergie | 36 |

Les quatre bouclent **au bit près** : `render(0.0) == render(1.0)`.

## Transitions (jouées UNE fois)

| Dossier | Rôle | Flash |
|---|---|---|
| `T1_ebranlement/` | La sphère s'ébranle et part vers le ciel | — |
| `T2_flash/` | Arrêt du voyage, **flash blanc progressif** | 0.92 |
| `T3_aspiration/` | Le halo s'ouvre, la sphère est aspirée | 0.35 |

Chaque transition part de l'état N à sa **phase 0** et arrive sur l'état N+1
à sa **phase 0**. Le joueur ne voit donc jamais de saut : on laisse la boucle
courante se terminer, puis on joue la transition.

## Intégration

```
état courant → boucle sur ses 36 frames, indéfiniment
appui CONTINUER → jouer les 18 frames de la transition, une fois
                → entrer dans la boucle suivante à sa frame 000
```

`sequence.json` donne pour chaque clip : frames, fps, durée, point de boucle,
raccord exact, déclencheur.

## Régénérer

```bash
.venv/bin/python -c "from soulhalo.sequence import build, BuildSpec; \
  build('output/intro', BuildSpec(w=960, h=540, loop_frames=36, fps=20))"
```
