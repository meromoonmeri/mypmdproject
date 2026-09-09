#!/usr/bin/env bash
# Récupère les sources amont dans external/ (ignoré par Git).
#
# SpriteCollab pèse ~1,9 Go : on fait un clone "blobless" + sparse-checkout,
# ce qui ne télécharge que les sprites demandés (~10 Mo par Pokémon).
#
#   ./tools/fetch_upstream.sh                 # Pokémon par défaut
#   ./tools/fetch_upstream.sh 0025 0006 0143  # une liste précise
#   ./tools/fetch_upstream.sh --all           # tout (long, ~1,9 Go)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"
SC="$EXT/SpriteCollab"
SB="$EXT/SpriteBot"

DEFAULT_MONS=(0025 0006 0133 0094)

mkdir -p "$EXT"

# ---------------------------------------------------------------- SpriteCollab
if [ ! -d "$SC/.git" ]; then
  echo ">> Clone blobless de SpriteCollab..."
  git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/PMDCollab/SpriteCollab.git "$SC"
  git -C "$SC" sparse-checkout init --cone
fi

if [ "${1:-}" = "--all" ]; then
  echo ">> Checkout COMPLET (~1,9 Go, patience)..."
  git -C "$SC" sparse-checkout disable
  git -C "$SC" checkout master
else
  if [ "$#" -gt 0 ]; then MONS=("$@"); else MONS=("${DEFAULT_MONS[@]}"); fi
  PATHS=()
  for m in "${MONS[@]}"; do PATHS+=("sprite/$m"); done
  echo ">> Sparse-checkout : ${MONS[*]}"
  git -C "$SC" sparse-checkout set "${PATHS[@]}"
  git -C "$SC" checkout master
fi

# ------------------------------------------------------------------- SpriteBot
if [ ! -d "$SB/.git" ]; then
  echo ">> Clone de SpriteBot (référence du format)..."
  git clone --depth 1 https://github.com/PMDCollab/SpriteBot.git "$SB"
else
  git -C "$SB" pull --ff-only --quiet || true
fi

# sprite_config.json est la source de vérité des noms/index d'action :
# on en garde une copie versionnée pour que pmdmax marche sans le clone.
mkdir -p "$ROOT/data"
cp "$SC/sprite_config.json" "$ROOT/data/sprite_config.json"

echo
echo "OK."
echo "  SpriteCollab : $SC"
echo "  SpriteBot    : $SB"
echo "  Sprites      : $(ls "$SC/sprite" 2>/dev/null | wc -l) dossier(s)"
