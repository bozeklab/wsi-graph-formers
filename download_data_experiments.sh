#!/usr/bin/env bash
#
# Download the data needed to reproduce the additional experiments (~391 MB)
# from Zenodo into ./data
#   Record: https://doi.org/10.5281/zenodo.21415205
#
# Creates:
#   data/experiments/
#   data/WSI-Graph-100splits/
#
# Requires curl and unzip. Works on Linux and macOS.

set -euo pipefail

BASE="https://zenodo.org/records/21415205/files"

# Resolve to the script's own directory so data/ lands in the repo root
# regardless of where the script is called from.
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p data

for f in experiments WSI-Graph-100splits; do
  if [ -d "data/$f" ]; then
    echo "$f already present, skipping"
    continue
  fi
  echo "Downloading $f.zip ..."
  curl -fL --retry 3 --progress-bar -o "data/$f.zip" "$BASE/$f.zip?download=1"
  unzip -q "data/$f.zip" -d data/
  rm "data/$f.zip"
  # Some archives were packaged on macOS and ship a __MACOSX/ resource-fork
  # tree, which unzip extracts literally. Not part of the dataset.
  rm -rf "data/__MACOSX"
done

echo
echo "Done. Contents of data/:"
ls -1 data