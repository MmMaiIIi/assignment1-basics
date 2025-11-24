#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Base URLs for hf-mirror.com
TINY_BASE="https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main"
OWT_BASE="https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main"

# Helper: download if not exists
download_if_missing() {
  local url="$1"
  local fname="$2"

  if [ -f "$fname" ]; then
    echo "[skip] $fname already exists"
  else
    echo "[download] $url -> $fname"
    wget "$url" -O "$fname"
  fi
}

# TinyStories
download_if_missing "$TINY_BASE/TinyStoriesV2-GPT4-train.txt" "TinyStoriesV2-GPT4-train.txt"
download_if_missing "$TINY_BASE/TinyStoriesV2-GPT4-valid.txt" "TinyStoriesV2-GPT4-valid.txt"

# OWT sample (gzipped)
download_if_missing "$OWT_BASE/owt_train.txt.gz" "owt_train.txt.gz"
download_if_missing "$OWT_BASE/owt_valid.txt.gz" "owt_valid.txt.gz"

# Unzip (only if plain txt not present)
if [ -f "owt_train.txt.gz" ] && [ ! -f "owt_train.txt" ]; then
  echo "[gunzip] owt_train.txt.gz"
  gunzip "owt_train.txt.gz"
fi

if [ -f "owt_valid.txt.gz" ] && [ ! -f "owt_valid.txt" ]; then
  echo "[gunzip] owt_valid.txt.gz"
  gunzip "owt_valid.txt.gz"
fi

echo "All done."
