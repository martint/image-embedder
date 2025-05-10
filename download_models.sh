#!/bin/bash

set -e

MODEL="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_DIR="models/clip-vit-h14"

mkdir -p "$MODEL_DIR"

echo "Downloading CLIP model files into $MODEL_DIR..."

wget -O "$MODEL_DIR/model.safetensors" \
  https://huggingface.co/"$MODEL"/resolve/main/model.safetensors?download=true

wget -O "$MODEL_DIR/vocab.json" \
  https://huggingface.co/"$MODEL"/resolve/main/vocab.json

wget -O "$MODEL_DIR/merges.txt" \
  https://huggingface.co/"$MODEL"/resolve/main/merges.txt

wget -O "$MODEL_DIR/tokenizer.json" \
  https://huggingface.co/"$MODEL"/resolve/main/tokenizer.json

echo "✅ CLIP model files downloaded."
