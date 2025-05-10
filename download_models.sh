#!/bin/bash

set -e

MODEL_DIR="models/clip-vit-b32-patch32"
mkdir -p "$MODEL_DIR"

echo "Downloading CLIP model files into $MODEL_DIR..."

wget -O "$MODEL_DIR/model.safetensors" \
  https://huggingface.co/openai/clip-vit-base-patch32/resolve/d15b5f29721ca72dac15f8526b284be910de18be/model.safetensors

wget -O "$MODEL_DIR/vocab.json" \
  https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/vocab.json

wget -O "$MODEL_DIR/merges.txt" \
  https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/merges.txt

wget -O "$MODEL_DIR/tokenizer.json" \
  https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/tokenizer.json

echo "✅ CLIP model files downloaded."
