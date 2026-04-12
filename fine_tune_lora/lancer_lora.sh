#!/bin/bash

source ../picto/env/bin/activate

# Prompt par défaut
DEFAULT_PROMPT="a black cat sleeping on a red cushion, minimalist line art, style arasaac"

PROMPT="${1:-$DEFAULT_PROMPT}"

# Nom du fichier de sortie (par défaut ou basé sur le prompt)
if [ -n "$1" ]; then
    OUTPUT_NAME=$(echo "$1" | tr ' ' '_' | cut -c1-50).png
else
    OUTPUT_NAME="picto_generated.png"
fi

echo "=========================================="
echo "Generation de pictogramme avec LoRA"
echo "=========================================="
echo "Prompt: $PROMPT"
echo "Sortie: $OUTPUT_NAME"
echo "=========================================="

# Lancer le script Python
python3 <<EOF
import torch
from diffusers import StableDiffusionPipeline
import sys
import os

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_picto_model/final"
PROMPT = """$PROMPT"""
OUTPUT_FILE = "$OUTPUT_NAME"

print("Chargement du pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

print(f"Chargement des poids LoRA depuis {LORA_PATH}...")
pipe.load_lora_weights(LORA_PATH)

print(f"Generation de l'image pour: {PROMPT}")
with torch.no_grad():
    with torch.autocast("cuda"):
        image = pipe(
            PROMPT,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

image.save(OUTPUT_FILE)
print(f"Image sauvegardee: {OUTPUT_FILE}")
print("Termine!")
EOF

echo ""
echo "Generation terminee! Fichier: $OUTPUT_NAME"
