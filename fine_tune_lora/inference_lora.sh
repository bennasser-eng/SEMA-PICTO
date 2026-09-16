#!/bin/bash

# Chargement de l'environnement
source ../picto/env/bin/activate

#  script python 
python3 - << 'EOF'
import torch
from diffusers import StableDiffusionPipeline
import os

# Configuration
model_id = "runwayml/stable-diffusion-v1-5"
#lora_path = "./lora_picto_model_2/final"
lora_path = "./rank64_alpha64/final"

output_dir = "./outputs_inference/rank64,alpha64"
os.makedirs(output_dir, exist_ok=True)

print(f"Chargement du modèle {model_id}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

if os.path.exists(lora_path):
    print(f"Chargement du LoRA: {lora_path}")
    #pipe.unet.load_adapter(lora_path)
    pipe.load_lora_weights(lora_path)

else:
    print(f" Checkpoint introuvable à {lora_path}")

prompts = ["A fast dog sprints across the grass.", 
           "A bright crimson vehicle drives down the road" ,
           "A young student looks at the pages of a story",
            "A single lowercase alphabetic character shaped with a rounded bowl and vertical stem",
            "A bold yellow triangle warns people of a hazard.",
            "Tenth lowercase letter of the Latin alphabet, representing a nasal consonant sound"
]

print(f" Génération en cours...")
for i, prompt in enumerate(prompts):
    with torch.autocast("cuda"):
        image = pipe(
            prompt, 
            num_inference_steps=40, 
            guidance_scale=8.5
        ).images[0]
    
    clean_name = prompt.replace(" ", "_").replace(",", "")[:30]
    save_path = os.path.join(output_dir, f"test_{i}_{clean_name}.png")
    image.save(save_path)
    print(f"Sauvegardée: {save_path}")
EOF
