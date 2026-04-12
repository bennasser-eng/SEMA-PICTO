#!/bin/bash

source picto/env/bin/activate

# Paramètres
PICTO_DIR="../picto/picto774"
REAL_DIR="../picto/images_real"
DEF_FILE="../picto/definitions.csv"
OUTPUT_CSV="dataset_judge.csv"
MIXUP_COUNT=1000


python3 - << 'EOF' "$PICTO_DIR" "$REAL_DIR" "$DEF_FILE" "$OUTPUT_CSV" "$MIXUP_COUNT"

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import re


# accès à ArasaacValidator
sys.path.append(os.getcwd())
from ArasaacValidator import ArasaacValidator

picto_dir, real_dir, def_file, out_file, n_mixup = sys.argv[1:6]
n_mixup = int(n_mixup)

# ÉTAPE 1 : Extraire les définitions du CSV ---
# Format : id_img [tags] le texte de la définition
prompt_map = {}
print(f"Chargement des définitions depuis {def_file}...")

try:
    with open(def_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Regex pour capturer : ID (avant le premier espace) et le Texte (après les crochets)
            # Exemple: 1234 [tag1, tag2] A red apple on a table
            match = re.match(r"^(\d+)\s+\[.*?\]\s+(.*)$", line)
            if match:
                img_id, description = match.groups()
                prompt_map[img_id] = description
except Exception as e:
    print(f"Erreur lecture CSV : {e}")
    sys.exit(1)


# ÉTAPE 2 : Initialisation ---
validator = ArasaacValidator()

ids_picto = {f.stem for f in Path(picto_dir).glob("*.png")}
ids_real = {f.stem for f in Path(real_dir).glob("*.png")}
common_ids = [i for i in ids_picto.intersection(ids_real) if i in prompt_map]

results = []

print(f"Étape 3 : Calcul des features sur {len(common_ids)} paires...")
for img_id in tqdm(common_ids):
    text_prompt = prompt_map[img_id]
    
    try:
        # Vrai Picto (Label 1.0)
        img_p = Image.open(f"{picto_dir}/{img_id}.png").convert("RGB")
        # On passe la définition réelle à CLIP
        br_p, ent_p, cl_p, sn_p = validator.get_metrics(img_p, text_prompt)
        results.append({"id": img_id, "brenner": br_p, "entropy": ent_p, "clip": cl_p, "sense": sn_p, "label": 1.0})
        
        # Image Réelle (Label 0.0)
        img_r = Image.open(f"{real_dir}/{img_id}.png").convert("RGB")
        br_r, ent_r, cl_r, sn_r = validator.get_metrics(img_r, text_prompt)
        results.append({"id": img_id, "brenner": br_r, "entropy": ent_r, "clip": cl_r, "sense": sn_r, "label": 0.0})
    except Exception as e:
        continue

# ÉTAPE 4 : Mixup & Sauvegarde ---
df_base = pd.DataFrame(results)
hybrid_data = []
for _ in range(n_mixup):
    rid = np.random.choice(common_ids)
    alpha = np.random.uniform(0.1, 0.9)
    p = df_base[(df_base['id'] == rid) & (df_base['label'] == 1.0)].iloc[0]
    r = df_base[(df_base['id'] == rid) & (df_base['label'] == 0.0)].iloc[0]
    hybrid_data.append({
        "id": f"{rid}_mix_{alpha:.2f}",
        "brenner": alpha * p['brenner'] + (1 - alpha) * r['brenner'],
        "entropy": alpha * p['entropy'] + (1 - alpha) * r['entropy'],
        "clip": alpha * p['clip'] + (1 - alpha) * r['clip'],
        "sense": alpha * p['sense'] + (1 - alpha) * r['sense'],
        "label": alpha
    })

df_final = pd.concat([df_base, pd.DataFrame(hybrid_data)], ignore_index=True)
stats = df_final[['brenner', 'entropy', 'clip', 'sense']].agg(['mean', 'std'])
stats.to_csv("normalization_stats.csv")
df_final.to_csv(out_file, index=False)
print(f"Dataset prêt : {len(df_final)} lignes.")
EOF
