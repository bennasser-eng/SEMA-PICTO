#!/bin/bash

source ../picto/env/bin/activate

#  Lancement  du test de validation du Juge
python3 - << 'EOF'

import torch
#import torch.nn as nn
from ArasaacValidator import ArasaacValidator
from PIL import Image
import os

# Initialisation du validateur
validator = ArasaacValidator()

# On prend une image Arasaac et son équivalent SDXL
picto_path = "../picto/picto774/3051.png"
photo_path = "../picto/images_real/3051.png"
prompt = "Concave brass disks struck together or with sticks, producing loud crashing sounds."

def check_score(path, label):
    if not os.path.exists(path):
        print(f"[-] Fichier introuvable : {path}")
        return
    
    img = Image.open(path).convert("RGB")
    score = validator.predict_score(img, prompt)
    
    print(f"\n *****TEST : {label}")
    print(f"    Fichier : {path}")
    print(f"    Score Arasaac : {score:.4f}")
    
    if score > 0.6:
        print("    Statut : STYLE VALIDE (Arasaac)")
    elif score < 0.4:
        print("    Statut :  STYLE REJETÉ (Photo Réel)")
    else:
        print("    Statut :  STYLE AMBIGU (Entre-deux)")

print("="*50)

print("VÉRIFICATION RAPIDE DU JUGE MLP")
print()

check_score(picto_path, "VRAI PICTOGRAMME")
check_score(photo_path, "IMAGE RÉELLE (SDXL)")

EOF
