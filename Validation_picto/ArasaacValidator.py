import torch
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal import CLIPScore
import lpips

import torch.nn as nn

from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    MobileViTImageProcessor, 
    MobileViTForImageClassification
)



#ARCHITECTURE DU JUGE(model entraine)---
class ArasaacJudge(nn.Module):
    def __init__(self, input_dim=4):
        super(ArasaacJudge, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.classifier(x)








class ArasaacValidator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"***Initialisation du Pipeline sur : {self.device} ---")
        
        # Modèles pour la Sémantique & Style (CLIP / LPIPS)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)  # SOTA today
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(device)
       
        #  Model pour le SENS : MobileViT (Léger et Rapide)
        self.sense_ckpt = "apple/mobilevit-small"
        self.sense_processor = MobileViTImageProcessor.from_pretrained(self.sense_ckpt)
        self.sense_model = MobileViTForImageClassification.from_pretrained(self.sense_ckpt).to(device)

        # Métrique Distributionnelle (KID)
        # subset_size=50 pour la stabilité statistique sur petits échantillons
        self.kid_metric = KernelInceptionDistance(subset_size=15, normalize=True).to(device)
       

        # CHARGEMENT DU JUGE MLP ---
        try:
            # On charge le fichier .pth (qui contient poids + stats)
            checkpoint = torch.load("arasaac_judge_final.pth", map_location=self.device)

            # On crée l'instance du modèle (défini au début du fichier)
            self.judge_model = ArasaacJudge(input_dim=4).to(self.device)
            self.judge_model.load_state_dict(checkpoint['model_state_dict'])
            self.judge_model.eval()     # Mode évaluation (important !)
 
            # On récupère les moyennes et écart-types pour le calcul du score
            s = checkpoint['stats']
            self.means = np.array([s['brenner']['mean'], s['entropy']['mean'], s['clip']['mean'], s['sense']['mean']])
            self.stds = np.array([s['brenner']['std'], s['entropy']['std'], s['clip']['std'], s['sense']['std']])

        except Exception as e:
            print(f"ERREUR ")


   
    # ==========================================
    # CATEGORIE 1 : SEMANTIQUE (CLIP)
    
    def _get_clip_alignment(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image , return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()


    # ==========================================
    # CATEGORIE 2 : TEXTURE & Contrast (Brenner/Entropie)
    
    def _get_texture_metrics(self, img):
        gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
        
        # Brenner : Netteté du trait
        diff = gray[2:, :] - gray[:-2, :]
        brenner = np.sum(diff**2) / (gray.shape[0] * gray.shape[1])
        
        # Entropie : Simplicité (entropie de Shannon)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        return brenner, entropy


    # ==========================================
    # CATEGORIE 3 : STYLE & PERCEPTION (KID/LPIPS)

    def update_style_reference(self, folder_path):
        """Prend les vrais pictos ARASAAC comme référence de style."""
        print(f"Calcul de la signature de style sur : {folder_path}")
        for img_p in Path(folder_path).glob("*.*"):
            img = Image.open(img_p).convert("RGB").resize((299, 299))
            img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.kid_metric.update(img_t, real=True)
    

    # =====================================================
    # CATEGORIE 4 : SENS (classifier)

    def get_sense_score(self, img):
        inputs = self.sense_processor(images=img , return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sense_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # On prend la confiance maximale (Top-1)
        top_conf, _ = torch.max(probs, dim=-1)
        return top_conf.item()

   




    # ******************************************************************
    # Fonction principale 
    def get_metrics(self, img, prompt):
        img_np = np.array(img)
        # On appelle les méthodes internes
        br, ent = self._get_texture_metrics(img_np)
        clip = self._get_clip_alignment(img, prompt)
        sense = self.get_sense_score(img)
        return br, ent, clip, sense

 


    # *******************************************************************
    # appel au Judge

    def predict_score(self, img, prompt):
        """Retourne le score final du Juge (0 à 1)."""
        # Extraction des metrics
        br, ent, cl, sn = self.get_metrics(img, prompt)
        features = np.array([br, ent, cl, sn])

        # Normalisation (Z-Score)
        features_scaled = (features - self.means) / self.stds

        # Prédiction
        input_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            score = self.judge_model(input_tensor).item()

        return score
