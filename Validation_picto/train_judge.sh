#!/bin/bash

source ../picto/env/bin/activate

# Exécution du pipeline d'entraînement
python3 - << 'EOF'

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os



# --- ARCHITECTURE DU JUGE ---
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



# --- PIPELINE ---
def run_training():
    print("--- Début d'entraînement du Juge ARASAAC ---")
    
    # Chargement des données
    if not os.path.exists("dataset_judge.csv") or not os.path.exists("normalization_stats.csv"):
        print("Erreur : Fichiers CSV introuvables..")
        return

    df = pd.read_csv("dataset_judge.csv")
    stats = pd.read_csv("normalization_stats.csv", index_col=0)
    
    cols = ["brenner", "entropy", "clip", "sense"]
    
    # Normalisation  selon les stats globales
    X_raw = df[cols].values
    X_scaled = (X_raw - stats.loc['mean'].values) / stats.loc['std'].values
    
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32).view(-1, 1)

    # Initialisation du modèle
    model = ArasaacJudge(input_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()        # Utilisation de MSE pour gérer les labels Mixup

    # Boucle d'entraînement
    epochs = 50
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss MSE: {loss.item():.6f}")


    # Sauvegarde complète (Modèle + Stats pour l'inférence)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'stats': stats.to_dict(),
        'input_cols': cols
    }

    torch.save(checkpoint, "arasaac_judge_final.pth")
    print(f"--- Succès ! Modèle sauvegardé sous : arasaac_judge_final.pth ---")

if __name__ == "__main__":
    run_training()
EOF
