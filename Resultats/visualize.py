import json

import matplotlib.pyplot as plt

def plot_training_history(data):
    # Séparation des données d'entraînement (epochs numériques) et de test
    epochs = [d['epoch'] for d in data if isinstance(d['epoch'], int)]
    train_loss = [d['train_loss'] for d in data if isinstance(d['epoch'], int)]
    val_kid = [d['val_kid'] for d in data if isinstance(d['epoch'], int)]
    val_score = [d['val_judge_score'] for d in data if isinstance(d['epoch'], int)]
    
    # Récupération des scores de test (dernier élément)
    test_data = data[-1]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Train Loss
    axs[0].plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].grid(True)

    # Plot KID (Kernel Inception Distance) - Plus bas est mieux
    axs[1].plot(epochs, val_kid, label='Val KID', color='orange', marker='s')
    if 'test_kid' in test_data:
        axs[1].axhline(y=test_data['test_kid'], color='red', linestyle='--', label='Test KID')
    elif 'val_kid' in test_data: # Fallback si le test est juste la dernière val
        axs[1].axhline(y=test_data['val_kid'], color='red', linestyle='--', label='Last Val KID')

    axs[1].set_title('KID Score (Lower is better)')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Judge Score
    axs[2].plot(epochs, val_score, label='Val Judge Score', color='green', marker='^')
    if 'test_judge_score' in test_data:
        axs[2].axhline(y=test_data['test_judge_score'], color='red', linestyle='--', label='Test Judge Score')
    elif 'val_judge_score' in test_data: # Fallback si le test est juste la dernière val
        axs[2].axhline(y=test_data['val_judge_score'], color='red', linestyle='--', label='Last Val Judge Score')
    axs[2].set_title('Judge Score')
    axs[2].set_xlabel('Epoch')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

try:
    with open('history2.json', 'r') as f:
        donnees = json.load(f)
    plot_training_history(donnees)
except FileNotFoundError:
    print("Erreur : Le fichier 'history2.json' est introuvable.")