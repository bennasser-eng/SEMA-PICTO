#!/usr/bin/env python3
# split_data.py

import json
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, metadata_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split un dataset d'images en train/val/test
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    metadata_path = Path(metadata_file)
    
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        (d / "images").mkdir(parents=True, exist_ok=True)
    
    # Lire le metadata.jsonl
    text_dict = {}
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # Enlever le prefixe "images/" si present
                file_name = item["file_name"]
                if file_name.startswith("images/"):
                    file_name = file_name.replace("images/", "")
                text_dict[file_name] = item["text"]
    
    print(f"Nombre d'entrees dans metadata: {len(text_dict)}")
    
    # Lister toutes les images
    all_images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
    all_images = [f.name for f in all_images]
    
    print(f"Nombre d'images trouvees: {len(all_images)}")
    
    # Verifier
    missing = [img for img in all_images if img not in text_dict]
    if missing:
        print(f"Attention: {len(missing)} images sans texte dans metadata")
        for img in missing[:5]:
            print(f"  - {img}")
        # Ne pas mettre de fallback, on leve une erreur
        raise ValueError(f"Les images suivantes n'ont pas de texte: {missing[:5]}")
    
    # Melanger
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    print(f"\nRepartition:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    def process_split(images, split_dir, split_name):
        # Copier les images
        for img_name in images:
            src = source_path / img_name
            dst = split_dir / "images" / img_name
            shutil.copy2(src, dst)
        
        # Creer metadata.jsonl
        with open(split_dir / "metadata.jsonl", 'w') as f:
            for img_name in images:
                line = json.dumps({
                    "file_name": img_name,
                    "text": text_dict[img_name]
                })
                f.write(line + "\n")
        
        print(f"  {split_name}: {len(images)} images -> {split_dir}/metadata.jsonl")
    
    print("\nCreation des splits:")
    process_split(train_images, train_dir, "Train")
    process_split(val_images, val_dir, "Val")
    process_split(test_images, test_dir, "Test")
    
    print(f"\nTermine! Structure creee dans {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./dataset_picto")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    split_dataset(
        args.source_dir,
        args.metadata_file,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
