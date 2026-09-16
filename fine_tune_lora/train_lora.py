#!/usr/bin/env python

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import os
import argparse
import json
import sys
import csv
import random
from PIL import Image

sys.path.append("../Validation_picto")
from ArasaacValidator import ArasaacValidator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_model")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--rank", type=int, default=64) # hyperparams !!!!!!!!!!!!
    parser.add_argument("--lambda_reward", type=float, default=10.0) #hyperparams !!!!!!!!!!
    parser.add_argument("--kid_samples", type=int, default=200, help="Nombre d'images pour KID")
    return parser.parse_args()


class LoRATrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initialisation sur {self.device}")
        # print(f"Lambda reward: {args.lambda_reward}")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder 
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank, #//2
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.pipe.unet = self.unet
        self.pipe = self.pipe.to(self.device)
        self.unet.print_trainable_parameters()
        
        self.validator = ArasaacValidator(device=self.device)
        
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=args.learning_rate)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name, subfolder="scheduler"
        )
        
        self.train_loader = self._load_dataset(args.train_data_dir, shuffle=True)
        self.val_loader = self._load_dataset(args.val_data_dir, shuffle=True)
        self.test_loader = self._load_dataset(args.test_data_dir, shuffle=False)
        
        self.history = []

    def _load_dataset_csv(self, data_dir, shuffle):
        metadata_path = os.path.join(data_dir, "metadata.csv")
        images_dir = os.path.join(data_dir, "images")
        
        # Lire toutes les lignes du CSV
        data = []
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        print(f"Chargement de {len(data)} images depuis {data_dir}")
        
        if shuffle:
            random.shuffle(data)
        
        transform = transforms.Compose([
            transforms.Resize(self.args.resolution),
            transforms.CenterCrop(self.args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        class PictoDataset(Dataset):
            def __init__(self, data, images_dir, transform, tokenizer, max_length):
                self.data = data
                self.images_dir = images_dir
                self.transform = transform
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                row = self.data[idx]
                img_path = os.path.join(self.images_dir, row["file_name"])
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                
                tokens = self.tokenizer(
                    row["text"],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    "pixel_values": img,
                    "input_ids": tokens.input_ids.squeeze(0),
                    "text": row["text"]
                }
        
        dataset = PictoDataset(data, images_dir, transform, self.tokenizer, self.tokenizer.model_max_length)
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle)
    

    def _load_dataset(self, data_dir, shuffle):
        # On cherche le fichier JSONL
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        images_dir = os.path.join(data_dir, "images")
        
        if not os.path.exists(metadata_path):
            print(f" ERREUR : Fichier introuvable -> {metadata_path}")
            sys.exit(1)
            
        # Chargement des données JSONL
        data = []
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():    # Évite les lignes vides
                        data.append(json.loads(line))
        except Exception as e:
            print(f" ERREUR lors de la lecture du JSONL : {e}")
            sys.exit(1)
            
        print(f"Chargement réussi : {len(data)} images trouvées dans {data_dir}")
        
        if shuffle:
            random.shuffle(data)
        
        # Définition des transformations (Prêt pour le Validator)
        transform = transforms.Compose([
            transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]), # Standard pour Stable Diffusion
        ])
        
        # Classe Dataset Interne
        class PictoDataset(Dataset):
            def __init__(self, data, images_dir, transform, tokenizer, max_length):
                self.data = data
                self.images_dir = images_dir
                self.transform = transform
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                row = self.data[idx]
                img_name = row["file_name"]
                img_path = os.path.join(self.images_dir, img_name)
                
                # Chargement de l'image
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = self.transform(img)
                except Exception as e:
                    print(f"Alerte : Impossible de lire {img_path}, erreur: {e}")
                    # return un exemple aleatoire
                    return self.__getitem__(random.randint(0, len(self.data)-1))
                    
                # Tokenization du texte (Description riche)
                tokens = self.tokenizer(
                    row["text"],
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
               
                return {
                    "pixel_values": img,
                    "input_ids": tokens.input_ids.squeeze(0),
                    "text": row["text"]
                }
        
        dataset = PictoDataset(
            data=data, 
            images_dir=images_dir, 
            transform=transform, 
            tokenizer=self.tokenizer, 
            max_length=self.tokenizer.model_max_length
        )
        
        return DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=shuffle,
            num_workers=2,           # Optimise la lecture disque
            drop_last=True     # Évite les problèmes de dimension sur le dernier batch
        )



    def compute_loss(self, noise_pred, noise, images, prompts):
        loss_diffusion = F.mse_loss(noise_pred, noise)
        
        if not images:      # Si la liste est vide
            return loss_diffusion, 0.0
        
        judge_scores = []
        for img , prompt in zip(images, prompts):
            score = self.validator.predict_score(img, prompt)
            judge_scores.append(score)
        
        mean_judge_score = np.mean(judge_scores)
        #loss = loss_diffusion * (mean_judge_score) * self.args.lambda_reward
        loss = loss_diffusion * (1 - mean_judge_score) * self.args.lambda_reward
        #loss = loss_diffusion * (1/(mean_judge_score + 1e-6)) * self.args.lambda_reward
        #loss = loss_diffusion
        return loss, mean_judge_score
    

    def generate_images(self, prompts, num_inference_steps=30):
        images = []
        for prompt in prompts:
            with torch.no_grad():
                with torch.autocast("cuda"):
                    image = self.pipe(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5
                    ).images[0]
            images.append(image)
        return images
    
    def compute_kid_score(self, generated_images, real_images):
        self.validator.kid_metric.reset()
        
        self.validator.kid_metric.update(real_images, real=True)
        
        gen_tensors = []
        for img in generated_images:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            img_tensor = (img_tensor / 255)  # img_tensor will be in [0,1]
            gen_tensors.append(img_tensor)
        gen_tensors = torch.cat(gen_tensors, dim=0)
        self.validator.kid_metric.update(gen_tensors, real=False)
        
        kid_mean, kid_std = self.validator.kid_metric.compute()
        return kid_mean.item()

    def validate(self, dataloader, split_name, epoch):
        print(f"\nValidation {split_name} - Epoch {epoch+1}")
        
        total_val_images = len(dataloader.dataset)
        num_kid_samples = min(self.args.kid_samples, total_val_images)

        all_real = []
        all_prompts = []
        
        for batch in dataloader:
            all_real.append(batch["pixel_values"])
            all_prompts.extend(batch["text"])
            if len(all_prompts) >= num_kid_samples:
                break
        
        all_real = torch.cat(all_real, dim=0)[:num_kid_samples].to(self.device)
        prompts_subset = all_prompts[:num_kid_samples]
        
        generated_subset = self.generate_images(prompts_subset)
        
        self.validator.kid_metric.reset()
        real_images = (all_real.detach().cpu() + 1.0) / 2.0 #on les met en [0,1]
        self.validator.kid_metric.update(real_images.to(self.device), real=True)
        
        gen_tensors = []
        for img in generated_subset:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            img_tensor = (img_tensor / 255)
            gen_tensors.append(img_tensor)
        gen_tensors = torch.cat(gen_tensors, dim=0)
        self.validator.kid_metric.update(gen_tensors, real=False)
       
        kid_mean, kid_std = self.validator.kid_metric.compute()
        
        kid_mean = kid_mean.item() if hasattr(kid_mean, 'item') else kid_mean
        kid_std = kid_std.item() if hasattr(kid_std, 'item') else kid_std

        all_judge_scores = []
        for img, prompt in zip(generated_subset, prompts_subset):
             score = self.validator.predict_score(img, prompt)
             all_judge_scores.append(score)

        mean_judge = np.mean(all_judge_scores)
        
        print(f"KID (sur {num_kid_samples} images): {kid_mean:.4f}")
        print(f"Judge Score (sur {len(all_judge_scores)} images): {mean_judge:.3f}")
        
        return {"kid": kid_mean, "judge_score": mean_judge}

    def test(self, dataloader):
        print("\nTEST FINAL")

        total_test_images = len(dataloader.dataset)
        print(f"Test sur {total_test_images} images")

        self.validator.kid_metric.reset()
        all_judge_scores = []
        total_images = 0
        
        for batch in tqdm(dataloader, desc="Test"):
            real_images = batch["pixel_values"].to(self.device)
            prompts = batch["text"]
            
            generated_images = self.generate_images(prompts)

            real_images = (real_images.detach().cpu() + 1.0) / 2.0
            self.validator.kid_metric.update(real_images.to(self.device), real=True)
           
            
            gen_tensors = []
            for img in generated_images:
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                img_tensor = (img_tensor / 255) 
                gen_tensors.append(img_tensor)
            gen_tensors = torch.cat(gen_tensors, dim=0)
            self.validator.kid_metric.update(gen_tensors, real=False)
        
            for img, prompt in zip(generated_images, prompts):
                score = self.validator.predict_score(img, prompt)
                all_judge_scores.append(score)
                total_images += 1
        
        kid_mean, kid_std = self.validator.kid_metric.compute()
        
        kid_mean = kid_mean.item() if hasattr(kid_mean, 'item') else kid_mean
        kid_std = kid_std.item() if hasattr(kid_std, 'item') else kid_std
     
        mean_judge = np.mean(all_judge_scores)
        
        print(f"\nRESULTATS TEST FINAL:")
        print(f"KID (sur {total_images} images): {kid_mean:.4f} (+/- {kid_std:.4f})")
        print(f"Judge Score (sur {total_images} images): {mean_judge:.3f}")
        
        return {"kid": kid_mean, "judge_score": mean_judge}
    
    def _convert_to_serializable(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return obj


    def train(self):
        for epoch in range(self.args.num_epochs):
            self.unet.train()
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for batch in progress_bar:
                pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(self.device)
                prompts = batch["text"]
                
                with torch.no_grad():
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                         (latents.shape[0],), device=self.device)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(input_ids)[0]
                
                noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                images_gen = self.generate_images(prompts) #necessaire si on ajoute le score en loss 
                #images_gen = []
                loss, mean_judge = self.compute_loss(noise_pred, noise, images_gen, prompts)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item(), "judge": f"{mean_judge:.2f}"})
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} - Loss moyenne: {avg_loss:.4f}")
            
            val_metrics = self.validate(self.val_loader, "val", epoch)
            
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_kid": val_metrics["kid"],
                "val_judge_score": val_metrics["judge_score"]
            })
            
            print(f"Resume Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val KID={val_metrics['kid']:.4f}, Val Judge={val_metrics['judge_score']:.3f}")
            
            if (epoch + 1) % 5 == 0:
                os.makedirs(self.args.output_dir, exist_ok=True)
                self.unet.save_pretrained(f"{self.args.output_dir}/checkpoint-{epoch+1}")
                with open(f"{self.args.output_dir}/history.json", "w") as f:
                    json.dump(self.history, f, indent=2)
        
        print("\nTEST FINAL")
        test_metrics = self.test(self.test_loader)
        
        self.history.append({
            "epoch": "test",
            "test_kid": test_metrics["kid"],
            "test_judge_score": test_metrics["judge_score"]
        })
        
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.unet.save_pretrained(f"{self.args.output_dir}/final")
        
        with open(f"{self.args.output_dir}/history.json", "w") as f:
            json.dump(self.history, f, indent=2, default=self._convert_to_serializable)

        print(f"\nEntraînement termine!")
        print(f"Test KID: {test_metrics['kid']:.4f}")
        print(f"Test Judge Score: {test_metrics['judge_score']:.3f}")
        print(f"Historique sauvegarde dans {self.args.output_dir}/history.json")


def main():
    args = parse_args()
    trainer = LoRATrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
