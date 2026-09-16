# SEMA-PICTO: A Multimodal Framework for Semantic Alignment and Controlled Pictogram Generation
SEMA-PICTO is a research-oriented multimodal pipeline for generating pictograms from semantic concepts and textual descriptions. The project combines language understanding, semantic alignment, and controlled image generation to produce pictograms that are both visually consistent and semantically relevant.

This repository includes training scripts, inference pipelines, validation tools, benchmark experiments, and examples of successful and failed generations.

## Overview

The core idea behind SEMA-PICTO is to bridge the gap between:
- natural language descriptions,
- semantic concepts and entity understanding,
- controlled generation of pictogram-like visual content.

The project is designed around:
- semantic-aware text interpretation,
- controlled visual generation with LoRA-based fine-tuning,
- evaluation and validation of generated pictograms,
- benchmarking and qualitative assessment.

## Key Features

- Multimodal generation pipeline for pictogram synthesis
- Semantic alignment of textual prompts and visual concepts
- LoRA fine-tuning for Stable Diffusion-based generation
- Inference scripts for generating pictograms from prompts
- Validation framework for evaluating pictogram quality
- Benchmarking support for Named Entity Recognition (NER) tasks
- Examples of successful and failure cases for qualitative analysis

## Repository Structure

```text
SEMA-PICTO/
├── Article_SEMA_Pipeline.pdf        # Research article / pipeline description
├── NER-Benchmark/                   # Benchmarking experiments for named entity recognition
├── Validation_picto/                # Validation and judging scripts
├── fine_tune_lora/                  # LoRA fine-tuning pipeline
├── inference_lora/                  # Inference experiments and checkpoints
├── success_cases/                   # Examples of successful generated pictograms
├── faillures_cases/                 # Examples of failure cases and problematic outputs
├── README.md
├── .gitignore
```

## Getting Started
### Prerequisites

This project is based on modern deep learning tooling and expects a Python environment with:

    * Python 3.9+
    * PyTorch
    * Diffusers
    * Transformers
    * Accelerate
    * PEFT / LoRA support
    * CUDA-enabled GPU recommended for training and inference

The repository contains scripts that assume a virtual environment is already activated, for example:
bash

source ../picto/env/bin/activate

Clone the Repository:
git clone https://github.com/bennasser-eng/SEMA-PICTO.git
cd SEMA-PICTO


### Install Dependencies
Install the required packages for your local setup, including the libraries needed for:
* model loading,
* LoRA fine-tuning,
* image generation,
* validation and dataset processing.

**Typical dependencies include:**
pip install torch torchvision diffusers transformers accelerate peft safetensors

If your environment already contains the required packages, you can skip this step and proceed directly to training or inference.
Training

The fine-tuning workflow is implemented in the fine_tune_lora directory.

A typical training launch is defined in:
bash

bash fine_tune_lora/train_lora.sh

This script executes training with a dataset configured through:

python fine_tune_lora.py \
    --train_data_dir
    --val_data_dir
    --test_data_dir
    --output_dir
    --batch_size
    --num_epochs
    --lambda_reward
    --kid_samples

Example configuration from the project:

python train_lora.py \
    --train_data_dir ./dataset_picto/dataset_final/train \
    --val_data_dir ./dataset_picto/dataset_final/val \
    --test_data_dir ./dataset_picto/dataset_final/test \
    --output_dir ./epochs500_rank64_alpha64_lamda5_loss_avec_-score \
    --batch_size 16 \
    --num_epochs 500 \
    --lambda_reward 5.0 \
    --kid_samples 200



### Inference
Inference scripts are available in the fine_tune_lora and inference_lora folders.
A representative example can be launched with:
**bash fine_tune_lora/inference_lora.sh**

This script loads a pretrained Stable Diffusion model, applies a LoRA adapter, and generates multiple pictogram-like outputs from input prompts.

The generated images are saved under an output directory such as:
**fine_tune_lora/outputs_inference/**
or within folders created under inference_lora/.


### Validation and Evaluation
The project includes a validation pipeline for pictogram quality assessment.
The Validation_picto/ directory contains scripts and datasets for evaluating generated outputs, including:
* dataset preparation,
* judge model training,
* validation checks,
* normalization statistics,
* dataset CSV files.
**Important files include:**
_ ArasaacValidator.py
_ prepare_judge_dataset.sh
_ train_judge.sh
_ test_judge.sh



### NER Benchmark
The NER-Benchmark/ directory contains benchmark scripts and notebooks for named entity recognition experiments, including:
* benchmark.py
* dataset.py
* metrics.py
* models.py
* Jupyter notebooks for exploratory analysis and benchmark execution

This part of the repository supports semantic analysis of text and can be used to evaluate concept extraction and entity-related tasks relevant to pictogram generation.



### Examples and Qualitative Results
The repository contains visual examples that illustrate the model’s outputs:
* success_cases/        — examples of successful generations
* faillures_cases/      — examples of failure cases and limitations


These folders are useful for qualitative inspection, debugging, and understanding the model’s strengths and weaknesses across different prompts.

### Research Context
This project is associated with a research paper: **Article_SEMA_Pipeline.pdf**.
The repository is intended for research, experimentation, and reproducibility, especially in the context of multimodal generation and semantic alignment.
