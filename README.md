# SEMA-PICTO
# A Multimodal Framework for Semantic Alignment and Controlled Pictogram Generation

# SEMA-PICTO

A Multimodal Framework for Semantic Alignment and Controlled Pictogram Generation

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
│   ├── benchmark.py
│   ├── benchmark.sh
│   ├── dataset.py
│   ├── metrics.py
│   ├── models.py
│   ├── NER.ipynb
│   └── NamedEntityRecognition.ipynb
├── Validation_picto/                # Validation and judging scripts
│   ├── ArasaacValidator.py
│   ├── dataset_judge.csv
│   ├── normalization_stats.csv
│   ├── prepare_judge_dataset.sh
│   ├── test_judge.sh
│   ├── train_judge.sh
│   └── arasaac_judge_final.pth
├── fine_tune_lora/                  # LoRA fine-tuning pipeline
│   ├── dataset_picto/
│   ├── lora_picto_model/
│   ├── inference_lora.sh
│   ├── lancer_lora.sh
│   ├── train_lora.py
│   ├── train_lora.sh
│   ├── train_lora_opt.py
│   ├── train_lora_opt.sh
│   ├── train_text_to_image_lora.py
│   └── test_text_to_image_lora.py
├── inference_lora/                  # Inference experiments and checkpoints
├── success_cases/                   # Examples of successful generated pictograms
├── faillures_cases/                 # Examples of failure cases and problematic outputs
├── README.md
├── .gitignore
└── LICENSE                         # if present in your local clone
