from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from dataset import DataSet
from models import NER_models
from metrics import Metrics
import gc
import torch
import os
from transformers import BitsAndBytesConfig #quantisation to avoid killing


"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)"""

token = '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'

model_names = ["camembert-base" , "PantagrueLLM/text-base-wiki", "xlm-roberta-base"]

all_results = []
for name in model_names:
    print(f"\n--- Évaluation de {name} ---")
   
    torch.cuda.empty_cache()
    gc.collect()

    model = NER_models(name,
                       num_labels=5, 
                       token=token
                       )
    
    tokenizer = AutoTokenizer.from_pretrained(name, 
                                              trust_remote_code=True,
                                              token=token
                                              )
    
    df_train, df_val, df_test = DataSet.prepare_data()
    train_dataset, val_dataset, test_dataset = DataSet.prepare_datasets(df_train, df_val, df_test, tokenizer)
    
    # On crée le collator spécifique au tokenizer (pour le padding)
    data_collator = DataSet.collator(tokenizer)

    # On change l'output_dir pour chaque modèle sinon ils s'écrasent
    training_args = TrainingArguments(
        output_dir=f"./results_{name.replace('/', '_')}", 
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, # Compense le petit batch
        fp16=True,                     # la précision demi-réelle
        num_train_epochs=15,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=Metrics.compute_metrics
    )

    trainer.train()
    
    # Extraction des résultats pour la comparaison
    eval_results = trainer.evaluate(test_dataset)
    all_results.append({
        "Model": name,
        "F1": eval_results["eval_f1"],
        "Precision": eval_results["eval_precision"],
        "Recall": eval_results["eval_recall"],
        "Accuracy": eval_results["eval_accuracy"]
    })

    del model
    del trainer
