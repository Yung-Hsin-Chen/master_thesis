from config.config_paths import RESULTS, MODELS
import torch
import os
from transformers import TrOCRProcessor

# General settings
general = {
    "batch_size": 2,
    "max_epochs": 3
}

model_config = {
    "charbert_config": "imvladikon/charbert-roberta-wiki",
    "trocr_config": "microsoft/trocr-large-handwritten",
    "prefix": "charbert."
}

model = {
    "processor": TrOCRProcessor.from_pretrained(model_config["trocr_config"]),
    "max_target_length": 512,
    "trocr_bypass": "decoder.model.decoder.embed_tokens.weight",
    "charbert_bypass": "charbert.embeddings.word_embeddings.weight"
}

# Configuration for trocr_charbert experiment
trocr_charbert = {
    "experiment1": {
        "optimizer_keys": {
            "lr": 0.00001  # leaning rate
        },
        "data_loader_keys": {
            "batch_size": general["batch_size"],
            "processor": model["processor"],
            "max_target_length": model["max_target_length"]
        },
        "training_keys": {
            "language": "English",
            "log_file_path": os.path.join(RESULTS, "trocr_charbert", "log_trocr_charbert_exp1.txt"),
            "model_path": os.path.join(MODELS, "trocr_charbert", "experiment1.pth"),
            "max_epochs": general['max_epochs'],
            "processor": model["processor"],
            "config_name": "TrOCR-CharBERT model",
            "text_output": True
        }
    }
}
