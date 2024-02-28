from config.config_paths import RESULTS, MODELS
import torch
import os
from transformers import TrOCRProcessor, RobertaTokenizer
from src.models.charbert.modeling.configuration_roberta import RobertaConfig
import json

# CharBERT Dataset

tokenizer = RobertaTokenizer.from_pretrained("roberta-base",
                                                do_lower_case=False,
                                                cache_dir=None)
class Args:
    def __init__(self):
        self.model_name_or_path = "roberta-base"
        self.tokenizer_name = "roberta-base"
        self.do_lower_case = False
        self.config_name = "roberta-base"
        self.char_vocab = "./src/models/charbert/data/dict/roberta_char_vocab"
        self.term_vocab = "./models/charbert/vocab.json"
        self.output_debug = False
        self.block_size = tokenizer.max_len_single_sentence
        self.train_data_file = "./src/models/charbert/train_data.txt"
        self.model_type = "roberta"
        self.seed = 42
        self.char_maxlen_for_word = 6
        self.tokenizer = tokenizer
        self.model_name_or_path = "roberta-base"

charbert_dataset_args = Args()

# Path to your config.json file
config_file_path = os.path.join(".", "models", "charbert", "config.json")

# Load the configuration from the json file
with open(config_file_path, "r") as config_file:
    config_dict = json.load(config_file)

charbert_config = RobertaConfig.from_dict(config_dict)

# General settings
general = {
    "batch_size": 8,
    "max_epochs": 120
}

model_config = {
    "roberta_config": "imvladikon/charbert-roberta-wiki",
    "trocr_config": "microsoft/trocr-large-handwritten",
    "charbert_config": RobertaConfig.from_dict(config_dict),
    "prefix": "charbert."
}

model = {
    "processor": TrOCRProcessor.from_pretrained(model_config["trocr_config"]),
    "max_target_length": 512,
    "trocr_bypass": ["decoder.model.decoder.embed_tokens.weight"],
    "charbert_bypass": ["charbert.embeddings.word_embeddings.weight"]
}

# Configuration for trocr evaluation
trocr = {
    "experiment1": {
        "optimizer_keys": {
            "lr": 1e-7, # leaning rate
            "weight_decay": 1e-7  # leaning rate
        },
        "data_loader_keys": {
            "batch_size": general["batch_size"],
            "processor": model["processor"],
            "max_target_length": model["max_target_length"]
            },
        "training_keys": {
            "language": "English",
            "log_file_path": os.path.join(RESULTS, "trocr", "log_trocr_exp1_train.txt"),
            "model_path": os.path.join(MODELS, "trocr", "experiment1.pth"),
            "max_epochs": general['max_epochs'],
            "processor": model["processor"],
            "config_name": "TrOCR model",
            "text_output": True
        },
        "testing_keys": {
            "language": "English",
            "log_file_path": os.path.join(RESULTS, "trocr", "log_trocr_exp1_test.txt"),
            "processor": model["processor"],
            "config_name": "TrOCR model",
            "text_output": True
        }
    }
}

# Configuration for trocr_charbert experiment
trocr_charbert = {
    "experiment1": {
        "optimizer_keys": {
            "lr": 1e-5, # leaning rate
            "weight_decay": 1e-5
        },
        "data_loader_keys": {
            "batch_size": general["batch_size"],
            "processor": model["processor"],
            "max_target_length": model["max_target_length"]
        },
        "training_keys": {
            "language": "GW",
            "log_file_path": os.path.join(RESULTS, "trocr_charbert", "log_trocr_charbert_exp1_train.txt"),
            "model_path": os.path.join(MODELS, "trocr_charbert", "experiment1.pth"),
            "max_epochs": general['max_epochs'],
            "processor": model["processor"],
            "config_name": "TrOCR-CharBERT model",
            "text_output": True
        },
        "testing_keys": {
            "language": "English",
            "log_file_path": os.path.join(RESULTS, "trocr_charbert", "log_trocr_exp1_test.txt"),
            "processor": model["processor"],
            "config_name": "TrOCR-CharBERT model",
            "text_output": True
        }
    }
}

# Configuration for seq_models experiment
seq_models = {
    "experiment1": {
        "optimizer_keys": {
            "lr": 1e-5, # leaning rate
            "weight_decay": 1e-5
        },
        "data_loader_keys": {
            "batch_size": general["batch_size"],
            "processor": model["processor"],
            "max_target_length": model["max_target_length"]
        },
        "testing_keys": {
            "language": "English",
            "log_file_path": os.path.join(RESULTS, "seq_models", "log_seq_models_exp1_test.txt"),
            "processor": model["processor"],
            "config_name": "Sequential model",
            "text_output": True
        }
    }
}