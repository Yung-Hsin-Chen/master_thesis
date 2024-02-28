from transformers import VisionEncoderDecoderModel
import torch
import config.model_config as cfg

CHARBERT_CONFIG = cfg.model_config["charbert_config"]
TROCR_CONFIG = cfg.model_config["trocr_config"]
MAX_TARGET_LENGTH = cfg.model["max_target_length"]

def load_model():
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_CONFIG)
    return model
        
def initialise_trocr_model(experiment_version=None):
    return load_model()
