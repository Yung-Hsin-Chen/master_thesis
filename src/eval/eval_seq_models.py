from src.models.charbert_model import initialise_charbert_model
import os
import time
import logging
from config.config import configure_logging
# from src.models.charbert_trocr_model import CharBERTrOCRModel
from src.processor.data_loader import get_data_loader
from src.utils.train import train
from src.utils.eval import eval
import torch.optim as optim
from argparse import Namespace
from src.models.trocr_model import initialise_trocr_model
import config.model_config as cfg
import torch.nn as nn
import torch

batch_size = cfg.general["batch_size"]
max_epochs = cfg.general["max_epochs"]
max_target_length = cfg.model["max_target_length"]

def eval_seq_models(experiment_version: str, test_loader, device, data, fine_tuned=None):
    model = initialise_charbert_model(experiment_version=fine_tuned)
    testing_keys = cfg.seq_models[experiment_version]["testing_keys"]
    testing_keys["test_loader"] = test_loader
    testing_keys["device"] = device
    testing_keys["model_name"] = "seq_models"
    testing_keys["language"] = "GW" if data=="gw" else "IAM"
    wer, cer = eval(model, **testing_keys)
    return wer, cer

