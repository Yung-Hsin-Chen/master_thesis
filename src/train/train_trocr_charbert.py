import os
import time
import logging
from config.config import configure_logging
# from src.models.charbert_trocr_model import CharBERTrOCRModel
from src.processor.data_loader import get_data_loader
from src.utils.train import train
import torch.optim as optim
from argparse import Namespace
from src.models.trocr_charbert_model import initialise_trocr_charbert_model
import config.model_config as cfg
import torch.nn as nn
import torch

batch_size = cfg.general["batch_size"]
max_epochs = cfg.general["max_epochs"]
max_target_length = cfg.model["max_target_length"]

# Training
def train_trocr_charbert(experiment_version: str, train_loader, val_loader, test_loader, device, freeze_mode, layers, data, model_name, fine_tuned=None):
    # Define model
    if fine_tuned:
        fine_tuned = experiment_version
    model = initialise_trocr_charbert_model(experiment_version=fine_tuned)
    # Specify the layers you do not want to freeze (by name or type)

    # Freeze all other layers in both the TrOCR and CharBERT parts of the model
    model.freeze(freeze_mode, layers)
    # print(model.ffnn_embeddings.fc1.weight.requires_grad)
    # model.charbert.freeze_except(layers_to_not_freeze)
    # Get data loaders
    # data_loader_keys = cfg.trocr_charbert[experiment_version]["data_loader_keys"]
    # gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(**data_loader_keys)
    # Train
    training_keys = cfg.trocr_charbert[experiment_version]["training_keys"]
    # Add 'optimizer' to the training_keys dictionary
    optimizer_keys = cfg.trocr_charbert[experiment_version]["optimizer_keys"]
    training_keys["language"] = "GW" if data=="gw" else "IAM"
    training_keys["optimizer"] = optim.Adam(model.parameters(), **optimizer_keys)
    # training_keys["optimizer"] = optim.Adam(model.parameters(), **optimizer_keys)
    # Add 'train_loader' and 'val_loader' to the training_keys dictionary
    # training_keys["train_loader"] = gw_data_loaders["cv1"]["train"]
    training_keys["train_loader"] = train_loader
    # training_keys["val_loader"] = gw_data_loaders["cv1"]["val"]
    training_keys["val_loader"] = val_loader
    training_keys["test_loader"] = test_loader
    training_keys["device"] = device
    training_keys["lr"] = cfg.trocr_charbert[experiment_version]["optimizer_keys"]["lr"]
    training_keys["weight_decay"] = cfg.trocr_charbert[experiment_version]["optimizer_keys"]["weight_decay"]
    training_keys["model_name"] = model_name
    # with torch.autograd.detect_anomaly():
    val_wer_score, val_cer_score = train(model, freeze_mode, layers, **training_keys)
    return val_wer_score, val_cer_score
