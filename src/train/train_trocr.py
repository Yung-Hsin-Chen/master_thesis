import os
import time
import logging
from config.config import configure_logging
# from src.models.charbert_trocr_model import CharBERTrOCRModel
from src.processor.data_loader import get_data_loader
from src.utils.train import train
import torch.optim as optim
from argparse import Namespace
from src.models.trocr_model import initialise_trocr_model
import config.model_config as cfg
import torch.nn as nn
import torch

batch_size = cfg.general["batch_size"]
max_epochs = cfg.general["max_epochs"]
max_target_length = cfg.model["max_target_length"]

def freeze(model, mode, layers):
    """
    Freeze all layers except the specified ones in the TrOCRCharBERTModel.

    Args:
        layers_to_not_freeze (list of str): Names or types of layers not to freeze.
    """
    mode_dict = {"freeze": {"none": True, "layers": False, "not_layers": True},
                "not_freeze": {"none": False, "layers": True, "not_layers": False}}
    if layers==[]:
        for name, param in model.named_parameters():
            # param.requires_grad = False
            param.requires_grad = mode_dict[mode]["none"]
    else:
        for name, param in model.named_parameters():
            for i in layers:
                if name.startswith(i):
                    param.requires_grad = mode_dict[mode]["layers"]
                else:
                    param.requires_grad = mode_dict[mode]["not_layers"]
    return model

# Training
def train_trocr(experiment_version: str, train_loader, val_loader, test_loader, device, freeze_mode, layers, data, model_name, text_file, fine_tuned=None):
    # Define model
    if fine_tuned:
        fine_tuned = experiment_version
    model = initialise_trocr_model(experiment_version=fine_tuned)
    # for name,param in model.named_parameters():
    #     print(param)
    #     break
    # Specify the layers you do not want to freeze (by name or type)

    # Freeze all other layers in both the TrOCR and CharBERT parts of the model
    model = freeze(model, freeze_mode, layers)
    # for name,param in model.named_parameters():
    #     print(param)
    #     break
    # print(model.ffnn_embeddings.fc1.weight.requires_grad)
    # model.charbert.freeze_except(layers_to_not_freeze)
    # Get data loaders
    # data_loader_keys = cfg.trocr[experiment_version]["data_loader_keys"]
    # gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(**data_loader_keys)
    # Train
    training_keys = cfg.trocr[experiment_version]["training_keys"]
    # Add 'optimizer' to the training_keys dictionary
    optimizer_keys = cfg.trocr[experiment_version]["optimizer_keys"]
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
    training_keys["lr"] = cfg.trocr[experiment_version]["optimizer_keys"]["lr"]
    training_keys["weight_decay"] = cfg.trocr[experiment_version]["optimizer_keys"]["weight_decay"]
    training_keys["model_name"] = model_name
    training_keys["text_file"] = text_file
    # with torch.autograd.detect_anomaly():
    val_wer_score, val_cer_score = train(model, freeze_mode, layers, **training_keys)
    return val_wer_score, val_cer_score
