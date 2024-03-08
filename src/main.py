# import config
# import logging

# # Configure logging
# config.configure_logging()

# logger = logging.getLogger(__name__)

from src.train.train_trocr_charbert import train_trocr_charbert
from src.train.train_trocr_charbert_small import train_trocr_charbert_small
from src.train.train_trocr import train_trocr
# from src.eval.eval_trocr_charbert import eval_trocr_charbert
from src.eval.eval_trocr import eval_trocr
from src.eval.eval_seq_models import eval_seq_models
import config.model_config as cfg
from src.processor.data_loader import get_data_loader
import numpy as np
import os
from src.utils.helpers import set_gpu, set_device
import torch

model_dict = {"train": {"trocr": train_trocr, "trocr_charbert": train_trocr_charbert, "trocr_charbert_small": train_trocr_charbert_small},
            "eval": {"trocr": eval_trocr, "seq_models": eval_seq_models}}

def main(**kwargs):
    experiment_version = kwargs["experiment_version"]
    gpu = kwargs["gpu"]
    fine_tuned = kwargs.get("fine_tuned", None)
    mode = kwargs["mode"]
    freeze_mode = kwargs["freeze_mode"]
    layers = kwargs["layers"]
    model = kwargs["model"]
    data = kwargs["data"]

    if gpu:
        # gpu = "1,2"
        set_gpu(gpu)
        device = set_device()
        # device = torch.device("cuda:0")
    else:
        device = None
    data_loader_keys = cfg.trocr_charbert[experiment_version]["data_loader_keys"]
    if mode=="eval" and model=="trocr":
        data_loader_keys["batch_size"] = 1
    # print(data_loader_keys)
    gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(**data_loader_keys)
    data_dict = {"gw": gw_data_loaders, "iam": iam_data_loaders, "bullinger": bullinger_data_loaders, "icfhr": icfhr_data_loaders}
    data_loaders = data_dict[data]
    wer_list, cer_list = [], []
    # for cv in ["cv1", "cv2", "cv3", "cv4"
    if data=="gw":
        # for cv in ["cv1"]:
        for cv in ["cv1", "cv2", "cv3", "cv4"]:
            if mode=="train":
                train_loader = data_loaders[cv]["train"]
                val_loader = data_loaders[cv]["val"]
                test_loader = data_loaders[cv]["test"]
                arguments = {"experiment_version": experiment_version, "train_loader": train_loader, 
                            "val_loader": val_loader, "test_loader": test_loader, "device": device, 
                            "freeze_mode": freeze_mode, "layers": layers, "data": data, "model_name": model,
                            "fine_tuned": fine_tuned}
                val_wer_score, val_cer_score = model_dict[mode][model](**arguments)
            if mode=="eval":
                test_loader = data_loaders[cv]["test"]
                arguments = {"experiment_version": experiment_version, "test_loader": test_loader, 
                            "device": device, "data": data, "model_name": model, "fine_tuned": fine_tuned}
                val_wer_score, val_cer_score = model_dict[mode][model](**arguments)
            wer_list.append(val_wer_score)
            cer_list.append(val_cer_score)
        wer = np.mean(wer_list)
        cer = np.mean(cer_list)
        print("WER: ", wer)
        print("CER: ", cer)
    else:
        if mode=="train":
            train_loader = data_loaders["cv1"]["train"]
            val_loader = data_loaders["cv1"]["val"]
            test_loader = data_loaders["cv1"]["test"]
            arguments = {"experiment_version": experiment_version, "train_loader": train_loader, 
                            "val_loader": val_loader, "test_loader": test_loader, "device": device, 
                            "freeze_mode": freeze_mode, "layers": layers, "data": data, "model_name": model,
                            "fine_tuned": fine_tuned}
            val_wer_score, val_cer_score = model_dict[mode][model](**arguments)
        if mode=="eval":
            test_loader = data_loaders["cv1"]["test"]
            arguments = {"experiment_version": experiment_version, "test_loader": test_loader, 
                            "device": device, "data": data, "model_name": model, "fine_tuned": fine_tuned}
            val_wer_score, val_cer_score = model_dict[mode][model](**arguments)
        print("WER: ", val_wer_score)
        print("CER: ", val_cer_score)
    return

if __name__=="__main__":
    keys = {
        "experiment_version": "experiment1",
        "gpu": "0",
        "fine_tuned": None,
        "mode": "train", # eval/train
        "freeze_mode": "freeze", # freeze/not_freeze
        # "layers": ["encoder.pooler", "encoder.layernorm", "encoder.encoder.layer.23"],
        "layers": ["encoder", "decoder"],
        "model": "trocr_charbert", # trocr/trocr_charbert/trocr_charbert_small
        "data": "iam" # gw/iam
    }
    main(**keys)
