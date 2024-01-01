# import config
# import logging

# # Configure logging
# config.configure_logging()

# logger = logging.getLogger(__name__)

from src.train.train_trocr_charbert import train_charbert_trocr
import config.model_config as cfg
from src.processor.data_loader import get_data_loader
import numpy as np

def main(experiment_version, fine_tuned):
    data_loader_keys = cfg.trocr_charbert[experiment_version]["data_loader_keys"]
    gw_data_loaders, iam_data_loaders, bullinger_data_loaders, icfhr_data_loaders = get_data_loader(**data_loader_keys)
    wer_list, cer_list = [], []
    for cv in ["cv1", "cv2", "cv3", "cv4"]:
        train_loader = gw_data_loaders[cv]["train"]
        val_loader = gw_data_loaders[cv]["val"]
        val_wer_score, val_cer_score = train_charbert_trocr(experiment_version=experiment_version, train_loader=train_loader, val_loader=val_loader, fine_tuned=fine_tuned)
        wer_list.append(val_wer_score)
        cer_list.append(val_cer_score)
    wer = np.mean(wer_list)
    cer = np.mean(cer_list)
    print("WER: ", wer)
    print("CER: ", cer)
    return