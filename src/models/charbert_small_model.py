from src.models.charbert.modeling.modeling_roberta import RobertaModel, RobertaForMaskedLM
from src.models.charbert.modeling.configuration_roberta import RobertaConfig
from transformers import RobertaTokenizer
import config.model_config as cfg
import torch
charbert_args = cfg.charbert_dataset_args
import os
from config.config_paths import MODELS
from src.models.adapted_charbert import AdaptedRobertaModel

def load_model():
    charbert_config = cfg.charbert_config
    model = RobertaForMaskedLM(config=charbert_config)
    # model = AdaptedRobertaModel.from_pretrained(charbert_args.model_name_or_path,
    #                                     from_tf=False,
    #                                     config=cfg.charbert_config,
    #                                     cache_dir=None)
    return model

def get_fine_tuned_param(experiment_version):
    prefix = "roberta."
    model = load_model()
    model_path = os.path.join(MODELS, "charbert_small", "pytorch_model.bin")
    fine_tuned_weights = torch.load(model_path)
    # print("\nPRETRAINED MODEL PARAM")
    # print(fine_tuned_weights.keys())
    # print("\nMODEL PARAM")
    # for name, param in model.named_parameters():
    #     print(name)
    fine_tuned_weights = {(k[8:] if k.startswith("roberta.") else k): v for k, v in fine_tuned_weights.items()}
    # print("\nPRETRAINED MODEL PARAM")
    # print(fine_tuned_weights.keys())
    # print("STATE DICT: \n", list(fine_tuned_weights.keys()), "\n\n")
    model.load_state_dict(fine_tuned_weights, strict=False)
    # for name, param in model.named_parameters():
    #     if name not in list(fine_tuned_weights.keys()):
    #         print("Parameter", name, " not in model.")
    #     pre_trained_param = fine_tuned_weights.get(name, None)
    #     pre_trained_param = pre_trained_param.to("cpu")
    #     if not torch.equal(pre_trained_param, param):
    #         message = f"Parameter {name} does not match."
    #         raise ValueError(message)
    # print("All pretrained parameters matched.")
    return model

def initialise_charbert_small_model(experiment_version=None):
    if not experiment_version:
        model = load_model()
        # print(model.state_dict().keys())
    if experiment_version:
        # print("here")
        model = get_fine_tuned_param(experiment_version)
    # else:
    #     model, state_dict, charbert_state_dict = get_pretrain_param()
    #     model.load_state_dict(state_dict, strict=False)
    #     # Check if the parameters are loaded correctly
    #     for name, param in model.named_parameters():
    #         # Set the print options to increase threshold
    #         # torch.set_printoptions(threshold=10_000)  # Set a high threshold value
    #         # If the parameter is not part of "ffnn" and does not match the state dict, raise an error
    #         # if name not in bypass_param:
    #             # print(name)
    #         if not torch.equal(charbert_state_dict.get(name, None), param):
    #             message = f"Parameter {name} does not match."
    #             raise ValueError(message)
    #     print("All pretrained parameters matched.")
    return model
