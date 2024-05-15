from src.models.charbert.modeling.modeling_roberta import RobertaModel, RobertaForMaskedLM
from src.models.charbert.modeling.configuration_roberta import RobertaConfig
from transformers import RobertaTokenizer
import config.model_config as cfg
import torch
charbert_args = cfg.charbert_dataset_args
import os
from config.config_paths import MODELS

def load_model():
    charbert_config = cfg.charbert_config
    model = RobertaModel(config=charbert_config)
    return model

def get_pretrain_param():
    # Standalone CharBERT model and state dict(adjusted according to the prefix)
    charbert_model = RobertaModel.from_pretrained(charbert_args.model_name_or_path,
                                        from_tf=False,
                                        config=cfg.charbert_config,
                                        cache_dir=None)
    charbert_state_dict = charbert_model.state_dict()
    # charbert_state_dict = {prefix + k: v for k, v in charbert_state_dict.items()}
    # Composite model and state dict
    model = load_model()
    state_dict = model.state_dict()
    # Load pretrained parameters
    for name, param in state_dict.items():
        if name in charbert_state_dict:
            state_dict[name].copy_(charbert_state_dict[name])

    return model, state_dict, charbert_state_dict

def get_fine_tuned_param(experiment_version):
    prefix = "roberta."
    model = load_model()
    model_path = os.path.join(MODELS, "charbert", "pytorch_model.bin")
    fine_tuned_weights = torch.load(model_path)
    fine_tuned_weights = {prefix + k: v for k, v in fine_tuned_weights.items()}
    # print("STATE DICT: \n", list(fine_tuned_weights.keys()), "\n\n")
    model.load_state_dict(fine_tuned_weights)
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

def initialise_charbert_model(experiment_version=None):
    if experiment_version:
        # print("here")
        model = get_fine_tuned_param(experiment_version)
    else:
        model, state_dict, charbert_state_dict = get_pretrain_param()
        model.load_state_dict(state_dict, strict=False)
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
