from transformers import VisionEncoderDecoderModel
import torch
import config.model_config as cfg

CHARBERT_CONFIG = cfg.model_config["charbert_config"]
TROCR_CONFIG = cfg.model_config["trocr_config"]
MAX_TARGET_LENGTH = cfg.model["max_target_length"]
processor = cfg.model["processor"]

def load_model():
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_CONFIG)
    # # set special tokens used for creating the decoder_input_ids from the labels
    # model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    # model.config.pad_token_id = processor.tokenizer.pad_token_id
    # # make sure vocab size is set correctly
    # model.config.vocab_size = model.config.decoder.vocab_size

    # # set beam search parameters
    # model.config.eos_token_id = processor.tokenizer.sep_token_id
    # model.config.max_length = 64
    # model.config.early_stopping = True
    # model.config.no_repeat_ngram_size = 3
    # model.config.length_penalty = 2.0
    # model.config.num_beams = 4
    return model
        
def initialise_trocr_model(experiment_version=None):
    return load_model()
