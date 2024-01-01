import torch
from transformers import RobertaModel, RobertaConfig
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
import torch.nn as nn
from src.utils.helpers import DummyLayer
import logging
from config.config import configure_logging
import config.model_config as cfg
import torch.nn.init as init
from config.config_paths import MODELS

CHARBERT_CONFIG = cfg.model_config["charbert_config"]
TROCR_CONFIG = cfg.model_config["trocr_config"]
MAX_TARGET_LENGTH = cfg.model["max_target_length"]
bypass_param = [cfg.model["trocr_bypass"], cfg.model["charbert_bypass"]]
prefix = cfg.model_config["prefix"]

class CharBERTRoberta(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.used_layers = []
        # Replace all layers in self.embeddings with DummyLayer
        self.embeddings.word_embeddings = DummyLayer()

        self._register_forward_hook()

    def freeze_except(self, layers_to_not_freeze):
        """
        Freeze all layers except the specified ones in the CharBERTRoberta model.

        Args:
            layers_to_not_freeze (list of str): Names or types of layers not to freeze.
        """
        for name, param in self.named_parameters():
            if not any(layer in name for layer in layers_to_not_freeze):
                param.requires_grad = False
            else:
                print(f"Layer trainable: {name}")

    def forward(self, embeddings, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        # Bypass the embedding layer
        # embeddings are assumed to be precomputed and passed directly

        self.used_layers = []

        outputs = super().forward(inputs_embeds=embeddings, attention_mask=attention_mask)

        return outputs

        # return (sequence_output, pooled_output) + encoder_outputs[1:]

    def _register_forward_hook(self):
        # Register a forward hook on each module
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        # This hook will add the layer name to the list if the layer is used
        def hook(module, input, output):
            if name not in self.used_layers:
                self.used_layers.append(name)
        return hook

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights and biases
        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def init_weights(self):
        # Initialize weights using Xavier (Glorot) initialization
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
    
class TrOCRCharBERTModel(VisionEncoderDecoderModel):
    def __init__(self, config, charbert_model_name, max_target_length):
        super().__init__(config)

        self.used_layers = []
        
        self.charbert = CharBERTRoberta(charbert_model_name)

        # Three layer FFNNs to generate CharBERT's arguments
        self.ffnn_embeddings = FFNN(input_dim=1024, hidden_dim=2048, output_dim=768)
        # Generate attention mask from embeddings
        self.ffnn_attention_mask = FFNN(input_dim=1024, hidden_dim=2048, output_dim=1)

        # Three layer FFNN to process CharBERT's output
        self.ffnn_charbert_output = FFNN(input_dim=768, hidden_dim=2048, output_dim=1024)

        self.decoder.model.decoder.embed_tokens = DummyLayer()
        self.max_target_length = max_target_length

        self._float_tensor = torch.FloatTensor()

        self._register_forward_hook()

    def freeze_except(self, layers_to_not_freeze):
        """
        Freeze all layers except the specified ones in the TrOCRCharBERTModel.

        Args:
            layers_to_not_freeze (list of str): Names or types of layers not to freeze.
        """
        for name, param in self.named_parameters():
            # Check if the layer's name starts with 'ffnn' or is in layers_to_not_freeze
            if name.startswith("ffnn") or any(layer in name for layer in layers_to_not_freeze):
                param.requires_grad = True
                print(f"Layer trainable: {name}")
            else:
                param.requires_grad = False


    def forward(self,
        pixel_values=None,
        decoder_input_embeddings=None, 
        decoder_attention_mask=None,
        encoder_outputs=None,
        labels=None,
        **kwargs):

        self.used_layers = []

        # Get the encoder outputs
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        # """
        # Process decoder input IDs with FFNNs to generate CharBERT's arguments
        # print("decoder input embeddings type: ", decoder_input_embeddings.dtype)
        # print("decoder input embeddings: ", decoder_input_embeddings.size())
        embeddings = self.ffnn_embeddings(decoder_input_embeddings)
        attention_logits = self.ffnn_attention_mask(decoder_input_embeddings)
        attention_mask = torch.sigmoid(attention_logits)
        # print("charbert embeddings type: ", embeddings.dtype)
        # print("charbert embeddings: ", embeddings.size())
        # print("charbert attention mask type: ", attention_mask.dtype)
        # print("charbert attention mask: ", attention_mask.size())

        # Process the input through CharBERT
        charbert_output = self.charbert.forward(embeddings, attention_mask)
        # print("charbert output type: ", charbert_output[0].dtype)
        # print("charbert output: ", charbert_output[0].size())

        # Process CharBERT's output with another three-layer FFNN
        transformed_decoder_input = self.ffnn_charbert_output(charbert_output[0])
        # print("transformed decoder input type: ", transformed_decoder_input.dtype)
        # print("transformed decoder input: ", transformed_decoder_input.size())

        decoder_outputs = self.decoder(
                inputs_embeds=transformed_decoder_input,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                labels=labels,
                **kwargs,
            )
        return decoder_outputs
    
    def _register_forward_hook(self):
        # Register a forward hook on each module
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        # This hook will add the layer name to the list if the layer is used
        def hook(module, input, output):
            if name not in self.used_layers:
                self.used_layers.append(name)
        return hook

def get_pretrain_param():
    """
    Loads and merges pre-trained parameters from standalone TrOCR and CharBERT models into a composite TrOCRCharBERTModel.

    This function initializes two standalone models: TrOCR and CharBERT, with their respective pre-trained configurations and state dictionaries. It then creates a composite TrOCRCharBERTModel, updates its state dictionary with the parameters from the standalone models, and returns the updated composite model along with its updated state dictionary.

    The CharBERT model's state dictionary keys are prefixed to match the naming convention in the composite model. The function assumes that the global variables `TROCR_CONFIG`, `CHARBERT_CONFIG`, `MAX_TARGET_LENGTH`, and `bypass_param` are predefined and accessible.

    Returns:
        Tuple[TrOCRCharBERTModel, dict]: A tuple containing:
            - The composite TrOCRCharBERTModel updated with pre-trained parameters.
            - The updated state dictionary of the composite model. 

    Notes:
        - This function requires the `transformers` library and access to the Hugging Face model repository.
        - Ensure that `TROCR_CONFIG`, `CHARBERT_CONFIG`, `MAX_TARGET_LENGTH`, and `bypass_param` are correctly defined and accessible.
        - The function logs a debug message for each parameter in the pre-trained state dictionaries that isn't found in the composite model's state dictionary, except those listed in `bypass_param`.
    """
    # Standalone TrOCR model and state dict
    trocr_config = VisionEncoderDecoderConfig.from_pretrained(TROCR_CONFIG)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_CONFIG)
    trocr_state_dict = trocr_model.state_dict()
    # Standalone CharBERT model and state dict(adjusted according to the prefix)
    charbert_config = RobertaConfig.from_pretrained(CHARBERT_CONFIG)
    charbert_model = RobertaModel.from_pretrained(CHARBERT_CONFIG)
    charbert_state_dict = charbert_model.state_dict()
    charbert_state_dict = {prefix + k: v for k, v in charbert_state_dict.items()}
    # Composite model and state dict
    model = TrOCRCharBERTModel(config=trocr_config, charbert_model_name=charbert_config, max_target_length=MAX_TARGET_LENGTH)
    state_dict = model.state_dict()
    # Load pretrained parameters
    trocr_state_dict.update(charbert_state_dict)
    for name, param in trocr_state_dict.items():
        if (name in state_dict) and (name not in bypass_param):
            state_dict[name].copy_(param)
        elif name not in bypass_param:
            logging.debug(f"Parameter {name} not found in the model.")

    return model, state_dict, trocr_state_dict

def get_fune_tuned_param(experiment_version):
    # Standalone TrOCR model and state dict
    trocr_config = VisionEncoderDecoderConfig.from_pretrained(TROCR_CONFIG)
    # Standalone CharBERT model and state dict(adjusted according to the prefix)
    charbert_config = RobertaConfig.from_pretrained(CHARBERT_CONFIG)
    # Composite model and state dict
    model = TrOCRCharBERTModel(config=trocr_config, charbert_model_name=charbert_config, max_target_length=MAX_TARGET_LENGTH)
    model_path = os.path.join(MODELS, "trocr_charbert", experiment_version)
    fine_tuned_weights = torch.load(model_path)
    model.load_state_dict(fine_tuned_weights)
    return model

def initialise_trocr_charbert_model(experiment_version=None):
    """
    Initializes and verifies a composite model with pre-trained parameters from TrOCR and CharBERT models.

    This function calls `get_pretrain_param` to obtain a composite TrOCRCharBERTModel pre-loaded with parameters from standalone TrOCR and CharBERT models. It then verifies that each parameter of the composite model matches its counterpart in the respective standalone model's state dictionary. If any mismatches are found, it raises a ValueError.

    Returns:
        TrOCRCharBERTModel: The verified composite model with pre-trained parameters loaded.

    Raises:
        ValueError: If any parameter from the composite model doesn't match the corresponding parameter in the standalone model's state dictionary.

    Notes:
        - The function assumes that the `trocr_state_dict` and `charbert_state_dict` are accessible and contain the state dictionaries of the standalone TrOCR and CharBERT models, respectively.
        - The `prefix` variable is expected to denote the naming convention used in the composite model for CharBERT parameters.
        - The function uses logging to debug issues where parameter mismatches occur.
        - It's crucial that `get_pretrain_param` function is defined and accessible, as it's responsible for initializing the composite model and loading the pre-trained parameters.
    """
    if experiment_version:
        model = get_fune_tuned_param(experiment_version)
    else:
        model, state_dict, trocr_state_dict = get_pretrain_param()
        model.load_state_dict(state_dict, strict=True)
    # Check if the parameters are loaded correctly
    for name, param in model.named_parameters():
        # Set the print options to increase threshold
        torch.set_printoptions(threshold=10_000)  # Set a high threshold value

        if name=="encoder.pooler.dense.bias":
            print(trocr_state_dict.get(name, None))
            print(param)
        if name=="encoder.pooler.dense.weight":
            print(trocr_state_dict.get(name, None))
            print(param)
        # If the parameter is not part of "ffnn" and does not match the state dict, raise an error
        if name[:4] != "ffnn" and not torch.equal(trocr_state_dict.get(name, None), param):
            message = f"Parameter {name} does not match."
            logging.debug(message)
            raise ValueError(message)
    logging.info("All pretrained parameters matched.")
    
    return model
