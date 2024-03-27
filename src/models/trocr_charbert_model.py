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
import os
from src.models.adapted_charbert import AdaptedRobertaModel
import torch.nn.functional as F

CHARBERT_CONFIG = cfg.model_config["charbert_config"]
TROCR_CONFIG = cfg.model_config["trocr_config"]
MAX_TARGET_LENGTH = cfg.model["max_target_length"]
bypass_param = cfg.model["trocr_bypass"] + cfg.model["charbert_bypass"]
prefix = cfg.model_config["prefix"]
charbert_args = cfg.charbert_dataset_args

class TensorTransform(nn.Module):
    def __init__(self, in_channels_dim, out_channels_dim, 
                        hid_channels_dim1, hid_channels_dim2, 
                        hid_features_dim1, hid_features_dim2,
                        in_features_dim, out_features_dim):
        super(TensorTransform, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv1d(in_channels=in_channels_dim, out_channels=hid_channels_dim1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hid_channels_dim1, out_channels=hid_channels_dim2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hid_channels_dim2, out_channels=out_channels_dim, kernel_size=3, stride=1, padding=1)

        # Activation function for convolution layers
        self.relu = nn.LeakyReLU(0.1)

        # Feed-forward layers
        self.ffnn1 = nn.Linear(in_features=in_features_dim, out_features=hid_features_dim1)
        self.ffnn2 = nn.Linear(in_features=hid_features_dim1, out_features=hid_features_dim2)
        self.ffnn3 = nn.Linear(in_features=hid_features_dim2, out_features=out_features_dim)

        # Batch Normalization layers for stability
        self.batch_norm1 = nn.BatchNorm1d(hid_channels_dim1)
        self.batch_norm2 = nn.BatchNorm1d(hid_channels_dim2)

        self.dropout = nn.Dropout(0.1)

        # Initialize weights and biases
        self.init_weights()

    def forward(self, x):
        # Apply first convolution layer
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        # Apply second convolution layer
        x = self.batch_norm1(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        # Apply third convolution layer
        x = self.batch_norm2(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        # Apply first FFNN layer
        x = self.relu(self.ffnn1(x))
        x = self.dropout(x)
        # Apply second FFNN layer
        x = self.relu(self.ffnn2(x))
        x = self.dropout(x)
        # Apply third FFNN layer
        x = self.relu(self.ffnn3(x))
        x = self.dropout(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
"""
class EmbeddingCombiner(nn.Module):
    def __init__(self, input_channels=1024, intermediate_channels_1=512, intermediate_channels_2=256, output_channels=1, num_models=3, seq_len=512):
        super(EmbeddingCombiner, self).__init__()
        self.num_models = num_models
        # Apply a Conv1d to reduce embedding dimension from 1024 to 1
        # The convolution is applied independently for each model's output
        # self.conv1d = nn.Conv1d(in_channels=input_channels,
        #                         out_channels=output_channels,
        #                         kernel_size=3, stride=1, padding=1)
        self.conv1d_1 = nn.Conv1d(in_channels=input_channels,
                                out_channels=intermediate_channels_1,
                                kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=intermediate_channels_1,
                                out_channels=intermediate_channels_2,
                                kernel_size=3, stride=1, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=intermediate_channels_2,
                                out_channels=output_channels,
                                kernel_size=3, stride=1, padding=1)
        self.seq_len = seq_len
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, embeddings):
        # x shape: (batch_size, num_models, seq_length, embedding_size)
        stacked_embeddings = torch.stack(embeddings).transpose(0, 1)
        batch_size, num_models, seq_length, embedding_size = stacked_embeddings.shape
        # Process each model's output independently
        att_weights = torch.zeros(batch_size, num_models, seq_length, device=stacked_embeddings.device)
        for i in range(self.seq_len):
            # Extract the embeddings for model i
            model_embeddings = stacked_embeddings[:, :, i, :]  # Shape: (batch_size, num_models, embedding_size)
            # Reshape for Conv1d: (batch_size, embedding_size, seq_length)
            model_embeddings = model_embeddings.transpose(1, 2)
            # Apply convolution to reduce embedding dimension
            reduced = self.conv1d_1(model_embeddings)  # Shape: (batch_size, 1, num_models)
            # print("1: ", reduced.size())
            reduced = self.relu(reduced)
            reduced = self.conv1d_2(reduced) 
            reduced = self.relu(reduced)
            # print("2: ", reduced.size())
            reduced = self.conv1d_3(reduced) 
            reduced = self.relu(reduced)
            # print("3: ", reduce d.size())
            # Squeeze and assign to output
            att_weights[:, :, i] = reduced.squeeze(1) # (batch_size, num_models, seq_length)
        att_weights = att_weights.unsqueeze(-1)
        weighted_outputs = stacked_embeddings * att_weights
        combined = weighted_outputs.sum(dim=1)
        return combined
"""

class EmbeddingCombiner(nn.Module):
    def __init__(self, seq_len=512, emb_size=1024, num_models=3):
        super(EmbeddingCombiner, self).__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_models = num_models
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(self.emb_size, 2048),  # First linear layer
            nn.LeakyReLU(0.1),                                        # Non-linear activation
            nn.Linear(2048, 512),                              # Second linear layer
            nn.LeakyReLU(0.1),                                        # Another non-linear activation
            nn.Linear(512, self.num_models),                  # Final layer to output attention weights
            nn.LeakyReLU(0.1)
        )

    def forward(self, embeddings):
        # embeddings: list of tensors of shape (batch_size, seq_len, emb_size)
        # Stack and transpose to (batch_size, num_models, seq_len, emb_size)
        stacked_embeddings = torch.stack(embeddings).transpose(0, 1)
        # att_weights = stacked_embeddings.mean(dim=3) # (8, 3, 512)
        # # att_weights = word_embeddings.unsqueeze(2) # (8, 3, 1)
        # att_weights = att_weights.unsqueeze(-1)
        # weighted_outputs = stacked_embeddings * att_weights
        # combined = weighted_outputs.sum(dim=1)

        # Apply attention network to each word in the sequence
        weights = []
        for i in range(self.seq_len):
            # Extract i-th word embeddings across all models
            word_embeddings = stacked_embeddings[:, :, i, :]  # shape: (batch_size, num_models, emb_size)
            # print("word_embeddings: ", word_embeddings.size()) # (8, 3, 1024)
            # word_embeddings = word_embeddings.mean(dim=2)  # Mean pooling # (8, 3)
            # print("word_embeddings: ", word_embeddings.size())

            # Compute attention weights
            word_embeddings = self.attention_net(word_embeddings)
            word_embeddings = word_embeddings.mean(dim=2)
            # print("word_embeddings: ", word_embeddings.size())
            att_weights = F.softmax(word_embeddings, dim=1)  # shape: (batch_size, num_models)
            # print("att_weights: ", att_weights.size())
            weights.append(att_weights.unsqueeze(2))

        # Concatenate weights for all words
        weights = torch.cat(weights, dim=2)  # shape: (batch_size, num_models, seq_len)

        # Apply weights and sum across models
        combined = (stacked_embeddings * weights.unsqueeze(3)).sum(dim=1)  # shape: (batch_size, seq_len, emb_size)

        return combined

    
class TrOCRCharBERTModel(VisionEncoderDecoderModel):
    def __init__(self, args, config, charbert_config, max_target_length):
        super().__init__(config)

        self.used_layers = []
        # """
        self.charbert = AdaptedRobertaModel.from_pretrained(args.model_name_or_path,
                                        from_tf=False,
                                        config=charbert_config,
                                        cache_dir=None)

        # Three layer FFNNs to generate CharBERT's arguments
        self.transform_char_embeds = TensorTransform(in_channels_dim=512, out_channels_dim=3060,
                                                    hid_channels_dim1=1024, hid_channels_dim2=2048,
                                                    in_features_dim=1024, out_features_dim=256,
                                                    hid_features_dim1=2048, hid_features_dim2=512)
        # Generate attention mask from embeddings
        # self.ffnn_attention_mask = FFNN(input_dim=1024, hidden_dim=2048, output_dim=1)

        # Three layer FFNN to process CharBERT's output
        self.transform_embeds = TensorTransform(in_channels_dim=512, out_channels_dim=510,
                                                    hid_channels_dim1=1024, hid_channels_dim2=512,
                                                    in_features_dim=1024, out_features_dim=768,
                                                    hid_features_dim1=2048, hid_features_dim2=1024)
        self.transform_output = TensorTransform(in_channels_dim=510, out_channels_dim=512,
                                                    hid_channels_dim1=1024, hid_channels_dim2=512,
                                                    in_features_dim=768, out_features_dim=1024,
                                                    hid_features_dim1=2048, hid_features_dim2=1024) 
        self.embedding_combiner = EmbeddingCombiner()    

        self.decoder.model.decoder.embed_tokens = DummyLayer()
        self.max_target_length = max_target_length

        self._float_tensor = torch.FloatTensor()
        # """
        self._register_forward_hook()

    def freeze(self, mode, layers):
        """
        Freeze all layers except the specified ones in the TrOCRCharBERTModel.

        Args:
            layers_to_not_freeze (list of str): Names or types of layers not to freeze.
        """
        mode_dict = {"freeze": {"none": True, "layers": False, "not_layers": True},
                    "not_freeze": {"none": False, "layers": True, "not_layers": False}}
        if layers==[]:
            for name, param in self.named_parameters():
                param.requires_grad = mode_dict[mode]["none"]
        else:
            for name, param in self.named_parameters():
                for i in layers:
                    if name.startswith(i):
                        param.requires_grad = mode_dict[mode]["layers"]
                    else:
                        param.requires_grad = mode_dict[mode]["not_layers"]


    def forward(self,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        encoder_outputs=None,
        labels=None,
        start_ids=None,
        end_ids=None,
        **kwargs):

        # print("Gradient tracking enabled:", torch.is_grad_enabled())

        self.used_layers = []
        # print("pixel values: ", pixel_values)
        # Get the encoder outputs
        encoder_outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
        # """
        # Process decoder input IDs with FFNNs to generate CharBERT's arguments
        # print("decoder input embeddings type: ", decoder_input_embeddings.dtype)
        # print("decoder input embeddings: ", decoder_input_embeddings.size())
        # print("decoder_input_embeddings: ", decoder_input_embeddings)
        # if decoder_inputs_embeds!=None:
        #     print("decoder_input_embeds: ", decoder_inputs_embeds.size())
        # decoder_inputs_embeds = decoder_inputs_embeds.view(2, -1)
        char_input_embeds = self.transform_char_embeds(decoder_inputs_embeds)
        input_embeds = self.transform_embeds(decoder_inputs_embeds)
        # embeddings = embeddings.view(-1, 512, 768)
        # attention_logits = self.ffnn_attention_mask(decoder_inputs_embeds)
        # attention_mask = torch.sigmoid(attention_logits)
        # print("charbert embeddings type: ", embeddings.dtype)
        # print("charbert embeddings: ", embeddings.size())
        # print("charbert attention mask type: ", attention_mask.dtype)
        # print("charbert attention mask: ", attention_mask.size())

        # Process the input through CharBERT
        # print("\nCharBERT embeddings: ", embeddings.size())
        # print("\nCharBERT attention mask: ", attention_mask.size())
        charbert_output = self.charbert(char_input_ids=None, start_ids=start_ids, end_ids=end_ids, 
                                        input_ids=None, char_input_embeds=char_input_embeds, 
                                        inputs_embeds=input_embeds)
        # print("\nCharBERT output: ", charbert_output[0])
        # print("charbert output type: ", charbert_output.last_hidden_state.dtype)
        # print("charbert output: ", charbert_output.last_hidden_state.size())

        # Process CharBERT's output with another three-layer FFNN
        transformed_token_repr = self.transform_output(charbert_output[0])
        transformed_char_repr = self.transform_output(charbert_output[2])
        # transformed_decoder_input = transformed_decoder_input.view(-1, 512, 1024)
        # print("transformed decoder input type: ", transformed_decoder_input.dtype)
        # print("transformed decoder input: ", transformed_decoder_input.size())
        # print("\ntransformed_decoder_input: ", transformed_decoder_input)
        # """
        # transformed_decoder_input = self.test(decoder_inputs_embeds)
        # transformed_decoder_input = torch.rand((2, 128, 1024))
        # print("decoder_attention_mask: ", decoder_attention_mask)

        mean = transformed_token_repr.mean(dim=2, keepdim=True)
        std = transformed_token_repr.std(dim=2, keepdim=True)
        transformed_token_repr = (transformed_token_repr - mean) / std
        mean = decoder_inputs_embeds.mean(dim=2, keepdim=True)
        std = decoder_inputs_embeds.std(dim=2, keepdim=True)
        decoder_inputs_embeds = (decoder_inputs_embeds - mean) / std
        mean = transformed_char_repr.mean(dim=2, keepdim=True)
        std = transformed_char_repr.std(dim=2, keepdim=True)
        transformed_char_repr = (transformed_char_repr - mean) / std
        decoder_input = self.embedding_combiner([transformed_token_repr, decoder_inputs_embeds, transformed_char_repr])

        outputs = super().forward(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_input,
            # decoder_inputs_embeds=decoder_inputs_embeds+transformed_token_repr+transformed_char_repr,
            encoder_outputs=encoder_outputs,
            labels=labels,
            **kwargs
        )
        return (outputs, charbert_output[0], charbert_output[2])
        # decoder_outputs = super().forward(
        #         inputs_embeds=decoder_input_embeddings,
        #         # inputs_embeds=transformed_decoder_input+decoder_input_embeddings,
        #         attention_mask=decoder_attention_mask,
        #         encoder_hidden_states=encoder_outputs[0],
        #         labels=labels,
        #         **kwargs,
        #     )
        # print("\ndecoder outputs: ", decoder_outputs.logits)
        # return decoder_outputs
    
    # def _register_forward_hook(self)
    #     for name, module in self.named_modules():
    #         if name == 'decoder.output_projection':  # Replace with the layer you want to hook
    #             module.register_forward_hook(self.print_hook)

    # @staticmethod
    # def print_hook(module, input, output):
    #     print("")
        # print("Input to the layer:", input)
        # print("Output of the layer:", output)
    
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
    
    
def load_model():
    trocr_config = VisionEncoderDecoderConfig.from_pretrained(TROCR_CONFIG)
    charbert_config = cfg.charbert_config
    model = TrOCRCharBERTModel(charbert_args, config=trocr_config, charbert_config=charbert_config, max_target_length=MAX_TARGET_LENGTH)
    return model

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
    trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_CONFIG)
    trocr_state_dict = trocr_model.state_dict()
    # Standalone CharBERT model and state dict(adjusted according to the prefix)
    charbert_model = AdaptedRobertaModel.from_pretrained(charbert_args.model_name_or_path,
                                        from_tf=False,
                                        config=cfg.charbert_config,
                                        cache_dir=None)
    charbert_state_dict = charbert_model.state_dict()
    charbert_state_dict = {prefix + k: v for k, v in charbert_state_dict.items()}
    # Composite model and state dict
    model = load_model()
    state_dict = model.state_dict()
    # Load pretrained parameters
    # print("\n\nSTATE DICT")
    # print(state_dict.keys())
    trocr_state_dict.update(charbert_state_dict)
    for name, param in state_dict.items():
        if name in trocr_state_dict:
            state_dict[name].copy_(trocr_state_dict[name])
        elif (not name.startswith("transform")) and (not name.startswith("embedding_combiner")) and (name not in trocr_state_dict):
            message = f"Parameter {name} not found in the model."
            raise ValueError(message)
    # for name, param in trocr_state_dict.items():
    #     if (name in state_dict) and (name not in bypass_param):
    #         state_dict[name].copy_(param)
    #     elif name not in bypass_param:
    #         message = f"Parameter {name} not found in the model."
    #         raise ValueError(message)

    return model, state_dict, trocr_state_dict

def get_fine_tuned_param(experiment_version):
    model = load_model()
    model_path = os.path.join(MODELS, "trocr_charbert", experiment_version+".pth")
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
        # print("here")
        model = get_fine_tuned_param(experiment_version)
    else:
        # print("here2")
        model, state_dict, trocr_state_dict = get_pretrain_param()
        model.load_state_dict(state_dict, strict=False)
    # print(model.ffnn_embeddings.fc1.weight.requires_grad)
    # for k,v in trocr_state_dict.items():
    #     v_clone = v.clone()
    #     if (v_clone==0).all():
    #         print(k)
        # else:
        #     print("ok")
    # """
    # Filter out unnecessary keys
    # filtered_state_dict = {k: v for k, v in trocr_state_dict.items() if k in state_dict}

    # Load the filtered state dict
    # model.load_state_dict(filtered_state_dict, strict=False)
    # """
    # """
        # Check if the parameters are loaded correctly
        for name, param in model.named_parameters():
            # Set the print options to increase threshold
            # torch.set_printoptions(threshold=10_000)  # Set a high threshold value
            # If the parameter is not part of "ffnn" and does not match the state dict, raise an error
            if name not in bypass_param:
                # print(name)
                if not name.startswith("transform") and not name.startswith("embedding_combiner") and not torch.equal(trocr_state_dict.get(name, None), param):
                    message = f"Parameter {name} does not match."
                    raise ValueError(message)
        print("All pretrained parameters matched.")
    # """
    return model

# if __name__=="__main__":
#     initialise_trocr_charbert_model()