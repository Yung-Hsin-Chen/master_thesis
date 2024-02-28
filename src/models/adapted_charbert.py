from src.models.charbert.modeling.modeling_charbert import CharBertModel, CharBertEmbeddings, BertEmbeddings
from src.models.charbert.modeling.modeling_roberta import RobertaModel, RobertaForMaskedLM
from src.models.charbert.modeling.configuration_roberta import RobertaConfig
from transformers import RobertaTokenizer
import json
import torch
import os
from collections import OrderedDict
import torch
import torch.nn as nn

class AdaptedCharBertEmbeddings(CharBertEmbeddings):
    def __init__(self, config, is_roberta=False):
        super(AdaptedCharBertEmbeddings, self).__init__(config, is_roberta)

    def forward(self, char_input_ids=None, start_ids=None, end_ids=None, input_embeds=None):
        # print("In adapted charbert embeddings")
        if input_embeds is not None:
            char_embeddings = input_embeds
            batch_size, char_maxlen, _ = char_embeddings.size()
        else:
            return super().forward(char_input_ids, start_ids, end_ids)

        self.rnn_layer.flatten_parameters()
        all_hiddens, _ = self.rnn_layer(char_embeddings)
        start_one_hot = nn.functional.one_hot(start_ids, num_classes=char_maxlen)
        end_one_hot = nn.functional.one_hot(end_ids, num_classes=char_maxlen)
        start_hidden = torch.matmul(start_one_hot.float(), all_hiddens)
        end_hidden = torch.matmul(end_one_hot.float(), all_hiddens)
        char_embeddings_repr = torch.cat([start_hidden, end_hidden], dim=-1)

        return char_embeddings_repr
    
class AdaptedRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        # Replace the embeddings with AdaptedRobertaEmbeddings
        self.char_embeddings = AdaptedCharBertEmbeddings(config, is_roberta=True)
        self.init_weights()
    
    def forward(self, char_input_ids=None, start_ids=None, end_ids=None, char_input_embeds=None, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        # char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids, char_input_embeds)
        # print(f'shape info in CharBertModel: input_ids {input_ids.size()}')
        # print(f'shape info in CharBertModel: char_input_ids {char_input_ids.size()}')
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                               encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,\
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        #print(f'input shape for bert_encoder: embedding_output {embedding_output.size()}')
        #print(f'extended_attention_mask: {extended_attention_mask.size()}') 
        #print(f'head_mask: {head_mask.size()} encoder_hidden_states: {encoder_hidden_states.size()}')
        #print(f'encoder_attention_mask: {encoder_extended_attention_mask.size()}')
        #encoder_outputs = self.encoder(embedding_output,
        #                               attention_mask=extended_attention_mask,
        #                               head_mask=head_mask,
        #                               encoder_hidden_states=encoder_hidden_states,
        #                               encoder_attention_mask=encoder_extended_attention_mask)

        # char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids)
        char_embeddings = self.char_embeddings(char_input_ids, start_ids, end_ids, char_input_embeds)
        char_encoder_outputs = self.encoder(char_embeddings,
                                       embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        
        sequence_output, char_sequence_output = char_encoder_outputs[0], char_encoder_outputs[1]
        pooled_output = self.pooler(sequence_output)
        char_pooled_output = self.pooler(char_sequence_output)

        outputs = (sequence_output, pooled_output, char_sequence_output, char_pooled_output) + char_encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
    