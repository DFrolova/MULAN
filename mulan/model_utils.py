from typing import Optional, List, Union, Tuple
import math

import torch
from torch import nn

from transformers import EsmModel, EsmConfig, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.esm.modeling_esm import (
    create_position_ids_from_input_ids, 
    EsmEmbeddings,
    EsmPooler, 
    EsmEncoder,
    symmetrize,
    )



def get_extended_attention_mask(attention_mask, input_shape, dtype):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


# MLP
class StructEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_attention_heads, 
                 dtype, mask_angle_inputs_with_plddt):
        super().__init__()
        self.MLP = nn.Linear(input_dim, hidden_dim)

        esm_config = EsmConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-05,
            emb_layer_norm_before=False,
            token_dropout=True,
            esmfold_config=None,
            vocab_list=None,
            position_embedding_type='rotary',
        )
        self.encoder = EsmEncoder(esm_config)
        self.dtype = dtype
        self.mask_angle_inputs_with_plddt = mask_angle_inputs_with_plddt

    def forward(
        self, struct_inputs, attention_mask,
    ):
        embeddings = self.MLP(struct_inputs)

        input_shape = embeddings.size()[:-1]
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, self.dtype)

        embeddings = self.encoder(embeddings,
                                  attention_mask=extended_attention_mask,)
        embeddings = embeddings['last_hidden_state']
        return embeddings


class CoordinatePositionalEncoder(nn.Module):
    """
    Embeds xyz positions into vector representations.
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, coords):
        return self.mlp(coords)
    

class StructEsmEmbeddings(EsmEmbeddings):
    """
    Same as EsmEmbeddings with extra structural positional embeddings.
    """

    def __init__(self, config, num_struct_embeddings_layers, struct_data_dim, 
                 use_struct_embeddings, dtype, 
                 mask_angle_inputs_with_plddt, add_foldseek_embeddings, fs_tokenizer):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, 
                                            padding_idx=config.pad_token_id)
        
        self.add_foldseek_embeddings = add_foldseek_embeddings
        if self.add_foldseek_embeddings:
            self.foldseek_embeddings = nn.Embedding(len(fs_tokenizer), config.hidden_size, 
                                                    padding_idx=fs_tokenizer.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

        self.use_struct_embeddings = use_struct_embeddings

        if self.use_struct_embeddings:
            self.struct_embeddings = StructEmbeddings(
                input_dim=struct_data_dim, 
                hidden_dim=config.hidden_size, 
                num_hidden_layers=num_struct_embeddings_layers, 
                num_attention_heads=config.num_attention_heads,
                dtype=dtype,
                mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt)
                        
    def forward(
        self, input_ids=None, attention_mask=None, 
        struct_inputs=None, position_ids=None, 
        inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.use_struct_embeddings:
            struct_embeddings = self.struct_embeddings(struct_inputs[0], attention_mask)
            # struct_embeddings = self.struct_embeddings(struct_inputs[0])
            embeddings += struct_embeddings

        if self.add_foldseek_embeddings:
            fs_embeddings = self.foldseek_embeddings(struct_inputs[2])
            embeddings += fs_embeddings

        # Matt: ESM has the option to handle masking in MLM in a slightly unusual way. If the token_dropout
        # flag is False then it is handled in the same was as BERT/RoBERTa. If it is set to True, however,
        # masked tokens are treated as if they were selected for input dropout and zeroed out.
        # This "mask-dropout" is compensated for when masked tokens are not present, by scaling embeddings by
        # a factor of (fraction of unmasked tokens during training) / (fraction of unmasked tokens in sample).
        # This is analogous to the way that dropout layers scale down outputs during evaluation when not
        # actually dropping out values (or, equivalently, scale up their un-dropped outputs in training).
        if self.token_dropout:
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            print('added position embeddings')

        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)

        return embeddings
    

def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class NonLinearHead(nn.Module):
    """ESM Head for angle regression."""

    def __init__(self, input_dim, output_dim, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)

        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        x = self.decoder(x)
        return x
    

def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    avg = x.sum(-1, keepdims=True) * x.sum(-2, keepdims=True)
    avg.div_(x.sum((-1, -2), keepdims=True))  # in-place to reduce memory
    x -= avg # normalize
    return x
    

class ContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, out_features, bias)

    def forward(self, tokens, attentions):
        # remove eos token attentions
        eos_mask = tokens.ne(self.eos_idx).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: batch x channels x tokens x tokens (symmetric)
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.regression(attentions).squeeze(3)
    

class StructEsmModel(EsmModel):
    def __init__(self, config, num_struct_embeddings_layers, struct_data_dim, 
                 use_struct_embeddings,
                 mask_angle_inputs_with_plddt,
                 add_foldseek_embeddings,
                 fs_tokenizer,
                 add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = StructEsmEmbeddings(config, 
                                              num_struct_embeddings_layers=num_struct_embeddings_layers, 
                                              struct_data_dim=struct_data_dim, 
                                              use_struct_embeddings=use_struct_embeddings,
                                              dtype=self.dtype,
                                              mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
                                              add_foldseek_embeddings=add_foldseek_embeddings,
                                              fs_tokenizer=fs_tokenizer)
        self.encoder = EsmEncoder(config)
        self.pooler = EsmPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        struct_inputs: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
            
        embedding_output = self.embeddings(
            input_ids=input_ids,
            struct_inputs=struct_inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
def auto_detect_base_tokenizer(config, use_foldseek_sequences):
    if config.num_hidden_layers == 6:
        base_checkpoint = 'facebook/esm2_t6_8M_UR50D'
    elif config.num_hidden_layers == 12:
        if use_foldseek_sequences:
            base_checkpoint = 'westlake-repl/SaProt_35M_AF2'
        else:
            base_checkpoint = 'facebook/esm2_t12_35M_UR50D'
    else:
        base_checkpoint = None
    assert base_checkpoint is not None, 'The base PLM undefined'
    esm_tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    return esm_tokenizer
