# Copyright 2023 ETH Zurich, Media Technology Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import random
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers import BartModel, BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import (
    BartDecoder,
    BartEncoder,
    BartEncoderLayer,
    BartAttention,
    _expand_mask,
    shift_tokens_right,
)
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)

from summarization.models.model_utils import DualBaseModelOutput
from summarization.models.mixin.load_model_mixin import GuidedModelLoadMixin

from typing import Optional, Union, Tuple, List

from summarization.utils.constants import (
    GUIDANCE_ATTENTION_MASK_NAME,
    GUIDANCE_IDS_NAME,
)

logger = logging.get_logger(__name__)


################################################################################
# Decoder Modules & Layers
################################################################################


class GuidedBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.cross_attn_guidance_first = config.cross_attn_guidance_first

        # Self attention
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # Corss attention 1 - Guidance
        self.encoder_attn1 = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm1 = nn.LayerNorm(self.embed_dim)

        # Cross attention 2 - Source
        self.encoder_attn2 = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm2 = nn.LayerNorm(self.embed_dim)

        # Feed forward
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guidance_encoder_hidden_states: Optional[torch.Tensor] = None,
        guidance_encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """Check BartDecoderLayer for arguments."""
        residual = hidden_states

        # Self Attention decoder uni-directional self-attention cached
        # key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        # Add present self-attn cache to positions 1,2 of present_key_value
        # tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if self.cross_attn_guidance_first:
            enc_hidden_states = guidance_encoder_hidden_states
            enc_attention_mask = guidance_encoder_attention_mask
            encoder_attention = self.encoder_attn1
            encoder_attention_layer_norm = self.encoder_attn_layer_norm1
        else:
            enc_hidden_states = encoder_hidden_states
            enc_attention_mask = encoder_attention_mask
            encoder_attention = self.encoder_attn2
            encoder_attention_layer_norm = self.encoder_attn_layer_norm2

        if enc_hidden_states is not None:
            ##### First Cross-Attention Block

            residual = hidden_states

            # Cross_attn cached key/values tuple is at positions 3,4 of
            # present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-4:-2] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = encoder_attention(
                hidden_states=hidden_states,
                key_value_states=enc_hidden_states,
                attention_mask=enc_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = encoder_attention_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        if not self.cross_attn_guidance_first:
            enc_hidden_states = guidance_encoder_hidden_states
            enc_attention_mask = guidance_encoder_attention_mask
            encoder_attention = self.encoder_attn1
            encoder_attention_layer_norm = self.encoder_attn_layer_norm1
        else:
            enc_hidden_states = encoder_hidden_states
            enc_attention_mask = encoder_attention_mask
            encoder_attention = self.encoder_attn2
            encoder_attention_layer_norm = self.encoder_attn_layer_norm2

        if enc_hidden_states is not None:
            ##### Second Cross-Attention Block

            residual = hidden_states

            # Cross_attn cached key/values tuple is at positions 3,4 of
            # present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = encoder_attention(
                hidden_states=hidden_states,
                key_value_states=enc_hidden_states,
                attention_mask=enc_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = encoder_attention_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


################################################################################
# Bart Encoder / Decoder
################################################################################


class GuidedBartEncoder(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention
    layers. Each layer is a [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        self.wait_for_post_init = True
        super().__init__(config, embed_tokens)
        self.wait_for_post_init = False

        self.n_bart_layers = len(self.layers)
        if not config.add_extra_bart_encoder_layers:
            self.shared_layers = self.layers[:-1]
        else:
            self.shared_layers = self.layers
        self.layers = None
        if self.config.source_top_encoder_layer:
            self.source_layer = BartEncoderLayer(config)
        self.guidance_layer = BartEncoderLayer(config)
        self.post_init()

    def post_init(self):
        if hasattr(self, "wait_for_post_init") and self.wait_for_post_init:
            return
        super().post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        guidance_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        guidance_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """Check `BartEncoder` for args"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and "
                "inputs_embeds at the same time"
            )

        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        else:
            raise ValueError("You have to specify either input_ids or " "inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Retrieve guidance_ids and guidance_embeds
        if guidance_ids is not None and guidance_embeds is not None:
            raise ValueError(
                "You cannot specify both guidance_ids and "
                "guidance_embeds at the same time"
            )

        elif guidance_ids is not None:
            guidance_shape = guidance_ids.size()
            guidance_ids = guidance_ids.view(-1, guidance_shape[-1])

        elif guidance_embeds is not None:
            guidance_shape = guidance_embeds.size()[:-1]

        else:
            raise ValueError(
                "You have to specify either guidance_ids or " "guidance_embeds"
            )
        if guidance_embeds is None:
            guidance_embeds = self.embed_tokens(guidance_ids) * self.embed_scale

        # Embed_pos hidden_state
        embed_pos = self.embed_positions(input_shape)

        # Hidden state input
        inputs_hidden_states = inputs_embeds + embed_pos
        inputs_hidden_states = self.layernorm_embedding(inputs_hidden_states)
        inputs_hidden_states = nn.functional.dropout(
            inputs_hidden_states, p=self.dropout, training=self.training
        )

        # Embed_pos guidance_hidden_state
        guidance_embed_pos = self.embed_positions(guidance_shape)

        # Hidden state guidance
        guidance_hidden_states = guidance_embeds + guidance_embed_pos
        guidance_hidden_states = self.layernorm_embedding(guidance_hidden_states)
        guidance_hidden_states = nn.functional.dropout(
            guidance_hidden_states, p=self.dropout, training=self.training
        )

        # Expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # Expand guidance_attention_mask
        if guidance_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            guidance_attention_mask = _expand_mask(
                guidance_attention_mask, guidance_embeds.dtype
            )

        # Check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.shared_layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.shared_layers)} "
                    "layers, but it is for {head_mask.size()[0]}."
                )

        global_encoder_states = ()
        global_attention_states = ()
        global_hidden_states = ()

        # Overall hidden states
        hidden_states = inputs_hidden_states

        for signal in ["source", "guidance"]:
            # Set hidden_states
            if signal == "source":
                hidden_states = inputs_hidden_states
                current_attention_mask = attention_mask
            elif signal == "guidance":
                hidden_states = guidance_hidden_states
                current_attention_mask = guidance_attention_mask

            # Tracking of output_hidden_states and output_attentions
            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for idx in range(len(self.shared_layers) + 1):
                if idx < len(self.shared_layers):
                    encoder_layer = self.shared_layers[idx]
                else:
                    if signal == "source":
                        if self.config.source_top_encoder_layer:
                            encoder_layer = self.source_layer
                        else:
                            encoder_layer = []
                    else:
                        encoder_layer = self.guidance_layer

                if not encoder_layer:
                    continue

                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                # Add LayerDrop
                # (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)

                if self.training and (dropout_probability < self.layerdrop):
                    # Skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.gradient_checkpointing and self.shared_layers:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            current_attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            current_attention_mask,
                            layer_head_mask=(
                                head_mask[idx] if head_mask is not None else None
                            ),
                            output_attentions=output_attentions,
                        )

                    # Get output hidden states
                    hidden_states = layer_outputs[0]

                # Track attentions
                if output_attentions:
                    all_attentions = all_attentions + (
                        (layer_outputs[0][1], layer_outputs[1][1]),
                    )

            if output_attentions:
                global_attention_states += (all_attentions,)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
                global_encoder_states += (encoder_states,)

            global_hidden_states += (hidden_states,)

        hidden_states = global_hidden_states
        encoder_states = tuple(zip(*encoder_states)) if output_hidden_states else None
        all_attentions = tuple(zip(*all_attentions)) if output_attentions else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )

        return DualBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class GuidedBartDecoder(BartDecoder):
    """Transformer decoder consisting of *config.decoder_layers* layers."""

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        self.wait_for_post_init = True
        super().__init__(config, embed_tokens)
        self.wait_for_post_init = False

        self.layers = nn.ModuleList(
            [GuidedBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        if hasattr(self, "wait_for_post_init") and self.wait_for_post_init:
            return
        super().post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        guidance_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        guidance_encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""Check Bart Decoder's method for documentation."""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and "
                "decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or "
                "decoder_inputs_embeds"
            )

        # Past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # Expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # Expand encoder guidance attention mask
        if (
            guidance_encoder_hidden_states is not None
            and guidance_encoder_attention_mask is not None
        ):
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            guidance_encoder_attention_mask = _expand_mask(
                guidance_encoder_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )

        # Embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        # Check if head_mask/cross_attn_head_mask has a correct number of
        # layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for "
                        f"{len(self.layers)} layers, but it is for "
                        f"{head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop
            # (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient "
                        "checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    guidance_encoder_hidden_states,
                    guidance_encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    guidance_encoder_hidden_states=guidance_encoder_hidden_states,
                    guidance_encoder_attention_mask=guidance_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


################################################################################
# Guided Bart Model
################################################################################


class GuidedBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = GuidedBartEncoder(config, self.shared)
        self.decoder = GuidedBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_ids: torch.LongTensor = None,
        guidance_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError("Should not happen.")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                guidance_ids=guidance_ids,
                attention_mask=attention_mask,
                guidance_attention_mask=guidance_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a
        # DualBaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, DualBaseModelOutput):
            encoder_outputs = DualBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value,
        # dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state[0],
            encoder_attention_mask=attention_mask,
            guidance_encoder_hidden_states=encoder_outputs.last_hidden_state[1],
            guidance_encoder_attention_mask=guidance_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


################################################################################
# Guided Bart Model for Conditional Generation
################################################################################


class GuidedBartConfig(BartConfig):
    model_type = "guided_bart"

    def __init__(
        self,
        load_encoder_shared: bool = True,
        load_encoder_source: bool = True,
        load_encoder_guidance: bool = True,
        load_decoder_crossattention_guidance: bool = False,
        load_decoder_crossattention_source: bool = True,
        source_top_encoder_layer: bool = False,
        cross_attn_guidance_first: bool = True,
        add_extra_bart_encoder_layers: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.load_encoder_shared = load_encoder_shared
        self.load_encoder_source = load_encoder_source
        self.load_encoder_guidance = load_encoder_guidance
        self.load_decoder_crossattention_guidance = load_decoder_crossattention_guidance
        self.load_decoder_crossattention_source = load_decoder_crossattention_source
        self.source_top_encoder_layer = source_top_encoder_layer
        self.cross_attn_guidance_first = cross_attn_guidance_first
        self.add_extra_bart_encoder_layers = add_extra_bart_encoder_layers


from transformers.utils import ModelOutput
from typing import Dict, Any


class GuidedBartForConditionalGeneration(
    GuidedModelLoadMixin, BartForConditionalGeneration
):
    config_class = GuidedBartConfig

    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.model = GuidedBartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_ids: torch.LongTensor = None,
        guidance_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""Check BartForConditionalGeneration for arguments."""

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` "
                    "since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            guidance_ids=guidance_ids,
            guidance_attention_mask=guidance_attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        guidance_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            GUIDANCE_IDS_NAME: None,  # encoder_outputs is defined. guidance_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            GUIDANCE_ATTENTION_MASK_NAME: guidance_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if "guidance_attention_mask" in model_kwargs:
            model_kwargs["guidance_attention_mask"] = model_kwargs[
                "guidance_attention_mask"
            ].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs[
                "last_hidden_state"
            ] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs
