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
from torch import nn
from torch.nn.init import xavier_uniform_

from transformers.utils import logging
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from typing import Optional, Union, Tuple

from models.neural import PositionalEncoding, TransformerEncoderLayer

logger = logging.get_logger(__name__)


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.wo = nn.Linear(d_model, 1, bias=True)
        # self.sigmoid = nn.Sigmoid()
        self.wo = nn.Linear(d_model, 2, bias=True)

    def forward(self, top_vecs, mask):
        """See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](
                i, x, x, 1 - mask
            )  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        # sent_scores = self.sigmoid(self.wo(x))
        sent_scores = self.wo(x)
        sent_scores = sent_scores.squeeze(-1) * mask[:, :, None].float()

        return sent_scores


class BertSumExtConfig(BertConfig):
    model_type = "extractive_bert"

    def __init__(
        self,
        inter_ff_size: int = 2048,
        inter_heads: int = 8,
        inter_layers: int = 2,
        inter_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.inter_ff_size = inter_ff_size
        self.inter_heads = inter_heads
        self.inter_layers = inter_layers
        self.inter_dropout = inter_dropout


class BertSumExt(BertPreTrainedModel):
    config_class = BertSumExtConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        """classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)"""

        """
        Additional arguments:
        - ff_size / feed-forward
        - heads
        - dropout
        - inter_layers
        """

        self.encoder = TransformerInterEncoder(
            d_model=config.hidden_size,
            d_ff=config.inter_ff_size,
            heads=config.inter_heads,
            dropout=config.inter_dropout,
            num_inter_layers=config.inter_layers,
        )

        # TODO: Is it necessary to initialize the encoder weights?
        """if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)"""

        # TODO: added, necessary?
        """for p in self.encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)"""

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cls_position_ids: Optional[torch.Tensor] = None,
        cls_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = bert_outputs[0]
        # TODO: add dropout?
        # hidden_states = self.dropout(hidden_states)

        # Extract cls hidden states
        cls_hidden_states = hidden_states[
            torch.arange(hidden_states.size(0)).unsqueeze(1), cls_position_ids
        ]
        # Apply mask
        cls_hidden_states = cls_hidden_states * cls_attention_mask[:, :, None].float()
        # Apply sentence transformer
        # outputs, logits = self.encoder(cls_hidden_states, cls_attention_mask).squeeze(-1)
        logits = self.encoder(cls_hidden_states, cls_attention_mask).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            print("Should not happen.")
            return None
            # output = (logits,) + outputs[2:]
            # return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,  # outputs.hidden_states
            attentions=None,  # outputs.attentions
        )
