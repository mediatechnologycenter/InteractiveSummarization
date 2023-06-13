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

from dataclasses import dataclass
from typing import Any, Optional, Union

from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from utils.constants import GUIDANCE_IDS_NAME, GUIDANCE_ATTENTION_MASK_NAME


@dataclass
class DataCollatorForGuidedSeq2Seq:
    """Check huggingface's DataCollatorForSeq2Seq for arguments."""

    tokenizer: PreTrainedTokenizerBase
    trainer: Optional[Trainer] = None
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )

        # We have to pad the labels before calling `tokenizer.pad` as this
        # method won't pad them and needs them of the same length to return
        # tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        if GUIDANCE_IDS_NAME in features[0]:
            guidance_features = self.tokenizer.pad(
                [
                    {
                        "input_ids": sample[GUIDANCE_IDS_NAME],
                        "attention_mask": sample.get(
                            GUIDANCE_ATTENTION_MASK_NAME, None
                        ),
                    }
                    for sample in features
                ],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features = self.tokenizer.pad(
                [
                    {
                        k: v
                        for k, v in sample.items()
                        if k not in [GUIDANCE_IDS_NAME, GUIDANCE_ATTENTION_MASK_NAME]
                    }
                    for sample in features
                ],
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            features[GUIDANCE_IDS_NAME] = guidance_features["input_ids"]
            features[GUIDANCE_ATTENTION_MASK_NAME] = guidance_features.get(
                "attention_mask", None
            )
        else:
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

        # prepare decoder_input_ids
        model = self.model
        if not model:
            if self.trainer and hasattr(self.trainer, "model"):
                model = self.trainer.model

        if (
            labels is not None
            and model is not None
            and hasattr(model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


# Adds zero-padding to have fixed-length sequences
def _pad(data, pad_id=0, max_size=-1, padding_side="right"):
    if max_size == -1:
        max_size = max(len(d) for d in data)

    if padding_side == "right":
        rtn_data = [list(d) + [pad_id] * (max_size - len(d)) for d in data]
    else:
        rtn_data = [[pad_id] * (max_size - len(d)) + list(d) for d in data]

    return rtn_data


@dataclass
class DataCollatorForSentenceClassification:
    """Check huggingface's DataCollatorForTokenClassification for arguments."""

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )

        # input_ids, attention_mask, token_type_ids will be properly padded here
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        padding_side = self.tokenizer.padding_side

        # Pad sentence labels
        batch[label_name] = _pad(
            labels, pad_id=self.label_pad_token_id, padding_side=padding_side
        )

        # Pad <CLS> token position ids
        cls_position_ids = [feature["cls_position_ids"] for feature in features]
        batch["cls_position_ids"] = _pad(cls_position_ids, padding_side=padding_side)

        # Pad <CLS> token attention masks
        cls_attention_mask = [feature["cls_attention_mask"] for feature in features]
        batch["cls_attention_mask"] = _pad(
            cls_attention_mask, padding_side=padding_side
        )

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
