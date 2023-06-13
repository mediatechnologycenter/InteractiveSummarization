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

import numpy as np
from functools import partial, reduce
from nltk.tokenize import sent_tokenize
from transformers import PreTrainedTokenizer
from typing import Optional

from summarization.utils.constants import (
    GUIDANCE_IDS_NAME,
    GUIDANCE_ATTENTION_MASK_NAME,
    ORACLE_IDS_NAME,
)
from summarization.utils.typing import HFDataset


def tokenizer_function_ext(
    batch,
    tokenizer,
    src_field: str = "document",
    oracle_field: str = ORACLE_IDS_NAME,
    min_src_ntokens: int = 5,
    max_src_ntokens: int = 200,
    min_src_nsents: int = 3,
    max_src_nsents: int = 100,
    max_length_src: int = 512,
    prefix: str = "",
    language: str = "english",
):
    model_inputs = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "cls_position_ids": [],
        "cls_attention_mask": [],
        "labels": [],
        "filtered_text": [],
        "is_valid": [],
    }

    # sentence_separator = ' {} {} '.format(tokenizer.sep_token, tokenizer.cls_token)
    sentence_separator = [tokenizer.sep_token, tokenizer.cls_token]

    for doc, oracle_ids in zip(batch[src_field], batch[oracle_field]):
        # Split and tokenize sentences
        original_sentences = sent_tokenize(prefix + doc, language=language)
        sentences = [tokenizer.tokenize(s) for s in original_sentences]

        # Get labels
        labels = [0] * len(sentences)
        for sentence_idx in oracle_ids:
            labels[sentence_idx] = 1

        # Limit number of sentence tokens to a specific range - filter if necessary
        idxs = [i for i, s in enumerate(sentences) if (len(s) > min_src_ntokens)]
        sentences = [sentences[i][:max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]

        # Limit number of sentences to a specific range - filter if necessary
        sentences = sentences[:max_src_nsents]
        labels = labels[:max_src_nsents]

        # Add special tokens
        # text = [' '.join(sent) for sent in sentences]
        # text = sentence_separator.join(text)
        src_subtokens = reduce(lambda l, r: l + sentence_separator + r, sentences)
        src_subtokens = src_subtokens[: (max_length_src - 2)]
        src_subtokens = [tokenizer.cls_token] + src_subtokens + [tokenizer.sep_token]

        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        model_inputs["input_ids"].append(input_ids)

        # Get attention mask
        attention_mask = [1 for _ in range(len(input_ids))]
        model_inputs["attention_mask"].append(attention_mask)

        # Get token type IDs
        _segs = [-1] + [
            i for i, t in enumerate(input_ids) if t == tokenizer.sep_token_id
        ]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        token_type_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                token_type_ids += s * [0]
            else:
                token_type_ids += s * [1]
        model_inputs["token_type_ids"].append(token_type_ids)

        # Get cls position IDs
        cls_position_ids = [
            i for i, t in enumerate(input_ids) if t == tokenizer.cls_token_id
        ]
        model_inputs["cls_position_ids"].append(cls_position_ids)

        # Get cls attention mask
        cls_attention_mask = [1 for _ in range(len(cls_position_ids))]
        model_inputs["cls_attention_mask"].append(cls_attention_mask)

        # Truncate labels if necessary
        labels = labels[: len(cls_position_ids)]
        model_inputs["labels"].append(labels)

        # Add filtered source text
        filtered_text = [original_sentences[i] for i in idxs][: len(cls_position_ids)]
        model_inputs["filtered_text"].append(filtered_text)

        # Check if document has enough sentences
        is_valid = len(labels) >= min_src_nsents
        model_inputs["is_valid"].append(is_valid)

    return model_inputs


def tokenize_datasets_ext(
    datasets: HFDataset,
    tokenizer: PreTrainedTokenizer,
    src_field: str = "article",
    oracle_field: str = ORACLE_IDS_NAME,
    min_src_ntokens: int = 5,
    max_src_ntokens: int = 200,
    min_src_nsents: int = 3,
    max_src_nsents: int = 100,
    max_length_src: int = 512,
    prefix: str = "",
    language: str = "english",
    num_proc: int = 12,
    filtering: bool = True,
    filter_truncated_train: bool = False,
    filter_truncated_valid: bool = True,
    filter_truncated_test: bool = True,
    testing: bool = False,
    padding: bool = False,
    return_tensors: Optional[str] = None,
):
    """Tokenizes datasets."""

    tokenized_dataset = datasets.map(
        function=partial(
            tokenizer_function_ext,
            tokenizer=tokenizer,
            src_field=src_field,
            oracle_field=oracle_field,
            min_src_ntokens=min_src_ntokens,
            max_src_ntokens=max_src_ntokens,
            min_src_nsents=min_src_nsents,
            max_src_nsents=max_src_nsents,
            max_length_src=max_length_src,
            prefix=prefix,
            language=language,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=testing,
    )

    if filtering:
        # Filter out questions with too few sentences
        tokenized_dataset = tokenized_dataset.filter(
            function=lambda batch: batch["is_valid"],
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=testing,
        )
    else:  # TODO: Replace values with arguments
        # For each individual split determine src + tgt lengths and filter
        for split, do_filter in zip(
            ["train", "validation", "test"],
            [filter_truncated_train, filter_truncated_valid, filter_truncated_test],
        ):
            if do_filter:
                # Get untruncated lengths
                dataset_for_lengths = datasets[split].map(
                    function=partial(
                        tokenizer_function_abs,
                        tokenizer=tokenizer,
                        max_length_src=1024,
                        max_length_tgt=1024,
                        truncation=False,
                        prefix=prefix,
                        guidance_field="guidance",
                        src_field=src_field,
                        tgt_field="highlights",
                        padding=padding,
                        return_tensors=return_tensors,
                    ),
                    batched=True,
                    num_proc=num_proc,
                    load_from_cache_file=testing,
                )

                # Filter input
                sl = np.array([len(i["input_ids"]) for i in dataset_for_lengths])
                lengths = sl <= 1024

                # Filter output
                tl = np.array([len(i["labels"]) for i in dataset_for_lengths])
                lengths = lengths & (tl <= 1024)

                # Filter guidance
                if GUIDANCE_IDS_NAME in tokenized_dataset[split].features:
                    gl = np.array([len(i["labels"]) for i in dataset_for_lengths])
                    lengths = lengths & (gl <= 1024)

                tokenized_dataset[split] = tokenized_dataset[split].filter(
                    function=lambda _, ids: lengths[ids],
                    batched=True,
                    num_proc=num_proc,
                    with_indices=True,
                    load_from_cache_file=testing,
                )

    return tokenized_dataset


def tokenizer_function_abs(
    examples,
    tokenizer,
    max_length_src: int,
    max_length_tgt: int,
    truncation: bool,
    prefix: str = "",
    guidance_field: str = "guidance",
    src_field: str = "article",
    tgt_field: str = None,
    padding: str = False,
    return_tensors: Optional[str] = None,
):
    inputs = [prefix + doc for doc in examples[src_field]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length_src,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )

    key_obj = examples.data if hasattr(examples, "data") else examples
    if guidance_field and guidance_field in key_obj.keys():
        guidance_intputs = tokenizer(
            examples[guidance_field],
            max_length=max_length_src,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
        model_inputs[GUIDANCE_IDS_NAME] = guidance_intputs["input_ids"]
        model_inputs[GUIDANCE_ATTENTION_MASK_NAME] = guidance_intputs["attention_mask"]

    # Setup the tokenizer for targets
    if tgt_field:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[tgt_field],
                max_length=max_length_tgt,
                truncation=truncation,
                padding=False,
                return_tensors=return_tensors,
            )

        model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def tokenize_datasets_abs(
    datasets: HFDataset,
    tokenizer: PreTrainedTokenizer,
    src_field: str = "article",
    tgt_field: str = "highlights",
    guidance_field: str = "guidance",
    max_src_sample_len: int = 1024,
    max_tgt_sample_len: int = 1024,
    num_proc: int = 12,
    prefix: str = "",
    truncation: bool = True,
    filter_truncated_train: bool = False,
    filter_truncated_valid: bool = True,
    filter_truncated_test: bool = True,
    testing: bool = False,
    padding: bool = False,
    return_tensors: Optional[str] = None,
):
    """Tokenizes datasets."""

    tokenized_dataset = datasets.map(
        function=partial(
            tokenizer_function_abs,
            tokenizer=tokenizer,
            max_length_src=max_src_sample_len,
            max_length_tgt=max_tgt_sample_len,
            truncation=truncation,
            prefix=prefix,
            guidance_field=guidance_field,
            src_field=src_field,
            tgt_field=tgt_field,
            padding=padding,
            return_tensors=return_tensors,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=testing,
    )

    if truncation:
        # For each individual split determine src + tgt lengths and filter
        for split, do_filter in zip(
            ["train", "validation", "test"],
            [filter_truncated_train, filter_truncated_valid, filter_truncated_test],
        ):
            if do_filter:
                # Get untruncated lengths
                dataset_for_lengths = datasets[split].map(
                    function=partial(
                        tokenizer_function_abs,
                        tokenizer=tokenizer,
                        max_length_src=max_src_sample_len,
                        max_length_tgt=max_tgt_sample_len,
                        truncation=False,
                        prefix=prefix,
                        guidance_field=guidance_field,
                        src_field=src_field,
                        tgt_field=tgt_field,
                        padding=padding,
                        return_tensors=return_tensors,
                    ),
                    batched=True,
                    num_proc=num_proc,
                    load_from_cache_file=testing,
                )

                # Filter input
                sl = np.array([len(i["input_ids"]) for i in dataset_for_lengths])
                lengths = sl <= max_src_sample_len

                # Filter output
                tl = np.array([len(i["labels"]) for i in dataset_for_lengths])
                lengths = lengths & (tl <= max_tgt_sample_len)

                # Filter guidance
                if GUIDANCE_IDS_NAME in tokenized_dataset[split].features:
                    gl = np.array([len(i["labels"]) for i in dataset_for_lengths])
                    lengths = lengths & (gl <= max_src_sample_len)

                tokenized_dataset[split] = tokenized_dataset[split].filter(
                    function=lambda _, ids: lengths[ids],
                    batched=True,
                    num_proc=num_proc,
                    with_indices=True,
                    load_from_cache_file=testing,
                )

    return tokenized_dataset
