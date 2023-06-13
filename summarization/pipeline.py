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

from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, MBartTokenizer, ModelCard, PreTrainedTokenizer
from transformers.pipelines.text2text_generation import SummarizationPipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor

from summarization.dataset.tokenize import tokenizer_function_abs
from summarization.models.guided_bart import GuidedBartForConditionalGeneration
from summarization.models.guided_mbart import GuidedMBartForConditionalGeneration

from typing import Optional, Union


class GuidedSummarizationPipeline(SummarizationPipeline):
    def __init__(
        self,
        pretrained_path: str = None,
        model: PreTrainedModel = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = "pt",
        task: str = "",
        device: int = -1,
        binary_output: bool = False,
        batch_size: int = 1,
        num_workers: int = None,
        max_src_length: int = 1024,
        max_tgt_length: int = 1024,
        use_auth_token: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        assert pretrained_path or (
            model and tokenizer
        ), "Please provide either pretrained_path or model+tokenizer."

        multilingual = False
        lang = "de_DE"

        if not (model and tokenizer):
            if not tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_path, use_fast=False, use_auth_token=use_auth_token
                )
                multilingual = isinstance(tokenizer, MBartTokenizer)
                if multilingual:
                    tokenizer.src_lang = lang
                    tokenizer.tgt_lang = lang

            if not model:
                if multilingual:
                    model = GuidedMBartForConditionalGeneration.from_pretrained(
                        pretrained_path, use_auth_token=use_auth_token
                    )
                    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[
                        lang
                    ]
                else:
                    model = GuidedBartForConditionalGeneration.from_pretrained(
                        pretrained_path, use_auth_token=use_auth_token
                    )

        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.modelcard = modelcard
        self.framework = framework
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.device = (
            device
            if framework == "tf"
            else torch.device("cpu" if device < 0 else f"cuda:{device}")
        )
        self.binary_output = binary_output

        # Special handling
        if self.framework == "pt" and self.device.type == "cuda":
            self.model = self.model.to(self.device)

        # Update config with task specific parameters
        task_specific_params = self.model.config.task_specific_params
        if task_specific_params is not None and task in task_specific_params:
            self.model.config.update(task_specific_params.get(task))

        self.call_count = 0
        self._batch_size = batch_size
        self._num_workers = num_workers
        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

        assert type(model) == GuidedMBartForConditionalGeneration

    def _parse_and_tokenize(self, *args, truncation):
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError(
                    "Please make sure that the tokenizer has "
                    "a pad_token_id when using a batch input"
                )
            padding = True
            inputs = args[0]

        elif isinstance(args[0], dict):
            padding = False
            inputs = [args[0]]
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. "
                f"The should be either of type `str` or type `list`"
            )

        inputs = {key: [i[key] for i in inputs] for key in inputs[0]}
        assert "article" in inputs and "guidance" in inputs

        prefix = (
            self.model.config.prefix if self.model.config.prefix is not None else ""
        )

        inputs = tokenizer_function_abs(
            inputs,
            tokenizer=self.tokenizer,
            truncation=truncation,
            padding=padding,
            max_length_src=self.max_src_length,
            max_length_tgt=self.max_tgt_length,
            tgt_field="",
            prefix=prefix,
            return_tensors=self.framework,
        )

        return inputs
