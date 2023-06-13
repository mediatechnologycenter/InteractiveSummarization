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

""" Metrics for Interactive Summarization. """

import nltk
import datasets
import transformers

from typing import List

_CITATION = """ TODO """

_DESCRIPTION = """ TODO """

_KWARGS_DESCRIPTION = """ TODO """


def simple_accuracy(preds, labels):
    return (preds == labels).mean().item()


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class ExtSumMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "labels_ids": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _download_and_prepare(self, dl_manager):
        """Init metrics used."""
        self.rouge = datasets.load_metric("rouge")

    def logits_to_predictions(self, logits):
        # HF seems to return more than predictions only in the case of BART
        logits = (logits > 0.5).astype(int)
        return logits

    @datasets.utils.file_utils.add_start_docstrings(_KWARGS_DESCRIPTION)
    def compute(
        self,
        eval_pred: transformers.EvalPrediction,
        tokenizer: transformers.PreTrainedTokenizer,
        predictions_are_logits: bool = True,
    ):
        """Computes interactive summarization metric between predictions and
        references."""

        # Extract relevant data
        predictions, labels = eval_pred

        # Update predictions if they are logits
        if predictions_are_logits:
            predictions = self.logits_to_predictions(logits=predictions)

        return {"accuracy": simple_accuracy(predictions, labels)}

    def get_rouge(
        self, extracted_sent: List, tgt_sum: List[str], metric_key_prefix: str = "test"
    ):
        """Computes summarization metric between predictions and
        references."""

        # Rouge expects a newline after each sentence
        predictions = ["\n".join(pred) for pred in extracted_sent]
        references = ["\n".join(nltk.sent_tokenize(tgt.strip())) for tgt in tgt_sum]

        result = self.rouge.compute(
            predictions=predictions, references=references, use_stemmer=True
        )

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(result.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                result[f"{metric_key_prefix}_{key}"] = result.pop(key)

        return {k: round(v, 4) for k, v in result.items()}
