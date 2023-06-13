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

import os
import randomname
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from transformers import TrainingArguments, Seq2SeqTrainingArguments

from typing import List, Optional, Dict

from .constants import GUIDANCE_IDS_NAME


################################################################################
# Mixin Arguments
################################################################################


@dataclass
class TrainMetricsArguments:
    """Extension for TrainingArguments to allow to log train metrics."""

    train_metric_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "How many samples to use (sliding window) to log train metrics. If"
            "train_metric_samples==None, no train metrics are logged."
        },
    )


@dataclass
class TokenBatchArguments:
    """Extension for TrainingArguments to allow token batching."""

    token_batching: bool = field(
        default=False,
        metadata={"help": "Whether to use token batching, instead of samples."},
    )

    batch_size_includes_padding: bool = field(
        default=True, metadata={"help": "Whether token batching considers for padding."}
    )

    token_batching_with_output: bool = field(
        default=True,
        metadata={
            "help": "Whether token batching consider the output/decoder " "samples."
        },
    )

    output_length_column_name: str = field(
        default="output_length",
        metadata={
            "help": "The name of dataset feature that contains the length of the "
            "output/decoder samples."
        },
    )

    model_output_name: str = field(
        default="labels",
        metadata={
            "help": "The name of the output/decoder samples to compute their lengths in "
            "case of token_batching_with_output==True and if "
            "output_length_column_name is not present in dataset."
        },
    )


@dataclass
class GuidedTokenBatchArguments(TokenBatchArguments):
    """Extension for TrainingArguments to allow token batching with
    guidance signal."""

    model_guidance_name: str = field(
        default=GUIDANCE_IDS_NAME,
        metadata={"help": "The name of the guidance ids to compute their lengths."},
    )


@dataclass
class HyperparameterSearchArguments:
    do_hyperparameter_search: bool = field(
        default=False, metadata={"help": "Whether to perform a hyperparameter search"}
    )

    hypersearch_space: Dict = field(
        default_factory=lambda: {"num_train_epochs": [2, 3, 4, 5]},
        metadata={
            "help": "The space over \
        which to perform the hyperparameter search."
        },
    )

    hypersearch_trials: int = field(
        default=10,
        metadata={
            "help": "The number of trial runs to test in the hyperparameter search."
        },
    )

    hypersearch_objective: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to optimize in the hyperparameter search."},
    )

    hypersearch_optimize_direction: str = field(
        default="maximize",
        metadata={"help": "Whether to maximize/minimize the hyperparameter objective."},
    )


################################################################################
# Helper Arguments
################################################################################


@dataclass
class PrependPathArguments:
    """Extension to allow to prepend a path to specific directories.
    Useful to switch easly between different working machines."""

    prepend_path: str = field(
        default=None, metadata={"help": "Path to prepend to a number of arguments."}
    )

    def apply_prepend_path(self, other_args):
        """Prepend a specific path to a number of arguments."""

        # If not provided do nothing
        if self.prepend_path is None:
            return

        # Helper func
        def prepend(path):
            return os.path.join(self.prepend_path, path)

        for arg_name in other_args:
            assert hasattr(self, arg_name), "Wrong arguments provided."

            if type(getattr(self, arg_name)) == str:
                # Path, i.e. strings
                setattr(
                    self, arg_name, prepend(self.prepend_path, getattr(self, arg_name))
                )

            elif type(getattr(self, arg_name)) == list and all(
                [type(i) == str for i in getattr(self, arg_name)]
            ):
                # Paths, i.e. list of strings
                setattr(
                    self,
                    arg_name,
                    [prepend(self.prepend_path, i) for i in getattr(self, arg_name)],
                )

            else:
                raise NotImplementedError


@dataclass
class TestingArguments:
    """Extension for testing."""

    testing: bool = field(
        default=False, metadata={"help": "Whether it is a testing run."}
    )


@dataclass
class FileExensions:
    """Extensions to handle data with files of specific file extensions."""

    src_file_extension: str = field(
        default=".src",
        metadata={
            "help": "File extension of src files, in case they are split into files."
        },
    )

    tgt_file_extension: str = field(
        default=".tgt",
        metadata={
            "help": "File extension of tgt files, in case they are split into files."
        },
    )

    guidance_file_extension: str = field(
        default=None,
        metadata={
            "help": "File extension of guidance files, in case they are split into files."
        },
    )


################################################################################
# Main Arguments (train.py)
################################################################################


@dataclass
class SumArguments:
    """General arguments to run the **train.py** python script."""

    metric_loading_script: str = field(
        default="metrics/metric_abs.py",
        metadata={"help": "Path or identifier for the metrics to load."},
    )

    log_to_file: bool = field(
        default=True,
        metadata={
            "help": "Whether to log to a file or output stream. In case \
        of run_as_test_case == True always logs to file."
        },
    )

    logging_level: str = field(default="INFO", metadata={"help": "Level of logging."})

    preprocess_num_proc: int = field(
        default=6, metadata={"help": "Number of processes to use during preprocessing"}
    )

    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the token as HTTP bearer authorization for remote files."
        },
    )

    wandb_project: str = field(
        default="ISum", metadata={"help": "Wandb project to log to"}
    )

    wandb_watch: str = field(
        default="all", metadata={"help": "Weights & biases - what to watch"}
    )

    wandb_dir: str = field(
        default="..", metadata={"help": "Path of local wandb run directory"}
    )

    def __post_init__(self):
        """Post init updates."""

        os.environ["WANDB_PROJECT"] = self.wandb_project
        os.environ["WANDB_WATCH"] = self.wandb_watch
        os.environ["WANDB_DIR"] = ".."


# Attention: should be same as in dataset/load_scripts/base.py -> ConfigNames
class GUIDANCE_OPTIONS(Enum):
    NOT_GUIDED: str = "not_guided"
    SENT_GUIDED: str = "sent_guided"


DATASET_LOAD_ARGS = [
    "data_dir",
    "src_field",
    "tgt_field",
    "src_path_field",
    "start_train_index",
    "start_valid_index",
    "start_test_index",
    "train_size",
    "valid_size",
    "test_size",
    "max_src_sample_len",
    "max_tgt_sample_len",
]


@dataclass
class SumDataArguments(FileExensions, TestingArguments):  # TODO: add prepend
    """Data arguments to run the **train.py** python script."""

    dataset_loading_script: str = field(
        default=None,
        metadata={
            "help": "Path or identifier for the dataset to load. And "
            "possible further kwargs that can be used to load external datasets. "
            "Thus, also accepts List[str]."
        },
    )

    guidance: str = field(
        default="not_guided",
        metadata={
            "help": "Guidance to use.",
            "choices": [e.value for e in GUIDANCE_OPTIONS],
        },
    )

    test_guidance_file: str = field(
        default=None, metadata={"help": "Path to guidance signal for test set."}
    )

    data_dir: str = field(
        default=None, metadata={"help": "Path(s) to the folder with the data"}
    )

    src_field: str = field(
        default="article", metadata={"help": "Name of the src field"}
    )

    tgt_field: str = field(
        default="summary", metadata={"help": "Name of the tgt field"}
    )

    src_path_field: str = field(
        default="article_path", metadata={"help": "Name of the src path field"}
    )

    guidance_field: str = field(
        default="guidance", metadata={"help": "Name of the output field"}
    )

    train_size: int = field(default=None, metadata={"help": "The train dataset size."})

    valid_size: int = field(
        default=None, metadata={"help": "The validation dataset size."}
    )

    test_size: int = field(default=None, metadata={"help": "The test dataset size."})

    start_train_index: float = field(
        default=0,
        metadata={
            "help": "Index from which to consider samples within train "
            "dataset. Absolute or proportional to dataset size"
        },
    )

    start_valid_index: float = field(
        default=0,
        metadata={
            "help": "Index from which to consider samples within dev "
            "dataset. Absolute or proportional to dataset size"
        },
    )

    start_test_index: float = field(
        default=0,
        metadata={
            "help": "Index from which to consider samples within test "
            "dataset. Absolute or proportional to dataset size"
        },
    )

    max_src_sample_len: int = field(
        default=1024, metadata={"help": "Maximal length of source tokens."}
    )

    max_tgt_sample_len: int = field(
        default=1024, metadata={"help": "Maximal length of target tokens."}
    )

    truncation: bool = field(
        default=True, metadata={"help": "Whether to truncate to long examples."}
    )

    filter_truncated_train: bool = field(
        default=False, metadata={"help": "Filter train samples that were truncated."}
    )

    filter_truncated_valid: bool = field(
        default=True,
        metadata={"help": "Filter validations samples that were truncated."},
    )

    filter_truncated_test: bool = field(
        default=True, metadata={"help": "Filter test samples that were truncated."}
    )

    language: str = field(
        default="english", metadata={"help": "Language of the dataset."}
    )

    def __post_init__(self):
        """Post init updates."""

        if self.testing:
            self.train_size = 20
            self.valid_size = 10
            self.test_size = 10

        assert self.guidance in ["not_guided", "sent_guided"], (
            "Please provide correct `guidance` argument. "
            f"Choose between: {[e.value for e in GUIDANCE_OPTIONS]}."
        )

        assert (
            self.dataset_loading_script is not None
        ), "Please provide path to `dataset_loading_script`."


GUIDANCE_MODEL_ARGS = [
    "load_encoder_shared",
    "load_encoder_source",
    "load_encoder_guidance",
    "load_decoder_crossattention_guidance",
    "load_decoder_crossattention_source",
    "source_top_encoder_layer",
    "cross_attn_guidance_first",
    "add_extra_bart_encoder_layers",
]


@dataclass
class AbsSumModelArguments:
    """Model arguments to run the **train.py** python script."""

    pretrained_model_name_or_path: str = field(
        default="facebook/bart-large",
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    load_encoder_shared: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the shared encoder part when using a BART checkpoint "
            "for a GuidedBart model."
        },
    )

    load_encoder_source: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the seperate encoder part for the source sentence "
            "when using a BART checkpoint for a GuidedBart model."
        },
    )

    load_encoder_guidance: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the seperate encoder part for the guidance signal "
            "when using a BART checkpoint for a GuidedBart model."
        },
    )

    load_decoder_crossattention_guidance: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the guidance cross attention part when using "
            "a BART checkpoint for a GuidedBart model."
        },
    )

    load_decoder_crossattention_source: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the source cross attention part when using "
            "a BART checkpoint for a GuidedBart model."
        },
    )

    source_top_encoder_layer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use an extra encoder layer for the source sentence."
        },
    )

    cross_attn_guidance_first: bool = field(
        default=True,
        metadata={
            "help": "Whether to attent to the guidance signal first within the decoder."
        },
    )

    add_extra_bart_encoder_layers: bool = field(
        default=True,
        metadata={
            "help": "Whether to add an extra layer for source/guidance or whether to use "
            "(True) or whether to use the last BART layer as individual "
            "source/guidance layer (False)."
        },
    )

    max_length: Optional[int] = field(
        default=20,
        metadata={
            "help": "Maximum length that will be used by default in the `generate` "
            "method of the model."
        },
    )

    min_length: Optional[int] = field(
        default=10,
        metadata={
            "help": "Minimum length that will be used by default in the `generate` "
            "method of the model."
        },
    )

    num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )

    num_beam_groups: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of groups to divide `num_beams` into in order to "
            "ensure diversity among different groups of beams. "
            "[this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details."
        },
    )

    diversity_penalty: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "This value is subtracted from a beam's score if it generates"
            "a token same as any beam from other group at a particular time."
            "Note that `diversity_penalty` is only effective if `group beam search` is enabled."
        },
    )

    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={
            "help": "he number of independently computed returned sequences"
            "for each element in the batch."
        },
    )

    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length that will be used by default in the "
            "`generate` method of the model."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "Value that will be used by default in the `generate` method of the "
            "model for `no_repeat_ngram_size`. If set to int > 0, all ngrams of "
            "that size can only occur once."
        },
    )

    dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The dropout probability for all fully connected layers in the "
            "embeddings, encoder, and pooler."
        },
    )

    attention_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )

    # classif_dropout: Optional[float] = field(default=0.0, metadata={"help":
    #     "The dropout ratio for classifiers."})

    # activation_dropout: Optional[float] = field(default=0.0, metadata={"help":
    #     "The dropout ratio for activations."})

    # use_cache: Optional[bool] = field(default=True, metadata={"help":
    #     "If set to True, past_key_values key value states are returned and can "
    #     "be used to speed up decoding (see past_key_values)."})


@dataclass
class AbsSumTrainingArguments(
    GuidedTokenBatchArguments,
    TrainMetricsArguments,
    HyperparameterSearchArguments,
    PrependPathArguments,
    Seq2SeqTrainingArguments,
):
    """Training arguments to run the **train.py** python script."""

    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions \
        and checkpoints will be written to."
        },
    )

    extend_dirs_with_run_name: bool = field(
        default=False,
        metadata={
            "help": "Whether to extend output_dir and logging_dir paths with run_name."
        },
    )

    random_run_name: bool = field(
        default=True,
        metadata={
            "help": "Whether to set a random run_name. Can be useful when using wandb "
            "sweeps."
        },
    )

    reshuffle_token_batched_samples: bool = field(
        default=False,
        metadata={
            "help": "Whether to reshuffle batches once token batching is applied."
        },
    )

    early_stopping_patience: int = field(
        default=2,
        metadata={
            "help": "How often stopping threshold has to fail in order to stop training."
        },
    )

    early_stopping_threshold: int = field(
        default=None,
        metadata={
            "help": "Threshold that decides on when to stop after early_stopping_patience "
            "steps. Works together with `greater_is_better`."
        },
    )

    exit_if_early_stop: int = field(
        default=False,
        metadata={
            "help": "Whether to exit script after early stopping - can be used to avoid "
            "long metric computations of eval/test set on bad models."
        },
    )

    def __post_init__(self):
        """Post init updates."""

        if self.load_best_model_at_end:
            assert (
                self.early_stopping_threshold is not None
            ), "Please set `early_stopping_threshold`."

        if not self.do_eval:
            self.evaluation_strategy = "no"
            self.eval_steps = None

        if self.random_run_name:
            self.run_name = (
                "abs-"
                + randomname.get_name()
                + "-"
                + datetime.now().strftime("%Y-%m-%d")
            )

        run_name = self.run_name
        super().__post_init__()
        self.run_name = run_name  # Is modified on __post_init__ if `== None`

        if self.output_dir is None:
            raise ValueError("Output directory is not set. Please set it.")

        if (
            self.run_name is not None
            and self.output_dir is not None
            and self.extend_dirs_with_run_name
        ):
            self.output_dir = os.path.join(self.output_dir, self.run_name)

        if self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "logs")

        self.apply_prepend_path(["output_dir"])


################################################################################
# Extractive Summarization
################################################################################


@dataclass
class ExtSumModelArguments:
    """Model arguments to run the **train.py** python script."""

    pretrained_model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "The model checkpoint for weights initialization."},
    )

    inter_ff_size: int = field(
        default=2048, metadata={"help": "The feed-forward filter size."}
    )

    inter_heads: int = field(
        default=8, metadata={"help": "Number of sentence-level attention heads."}
    )

    inter_layers: int = field(
        default=2, metadata={"help": "Number of sentence-level transformer layers."}
    )

    inter_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio of the sentence-level transformer."},
    )


@dataclass
class ExtSumTrainingArguments(
    TokenBatchArguments,
    TrainMetricsArguments,
    HyperparameterSearchArguments,
    PrependPathArguments,
    TrainingArguments,
):
    """Training arguments to run the **train.py** python script."""

    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions \
        and checkpoints will be written to."
        },
    )

    extend_dirs_with_run_name: bool = field(
        default=False,
        metadata={
            "help": "Whether to extend output_dir and logging_dir paths with run_name."
        },
    )

    random_run_name: bool = field(
        default=True,
        metadata={
            "help": "Whether to set a random run_name. Can be useful when using wandb "
            "sweeps."
        },
    )

    reshuffle_token_batched_samples: bool = field(
        default=False,
        metadata={
            "help": "Whether to reshuffle batches once token batching is applied."
        },
    )

    early_stopping_patience: int = field(
        default=2,
        metadata={
            "help": "How often stopping threshold has to fail in order to stop training."
        },
    )

    early_stopping_threshold: int = field(
        default=None,
        metadata={
            "help": "Threshold that decides on when to stop after early_stopping_patience "
            "steps. Works together with `greater_is_better`."
        },
    )

    exit_if_early_stop: int = field(
        default=False,
        metadata={
            "help": "Whether to exit script after early stopping - can be used to avoid "
            "long metric computations of eval/test set on bad models."
        },
    )

    def __post_init__(self):
        """Post init updates."""

        if self.load_best_model_at_end:
            assert (
                self.early_stopping_threshold is not None
            ), "Please set `early_stopping_threshold`."

        if not self.do_eval:
            self.evaluation_strategy = "no"
            self.eval_steps = None

        if self.random_run_name:
            self.run_name = (
                "ext-"
                + randomname.get_name()
                + "-"
                + datetime.now().strftime("%Y-%m-%d")
            )

        run_name = self.run_name
        super().__post_init__()
        self.run_name = run_name  # Is modified on __post_init__ if `== None`

        if self.output_dir is None:
            raise ValueError("Output directory is not set. Please set it.")

        if (
            self.run_name is not None
            and self.output_dir is not None
            and self.extend_dirs_with_run_name
        ):
            self.output_dir = os.path.join(self.output_dir, self.run_name)

        if self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "logs")

        self.apply_prepend_path(["output_dir"])


################################################################################
# Guidance Extraction
################################################################################


@dataclass
class AbsSumSentenceGuidanceArguments(FileExensions):
    """Arguments to extract sentence guidance with
    `summarization/dataset/guidance_sigmal_sents.py`.
    Can be either from a set of files from directory or single files."""

    src_path: str = field(
        default=None, metadata={"help": "Path to source documents source file/dir."}
    )

    tgt_path: str = field(
        default=None, metadata={"help": "Path to reference summaries tgt file/dir."}
    )

    output_path: str = field(
        default=None, metadata={"help": "Name of the output guidance file/dir."}
    )

    tokenize: bool = field(
        default=False, metadata={"help": "Preprocess sentences with tokenization."}
    )

    guidance_file_extension: str = field(
        default=".sent_guide",
        metadata={
            "help": "File extensions of files, in case src/tgt are split " "into files."
        },
    )

    language: str = field(default="english", metadata={"help": "Language of text."})

    def __post_init__(self):
        """Post init updates."""

        assert self.src_path is not None
        assert self.tgt_path is not None
        assert self.output_path is not None

        assert (os.path.isdir(self.src_path) and os.path.isdir(self.tgt_path)) or (
            ~os.path.isdir(self.src_path) and ~os.path.isdir(self.tgt_path)
        )
