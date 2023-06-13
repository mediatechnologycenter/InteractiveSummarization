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
import wandb
import logging
import dataclasses
from optuna import Trial
from functools import partial
from argparse import Namespace
from statistics import mean

from datasets import load_dataset, load_metric, Metric
from torch.nn.functional import softmax
from transformers.integrations import WandbCallback
from transformers.trainer_callback import ProgressCallback
from transformers import (
    BertTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    set_seed,
)

from typing import Callable, Optional, Dict

from dataset.tokenize import tokenize_datasets_ext
from dataset.dataset_utils import resize_dataset
from dataset.data_collator import DataCollatorForSentenceClassification

from models.bert_sum import BertSumExt
from dataset.guidance.sents import guidance_sents_extract, get_extracted_summary

from trainer.trainer import CustomTrainer
from trainer.train_utils import get_last_checkpoint, save_config, reports_to

from utils.typing import HFDataset
from utils.logger import setup_logging
from utils.parser import ArgumentParser
from utils.misc import write_lines
from utils.wandb import update_wandb_config
from utils.fragments import extraction_analysis
from utils.callbacks import (
    TqdmFixProgressCallback,
    CustomWandbCallback,
    CustomEarlyStoppingCallback,
)
from utils.constants import TRAINER_STATE_FILE_NAME, ORACLE_IDS_NAME
from utils.arguments import (
    SumArguments,
    SumDataArguments,
    ExtSumModelArguments,
    ExtSumTrainingArguments,
    DATASET_LOAD_ARGS,
)

logger = logging.getLogger(__name__)


class ExtSumTrainer:
    """Class to train custom summarization models."""

    def __init__(
        self,
        args: SumArguments,
        data_args: SumDataArguments,
        model_args: ExtSumModelArguments,
        training_args: ExtSumTrainingArguments,
        cli_args: Namespace,
    ):
        # Arguments
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.cli_args = cli_args

        # Seed
        set_seed(training_args.seed)

        # Tokenizer
        self.tokenizer = self.load_tokenizer()

        # Data
        self.datasets, self.tokenized_datasets = self.load_datasets()

        # Model init
        self.model_init = partial(
            self.model_init,
            args=args,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
        )

        # Metrics
        self.metric = self.load_metric()
        self.metric_fn = self.get_metric_fn()

        # Trainer
        self.hf_trainer = self.load_hf_trainer()

    def load_datasets(
        self,
        args: Optional[SumArguments] = None,
        data_args: Optional[SumDataArguments] = None,
        model_args: Optional[ExtSumModelArguments] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> HFDataset:
        """Returns the datasets to be used."""

        # Get arguments that are not provided
        if not args:
            args = self.args
        if not data_args:
            data_args = self.data_args
        if not model_args:
            model_args = self.model_args
        if not tokenizer:
            tokenizer = self.tokenizer

        # Prepare loading
        dataset_loading_script = self.data_args.dataset_loading_script
        guidance = data_args.guidance

        # Prepare dataset_loading_script tag
        additional_kwargs = []
        if type(dataset_loading_script) == list:
            additional_kwargs = dataset_loading_script[1:]
            dataset_loading_script = dataset_loading_script[0]

        # Load datasets
        if os.path.isfile(dataset_loading_script):
            # Additional arguments to load local dataset
            load_args = {}
            for arg in DATASET_LOAD_ARGS:
                load_args[arg] = getattr(data_args, arg)

            # Own dataset
            datasets = load_dataset(
                dataset_loading_script, *additional_kwargs, guidance, **load_args
            )
            datasets = resize_dataset(datasets, data_args=data_args)
        else:
            # External dataset
            datasets = load_dataset(dataset_loading_script, *additional_kwargs)
            datasets = resize_dataset(datasets, data_args=data_args)

        datasets = datasets.map(
            lambda batch: {
                ORACLE_IDS_NAME: guidance_sents_extract(
                    srcs=batch[data_args.src_field],
                    tgts=batch[data_args.tgt_field],
                    get_text=False,
                    language=data_args.language,
                )
            },
            num_proc=args.preprocess_num_proc,
            batched=True,
            load_from_cache_file=data_args.testing,
        )

        # Tokenize datasets
        prefix = (
            f"summarization: "
            if "t5-" in model_args.pretrained_model_name_or_path
            else ""
        )

        tokenized_datasets = tokenize_datasets_ext(
            datasets=datasets,
            tokenizer=tokenizer,
            src_field=data_args.src_field,
            oracle_field=ORACLE_IDS_NAME,
            max_length_src=data_args.max_src_sample_len,
            prefix=prefix,
            language=data_args.language,
            num_proc=args.preprocess_num_proc,
            filtering=training_args.do_train,
            filter_truncated_train=data_args.filter_truncated_train,
            filter_truncated_valid=data_args.filter_truncated_valid,
            filter_truncated_test=data_args.filter_truncated_test,
            testing=data_args.testing,
        )

        return datasets, tokenized_datasets

    def load_tokenizer(
        self, pretrained_model_name_or_path: Optional[str] = None
    ) -> PreTrainedTokenizer:
        """Loads the tokenizer to be used."""

        # Get arguments that are not provided
        if not pretrained_model_name_or_path:
            pretrained_model_name_or_path = (
                self.model_args.pretrained_model_name_or_path
            )

        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_auth_token=self.args.use_auth_token
        )
        return tokenizer

    def model_init(
        self,
        trial: Optional[Trial] = None,
        args: Optional[SumArguments] = None,
        model_args: Optional[ExtSumModelArguments] = None,
        data_args: Optional[SumDataArguments] = None,
        training_args: Optional[ExtSumTrainingArguments] = None,
    ) -> PreTrainedModel:
        """Loads the model to be used."""

        # Get arguments that are not provided
        if not model_args:
            model_args = self.model_args
        if not data_args:
            data_args = self.data_args
        if not training_args:
            training_args = self.training_args

        # Reset seed
        set_seed(training_args.seed)

        args = dataclasses.asdict(args)
        model_load_args = dataclasses.asdict(model_args)
        data_args = dataclasses.asdict(data_args)

        if trial is not None:
            trial = trial.params
            for key in trial:
                if key in model_load_args:
                    model_load_args[key] = trial[key]
                if key in data_args:
                    data_args[key] = trial[key]

            # Save config file
            save_config(
                output_dir=training_args.output_dir,
                cli_args=cli_args,
                hypersearch_args=trial.params,
            )

        if training_args.do_train:
            return BertSumExt.from_pretrained(**model_load_args)
        else:
            return BertSumExt.from_pretrained(
                pretrained_model_name_or_path=model_load_args[
                    "pretrained_model_name_or_path"
                ],
                use_auth_token=args["use_auth_token"],
            )

    def load_metric(self, metric_loading_script: Optional[str] = None) -> Metric:
        """Loads the metric to be used."""

        # Get arguments that are not provided
        if not metric_loading_script:
            metric_loading_script = self.args.metric_loading_script

        return load_metric(metric_loading_script)

    def get_metric_fn(
        self,
        metric: Optional[Metric] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> Callable:
        """Prepares the metric function to be used, given the metric obj."""

        # Get arguments that are not provided
        if not metric:
            metric = self.metric
        if not tokenizer:
            tokenizer = self.tokenizer

        return partial(metric.compute, tokenizer=tokenizer, predictions_are_logits=True)

    def load_hf_trainer(
        self,
        args: Optional[SumArguments] = None,
        data_args: Optional[SumDataArguments] = None,
        training_args: Optional[TrainingArguments] = None,
        model_init: Optional[Callable[[Dict], PreTrainedModel]] = None,
        metric_fn: Optional[Callable] = None,
        tokenized_datasets: Optional[HFDataset] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> CustomTrainer:
        """Loads the huggingface internal hf_trainer."""

        # Get arguments that are not provided
        if not args:
            args = self.args
        if not data_args:
            data_args = self.data_args
        if not training_args:
            training_args = self.training_args
        if not model_init:
            model_init = self.model_init
        if not metric_fn:
            metric_fn = self.metric_fn
        if not tokenized_datasets:
            tokenized_datasets = self.tokenized_datasets
        if not tokenizer:
            tokenizer = self.tokenizer

        # Get data collator
        data_collator = DataCollatorForSentenceClassification(tokenizer)

        # Preprocess logits
        def preprocess_logits(logits, labels):
            if type(logits) == tuple:
                logits = logits[0]
            if len(logits.shape) == 3:
                logits = softmax(logits, dim=-1)[:, :, 1]  # TODO: explain
            return logits

            # Init trainer

        logger.info(
            "Initializing Trainer... This might take a while depending "
            "on the size of the datasets."
        )
        hf_trainer = CustomTrainer(
            args=training_args,
            model_init=model_init,
            data_collator=data_collator,
            compute_metrics=metric_fn,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            preprocess_logits_for_metrics=preprocess_logits,
        )
        data_collator.trainer = hf_trainer

        # Update callbacks

        # Replace callback that fixes tqdm / logging
        if hf_trainer.pop_callback(ProgressCallback):
            hf_trainer.add_callback(TqdmFixProgressCallback())

        # Replace wandb logger
        if reports_to("wandb", hf_trainer):
            hf_trainer.remove_callback(WandbCallback)
            hf_trainer.add_callback(
                CustomWandbCallback(
                    after_setup_callbacks=[lambda: update_wandb_config(args, data_args)]
                )
            )

        # Add EarlyStopCallback if load_best_model_at_end
        if training_args.load_best_model_at_end:
            hf_trainer.add_callback(
                CustomEarlyStoppingCallback(
                    training_args.early_stopping_patience,
                    training_args.early_stopping_threshold,
                )
            )

        return hf_trainer

    def hyperparameter_search(
        self,
        hf_trainer: Optional[CustomTrainer] = None,
        training_args: Optional[TrainingArguments] = None,
    ):
        """Hyperparameter search."""
        if not hf_trainer:
            hf_trainer = self.hf_trainer
        if not training_args:
            training_args = self.training_args

        # Search
        hypersearch_training_args = hf_trainer.hyperparameter_search()

        # Set best parameters to training_args & log
        logger.info("***** Hyperparameter best run *****")
        for key, value in hypersearch_training_args.hyperparameters.items():
            setattr(training_args, key, value)
            logger.info(f"  {key} = {value}")

        # Reload trainer with arguments
        self.hf_trainer = self.load_hf_trainer(training_args=training_args)

        return hypersearch_training_args

    def fit(
        self,
        hf_trainer: Optional[CustomTrainer] = None,
        training_args: Optional[TrainingArguments] = None,
        model_args: Optional[ExtSumModelArguments] = None,
        cli_args=None,
        hypersearch_args: Optional[Dict] = None,
    ):
        """Trains and saves config / model / state."""

        # Get arguments if not provided
        if not hf_trainer:
            hf_trainer = self.hf_trainer
        if not training_args:
            training_args = self.training_args
        if not model_args:
            model_args = self.model_args
        if not cli_args:
            cli_args = self.cli_args

        # Get checkpoint if one exists to resume from there
        last_checkpt = get_last_checkpoint(
            training_args=training_args, model_args=model_args, verify=True
        )

        # Save config file
        save_config(
            output_dir=training_args.output_dir,
            cli_args=cli_args,
            hypersearch_args=hypersearch_args,
        )

        # Train
        train_result = self.hf_trainer.train(resume_from_checkpoint=last_checkpt)

        # Saves the tokenizer too for easy upload
        hf_trainer.save_model()

        # Need to save the state, since Trainer.save_model saves only the
        # tokenizer with the model
        hf_trainer.state.save_to_json(
            os.path.join(training_args.output_dir, TRAINER_STATE_FILE_NAME)
        )

        # Save metrics
        hf_trainer.log_metrics("train", train_result.metrics)
        hf_trainer.save_metrics("train", train_result.metrics)

        # Do not predict if early stopped (currently eary stop is done if
        # training stop early because of poor performance)
        if (
            training_args.exit_if_early_stop
            and trainer.hf_trainer.state.global_step
            != trainer.hf_trainer.state.max_steps
        ):
            exit()

        return train_result

    def evaluate(
        self,
        dataset: HFDataset,
        split: str,
        hf_trainer: Optional[CustomTrainer] = None,
        data_args: Optional[ExtSumTrainingArguments] = None,
        training_args: Optional[ExtSumTrainingArguments] = None,
    ):
        """Evaluates a dataset against a model."""

        print("Evaluating Model")

        # Get arguments that are not provided
        if not hf_trainer:
            hf_trainer = self.hf_trainer
        if not data_args:
            data_args = self.data_args
        if not training_args:
            training_args = self.training_args

        # Get predictions
        predictions, label_ids, metrics = hf_trainer.predict(
            dataset, metric_key_prefix=split
        )

        # Added
        dataset_with_predictions = dataset.add_column(
            "predictions", predictions.tolist()
        )
        dataset_with_predictions = dataset_with_predictions.map(
            lambda batch: {
                "extracted_sent": get_extracted_summary(  # TODO: replace keys
                    tgts=batch["filtered_text"],
                    preds=batch["predictions"],
                ),
                **extraction_analysis(
                    source=batch[data_args.src_field],
                    summary=batch[data_args.tgt_field],
                    language=data_args.language,
                    prefix="tgt",
                ),
            },
            num_proc=args.preprocess_num_proc,
            batched=True,
            load_from_cache_file=data_args.testing,
        )

        tgt_coverage = mean(dataset_with_predictions["tgt_coverage"])

        result = self.metric.get_rouge(
            extracted_sent=dataset_with_predictions["extracted_sent"],
            tgt_sum=dataset_with_predictions[data_args.tgt_field],
            metric_key_prefix=split,
        )

        result["test_tgtCoverage"] = tgt_coverage

        # Log+save metrics
        hf_trainer.log_metrics(split, result)
        hf_trainer.save_metrics(split, result)
        if reports_to("wandb", hf_trainer) and wandb.run is not None:
            wandb.log({k.replace("_", "/", 1): v for k, v in result.items()})
        # Save predictions
        write_lines(
            ["<q>".join(sent) for sent in dataset_with_predictions["extracted_sent"]],
            os.path.join(training_args.output_dir, f"{split}_predictions.txt"),
        )

        if data_args.src_path_field in dataset_with_predictions.column_names:
            write_lines(
                dataset_with_predictions[data_args.src_path_field],
                os.path.join(training_args.output_dir, f"{split}_src_paths.txt"),
            )

        write_lines(
            dataset_with_predictions[data_args.tgt_field],
            os.path.join(training_args.output_dir, f"{split}_references.txt"),
        )

        return predictions


if __name__ == "__main__":
    # Parser
    hf_parser = ArgumentParser(
        (SumArguments, SumDataArguments, ExtSumModelArguments, ExtSumTrainingArguments)
    )
    (
        args,
        data_args,
        model_args,
        training_args,
        remaining,
    ) = hf_parser.parse_args_with_json_into_dataclasses_with_default(
        return_remaining_strings=True
    )
    hf_parser.check_no_remaining_args(remaining)
    cli_args = hf_parser.cli_arguments()

    # Logging
    setup_logging(args, training_args)

    # Train class
    trainer = ExtSumTrainer(args, data_args, model_args, training_args, cli_args)

    # Hyperparameter Search
    hypersearch_args = {}
    if training_args.do_hyperparameter_search:
        hypersearch_args = trainer.hyperparameter_search()

    # Train
    if training_args.do_train:
        train_pred = trainer.fit(hypersearch_args=hypersearch_args)

    # Evaluate
    if training_args.do_predict:
        # Final evaluation of test dataset
        test_pred = trainer.evaluate(trainer.tokenized_datasets["test"], "test")
