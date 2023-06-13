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
from functools import partial
from argparse import Namespace
from statistics import mean

from datasets import load_dataset, Metric
from transformers.integrations import WandbCallback
from transformers.trainer_callback import ProgressCallback
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer,
    BartTokenizer,
    BartForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    set_seed,
)

from typing import Callable, Optional, Dict

from dataset.tokenize import tokenize_datasets_abs
from dataset.dataset_utils import resize_dataset
from dataset.data_collator import DataCollatorForGuidedSeq2Seq
from dataset.guidance.sents import guidance_sents_extract

from metrics.metric_abs import compute_metrics

from models.guided_bart import GuidedBartForConditionalGeneration
from models.guided_mbart import GuidedMBartForConditionalGeneration

from trainer.trainer import CustomSeq2SeqTrainer
from trainer.train_utils import get_last_checkpoint, save_config, reports_to

from utils.typing import HFDataset
from utils.logger import setup_logging
from utils.parser import ArgumentParser
from utils.misc import read_lines, write_lines
from utils.wandb import update_wandb_config
from utils.fragments import extraction_analysis
from utils.callbacks import (
    TqdmFixProgressCallback,
    CustomWandbCallback,
    CustomEarlyStoppingCallback,
)
from utils.constants import TRAINER_STATE_FILE_NAME, GUIDANCE_IDS_NAME
from utils.arguments import (
    SumArguments,
    SumDataArguments,
    AbsSumModelArguments,
    AbsSumTrainingArguments,
    DATASET_LOAD_ARGS,
    GUIDANCE_OPTIONS,
    GUIDANCE_MODEL_ARGS,
)

logger = logging.getLogger(__name__)


class AbsSumTrainer:
    """Class to train custom summarization models."""

    def __init__(
        self,
        args: SumArguments,
        data_args: SumDataArguments,
        model_args: AbsSumModelArguments,
        training_args: AbsSumTrainingArguments,
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
        self.metric_fn = partial(compute_metrics, tokenizer=self.tokenizer)

        # Trainer
        self.hf_trainer = self.load_hf_trainer()

    def load_datasets(
        self,
        args: Optional[SumArguments] = None,
        data_args: Optional[SumDataArguments] = None,
        model_args: Optional[AbsSumModelArguments] = None,
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
        test_guidance_file = data_args.test_guidance_file

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
            if guidance != GUIDANCE_OPTIONS.NOT_GUIDED.value:
                if guidance == GUIDANCE_OPTIONS.SENT_GUIDED.value:
                    datasets = datasets.map(
                        lambda batch: {
                            data_args.guidance_field: guidance_sents_extract(
                                srcs=batch[data_args.src_field],
                                tgts=batch[data_args.tgt_field],
                                language=data_args.language,
                            )
                        },
                        num_proc=args.preprocess_num_proc,
                        batched=True,
                        load_from_cache_file=data_args.testing,
                    )

        if test_guidance_file is not None:
            guidance_sentences = read_lines(test_guidance_file)
            datasets["test"] = (
                datasets["test"]
                .remove_columns(data_args.guidance_field)
                .add_column(data_args.guidance_field, guidance_sentences)
            )

        # Tokenize datasets
        prefix = (
            f"summarization: "
            if "t5-" in model_args.pretrained_model_name_or_path
            else ""
        )

        tokenized_datasets = tokenize_datasets_abs(
            datasets=datasets,
            tokenizer=tokenizer,
            src_field=data_args.src_field,
            tgt_field=data_args.tgt_field,
            guidance_field=data_args.guidance_field,
            max_src_sample_len=data_args.max_src_sample_len,
            max_tgt_sample_len=data_args.max_tgt_sample_len,
            num_proc=args.preprocess_num_proc,
            prefix=prefix,
            truncation=data_args.truncation,
            filter_truncated_train=data_args.filter_truncated_train,
            filter_truncated_valid=data_args.filter_truncated_valid,
            filter_truncated_test=data_args.filter_truncated_test,
            testing=data_args.testing,
        )

        return datasets, tokenized_datasets

    def load_tokenizer(
        self, pretrained_model_name_or_path: Optional[str] = None, language=None
    ) -> PreTrainedTokenizer:
        """Loads the tokenizer to be used."""

        # Get arguments that are not provided
        if not pretrained_model_name_or_path:
            pretrained_model_name_or_path = (
                self.model_args.pretrained_model_name_or_path
            )
        if not language:
            language = self.data_args.language

        if language == "english":
            tokenizer = BartTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_auth_token=self.args.use_auth_token
            )
        elif language == "german":
            tokenizer = MBartTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                src_lang="de_DE",
                tgt_lang="de_DE",
                use_auth_token=self.args.use_auth_token,
            )
        else:
            raise NotImplementedError
        return tokenizer

    def model_init(
        self,
        args: Optional[SumArguments] = None,
        model_args: Optional[AbsSumModelArguments] = None,
        data_args: Optional[SumDataArguments] = None,
        training_args: Optional[AbsSumTrainingArguments] = None,
    ) -> PreTrainedModel:
        """Loads the model to be used."""

        # Get arguments that are not provided
        if not args:
            args = self.args
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

        if data_args["guidance"] == GUIDANCE_OPTIONS.NOT_GUIDED.value:
            bart_class = (
                BartForConditionalGeneration
                if data_args["language"] == "english"
                else MBartForConditionalGeneration
            )
        else:
            bart_class = (
                GuidedBartForConditionalGeneration
                if data_args["language"] == "english"
                else GuidedMBartForConditionalGeneration
            )

        pretrained_model_name_or_path = model_load_args["pretrained_model_name_or_path"]
        use_auth_token = args["use_auth_token"]
        decoder_start_token_id = (
            self.tokenizer.lang_code_to_id["de_DE"]
            if data_args["language"] == "german"
            else None
        )
        model_load_args["decoder_start_token_id"] = decoder_start_token_id
        if training_args.do_train:
            if data_args["guidance"] == GUIDANCE_OPTIONS.NOT_GUIDED.value:
                return bart_class.from_pretrained(
                    **{
                        k: v
                        for k, v in model_load_args.items()
                        if k not in GUIDANCE_MODEL_ARGS
                    }
                )
            else:
                return bart_class.from_pretrained(**model_load_args)
        else:
            return bart_class.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                use_auth_token=use_auth_token,
            )

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
    ) -> CustomSeq2SeqTrainer:
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
        data_collator = DataCollatorForGuidedSeq2Seq(tokenizer)

        # Init trainer
        logger.info(
            "Initializing Trainer... This might take a while depending "
            "on the size of the datasets."
        )
        # Always use generate for prediction
        training_args.predict_with_generate = True
        hf_trainer = CustomSeq2SeqTrainer(
            args=training_args,
            model_init=model_init,
            data_collator=data_collator,
            compute_metrics=metric_fn,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
        )
        assert hf_trainer.args.predict_with_generate

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
        hf_trainer: Optional[CustomSeq2SeqTrainer] = None,
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
        hf_trainer: Optional[CustomSeq2SeqTrainer] = None,
        training_args: Optional[TrainingArguments] = None,
        model_args: Optional[AbsSumModelArguments] = None,
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
        hf_trainer: Optional[CustomSeq2SeqTrainer] = None,
        data_args: Optional[SumDataArguments] = None,
        model_args: Optional[AbsSumModelArguments] = None,
        training_args: Optional[AbsSumTrainingArguments] = None,
    ):
        """Evaluates a dataset against a model."""

        # Get arguments that are not provided
        if not hf_trainer:
            hf_trainer = self.hf_trainer
        if not data_args:
            data_args = self.data_args
        if not model_args:
            model_args = self.model_args
        if not training_args:
            training_args = self.training_args

        # Get predictions
        predictions = hf_trainer.predict(
            dataset,
            metric_key_prefix=split,
            min_length=model_args.min_length,
            max_length=model_args.max_length,
            num_beams=model_args.num_beams,
            num_beam_groups=model_args.num_beam_groups,
            diversity_penalty=model_args.diversity_penalty,
            num_return_sequences=model_args.num_return_sequences,
        )

        # Log+save metrics
        if predictions.metrics:
            hf_trainer.log_metrics(split, predictions.metrics)
            hf_trainer.save_metrics(split, predictions.metrics)
            if reports_to("wandb", hf_trainer) and wandb.run is not None:
                wandb.log(
                    {k.replace("_", "/", 1): v for k, v in predictions.metrics.items()}
                )

        logger.info(f"Writing results to {training_args.output_dir}")

        # Save predictions
        write_lines(
            trainer.tokenizer.batch_decode(
                predictions.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ),
            os.path.join(training_args.output_dir, f"{split}_predictions.txt"),
        )

        write_lines(
            trainer.tokenizer.batch_decode(
                trainer.tokenized_datasets["test"]["labels"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ),
            os.path.join(training_args.output_dir, f"{split}_references.txt"),
        )

        if GUIDANCE_IDS_NAME in dataset.features:
            write_lines(
                hf_trainer.tokenizer.batch_decode(
                    dataset[GUIDANCE_IDS_NAME],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                os.path.join(hf_trainer.args.output_dir, f"{split}_guidance.txt"),
            )

        return predictions


if __name__ == "__main__":
    # Parser
    hf_parser = ArgumentParser(
        (SumArguments, SumDataArguments, AbsSumModelArguments, AbsSumTrainingArguments)
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
    trainer = AbsSumTrainer(args, data_args, model_args, training_args, cli_args)

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
