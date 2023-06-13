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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_torch_tpu_available
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_init
from transformers.trainer_pt_utils import nested_concat, nested_numpify, find_batch_size
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput, has_length
from transformers.utils import logging
from typing import Union, Any, Optional, Tuple, Dict, List

from trainer.mixin.token_batch_mixin import TokenBatchMixin, GuidedTokenBatchMixin
from trainer.mixin.train_metrics_mixin import TrainMetricsMixin
from trainer.mixin.hyperparameter_mixin import HyperParameterSearchMixin
from utils.constants import GUIDANCE_IDS_NAME, GUIDANCE_ATTENTION_MASK_NAME

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(GuidedTokenBatchMixin, HyperParameterSearchMixin, Trainer):
    """HuggingFace trainer extended with the possibility of Token batching."""

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"]
            if gen_kwargs.get("max_length") is not None
            else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            gen_kwargs:
                Additional `generate` specific kwargs.
        <Tip>
        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"]
            if gen_kwargs.get("max_length") is not None
            else self.args.generation_max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        if gen_kwargs.get("num_return_sequences") > 1:
            test_dataloader = self.get_test_dataloader(test_dataset)
            output = self.multiple_prediction_loop(
                test_dataloader, description="Prediction", ignore_keys=ignore_keys
            )
            return PredictionOutput(
                predictions=output.predictions, label_ids=output.label_ids, metrics=None
            )

        return super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Check HuggingFace prediction_step for parameters."""

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"]
            if gen_kwargs.get("max_length") is not None
            else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        gen_kwargs["num_beam_groups"] = (
            gen_kwargs["num_beam_groups"]
            if gen_kwargs.get("num_beam_groups") is not None
            else self.model.config.num_beam_groups
        )
        gen_kwargs["diversity_penalty"] = (
            gen_kwargs["diversity_penalty"]
            if gen_kwargs.get("diversity_penalty") is not None
            else self.model.config.diversity_penalty
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )
        if GUIDANCE_IDS_NAME in inputs:
            gen_kwargs[GUIDANCE_IDS_NAME] = inputs.get(GUIDANCE_IDS_NAME, None)
        if GUIDANCE_ATTENTION_MASK_NAME in inputs:
            gen_kwargs[GUIDANCE_ATTENTION_MASK_NAME] = inputs.get(
                GUIDANCE_ATTENTION_MASK_NAME, None
            )
        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should
        # be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def multiple_prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        ignore_keys: Optional[List[str]] = None,
    ) -> EvalLoopOutput:
        """
        Prediction loop for multiple output sequences
        Works without labels and does not compute scores
        """
        args = self.args

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted
        if args.fp16_full_eval:
            model = model.to(dtype=torch.float16, device=args.device)
        elif args.bf16_full_eval:
            model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(
                args.device
            )

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # preds on GPU/TPU (accumulated for eval_accumulation_steps)
        preds_host = None

        # preds on CPU (final containers)
        all_preds = None

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            _, logits, _ = self.prediction_step(
                model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
            )

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                preds_host = None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )

        return EvalLoopOutput(
            predictions=all_preds, label_ids=None, metrics=None, num_samples=None
        )

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


def get_noam_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / (float(max(1, num_warmup_steps)) ** 1.5)
        return float(current_step) ** (-0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomTrainer(
    TokenBatchMixin, TrainMetricsMixin, HyperParameterSearchMixin, Trainer
):

    """Trainer with a custom learning rate scheduler."""

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            print("Using custom scheduler: Noam Decay")
            self.lr_scheduler = get_noam_schedule(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler
