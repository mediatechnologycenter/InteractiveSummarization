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

import math
import numpy as np

from transformers import TrainerCallback, IntervalStrategy
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_pt_utils import nested_concat, nested_numpify

from typing import Callable, Optional, Dict


class TrainMetricsCallback(TrainerCallback):
    def __init__(
        self, train_metric_samples: int, compute_metrics: Callable, log_fn: Callable
    ):
        # Local variables
        self.remaining_savings, self.last_compute = 0, 0
        self.train_metric_samples = train_metric_samples
        self.compute_metrics = compute_metrics
        self.log_fn = log_fn
        self.reset_buffers()

    def reset_buffers(self):
        """Resets buffers in which the labels / predictions are stored."""
        self.preds_buffer = None  # deque([], train_metric_samples)
        self.labels_buffer = None
        self.inputs_buffer = None

    def is_save_state(self):
        """Saves labels / predictions for future metrics computation."""
        return self.remaining_savings >= 1

    def save_step(
        self, logits: np.array, labels: np.array, inputs: Optional[np.array] = None
    ):
        """Saves logits and labels for a step."""

        # Save logits
        self.preds_buffer = (
            logits
            if self.preds_buffer is None
            else nested_concat(self.preds_buffer, logits, padding_index=-100)
        )
        self.preds_buffer = self.preds_buffer[-self.train_metric_samples :]

        # Save labels
        self.labels_buffer = (
            labels
            if self.labels_buffer is None
            else nested_concat(self.labels_buffer, labels, padding_index=-100)
        )
        self.labels_buffer = self.labels_buffer[-self.train_metric_samples :]

        # Save inputs
        if self.inputs_buffer:
            self.inputs_buffer = (
                inputs
                if self.inputs_buffer is None
                else nested_concat(self.inputs_buffer, inputs, padding_index=-100)
            )
            self.inputs_buffer = self.inputs_buffer[-self.train_metric_samples :]

        # Update last saved
        self.remaining_savings -= 1

    def get_eval_prediction(self):
        """Get EvalPrediction object to be used to compute metrics."""

        # Return EvalPrediction object
        if self.inputs_buffer is None:
            return EvalPrediction(
                predictions=self.preds_buffer, label_ids=self.labels_buffer
            )
        else:
            return EvalPrediction(
                predictions=self.preds_buffer,
                label_ids=self.labels_buffer,
                inputs=self.inputs_buffer,
            )

    def _log_train_metrics(self, metrics: Dict):
        """Logs the train metrics."""
        self.log_fn(metrics)

    def on_step_begin(self, args, state, control, logs=None, **kwargs):
        """Update the current status."""

        # With no evaluation skip
        if args.evaluation_strategy == IntervalStrategy.NO:
            return
        elif args.evaluation_strategy == IntervalStrategy.STEPS:
            eval_steps = args.eval_steps
        elif args.evaluation_strategy == IntervalStrategy.EPOCH:
            eval_steps = state.max_steps // state.num_train_epochs
        else:
            raise NotImplementedError

        collect_iters = math.ceil(
            self.train_metric_samples / args.gradient_accumulation_steps
        )

        # Save such that we have at least train_metric_samples samples to
        # evalaute at computation step
        if (state.global_step + collect_iters) % eval_steps < collect_iters:
            self.remaining_savings = args.gradient_accumulation_steps

        # Single computation step
        if (
            (state.global_step % eval_steps == 0 and state.global_step != 0)
            or (
                state.global_step == state.max_steps - 1
                and (state.global_step + 1) % eval_steps == 0
            )
        ) and self.last_compute != state.global_step:
            # Check it collected enough samples
            if self.labels_buffer is None or len(self.labels_buffer) == 0:
                return

            # Metrics
            eval_prediction_obj = self.get_eval_prediction()
            train_metrics = self.compute_metrics(eval_prediction_obj)

            # Log
            self._log_train_metrics(train_metrics)
            self.last_compute = state.global_step


class TrainMetricsMixin:
    """Enables computation of train metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if (
            self.args.train_metric_samples
            and self.args.evaluation_strategy != IntervalStrategy.NO
        ):
            # Init required callback that saves train labels / predicitons
            self.tm_callback = TrainMetricsCallback(
                train_metric_samples=self.args.train_metric_samples,
                compute_metrics=self.compute_metrics,
                log_fn=self.log,
            )

            # Register callback
            self.add_callback(self.tm_callback)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Keeps track of train metrics when computing the loss."""

        # Retreive labels
        labels = None
        if (
            self.args.train_metric_samples
            and self.args.evaluation_strategy != IntervalStrategy.NO
        ):
            if self.tm_callback.is_save_state() and "labels" in inputs:
                # Prepare labels
                labels = inputs["labels"].detach()
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels = nested_numpify(labels)

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Save train samples
        if (
            self.args.train_metric_samples
            and self.args.evaluation_strategy != IntervalStrategy.NO
        ):
            if self.tm_callback.is_save_state() and labels is not None:
                # Prepare logits
                logits = outputs.logits.detach()
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = nested_numpify(logits)

                # Prepare inputs
                inputs = None
                if (
                    hasattr(self, "include_inputs_for_metrics")
                    and self.args.include_inputs_for_metrics
                ):
                    inputs = inputs["inputs_ids"].detach()
                    inputs = self._pad_across_processes(inputs)
                    inputs = self._nested_gather(inputs)
                    inputs = nested_numpify(inputs)

                self.tm_callback.save_step(logits=logits, labels=labels, inputs=inputs)

        return (loss, outputs) if return_outputs else loss
