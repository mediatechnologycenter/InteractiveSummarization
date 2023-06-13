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

import logging
from typing import List, Callable

from transformers.integrations import WandbCallback
from transformers.trainer_callback import ProgressCallback, EarlyStoppingCallback


class TqdmFixProgressCallback(ProgressCallback):
    """A callback that fixes certain visulaization / logging with tqdm."""

    def __init__(self):
        super().__init__()

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Fix to show tqdm output in logger"""
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            # self.training_bar.write(str(logs))
            logging.info(str(logs))

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        """Fix for tqdm bat when running multiple times hf_trainer.predict"""
        if (
            self.prediction_bar is not None
            and self.prediction_bar.n == self.prediction_bar.total
        ):
            self.prediction_bar.close()
            self.prediction_bar = None
        super().on_prediction_step(
            args, state, control, eval_dataloader=eval_dataloader, **kwargs
        )


class CustomWandbCallback(WandbCallback):
    """Extended WandbCallback to log further informations."""

    def __init__(self, after_setup_callbacks: List[Callable] = []):
        super().__init__()
        self.after_setup_callbacks = after_setup_callbacks

    def setup(self, args, state, model, **kwargs):
        super().setup(args=args, state=state, model=model, **kwargs)
        for f in self.after_setup_callbacks:
            f()
        self.after_setup_callbacks = []


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        if (
            metric_value >= self.early_stopping_threshold
            if args.greater_is_better
            else metric_value <= self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
