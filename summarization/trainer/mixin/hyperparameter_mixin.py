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

from transformers.utils import logging

logger = logging.get_logger(__name__)


class HyperParameterSearchMixin:
    def hyperparameter_search(self):
        # Define search space
        def optuna_hp_space(trial):
            # Alternatives to suggest_categorical can be found at:
            # https://optuna.readthedocs.io/en/stable/reference/
            # multi_objective/generated/
            # optuna.multi_objective.trial.MultiObjectiveTrial.html?
            # highlight=suggest#optuna.multi_objective.trial.
            # MultiObjectiveTrial.suggest_categorical
            assert all(
                type(item) == list for item in self.args.hypersearch_space.values()
            ), (
                "Hyperparameter space should be defined as a dict" " of names to lists."
            )
            hp_space = {
                key: trial.suggest_categorical(key, value)
                for key, value in self.args.hypersearch_space.items()
            }
            return hp_space

        # Objective
        obj = self.args.hypersearch_objective

        def objective(obj_metrics):
            if obj:
                if obj in obj_metrics:
                    return obj_metrics[obj]
                elif "eval_" + obj in obj_metrics:
                    return obj_metrics["eval_" + obj]
            else:
                return None
            raise RuntimeError(
                "Could not determine objective for " "hyperparameter search"
            )

        # Perform hyperparameter search
        best_run = super().hyperparameter_search(
            hp_space=optuna_hp_space,
            compute_objective=objective,
            n_trials=self.args.hypersearch_trials,
            backend="optuna",
            direction=self.args.hypersearch_optimize_direction,
        )

        return best_run.hyperparameters
