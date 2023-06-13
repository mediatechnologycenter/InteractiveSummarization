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
import json
import logging
import argparse
from argparse import Namespace
from typing import List, Dict, Optional, Union, Tuple

from transformers import Seq2SeqTrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint as get_last_checkpoint_hf

from utils.parser import ArgumentParser
from utils.arguments import AbsSumModelArguments, ExtSumModelArguments
from utils.constants import CONFIG_FILE_NAME, LOG_FILE_NAME

logger = logging.getLogger(__name__)


def get_run_config_from_checkpoint(checkpoint: str) -> str:
    """Returns the config used for the training of a checkpoint.

    Parameters
    ----------
    checkpoint:
        Path to a checkpoint (and it's config)

    Returns
    -------
    Path to the config used for the training of the checkpoint if there is one
    else None.
    """
    path = os.path.join(checkpoint, os.pardir, CONFIG_FILE_NAME)
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    else:
        logger.warning(
            f"Could not find the run_config file of the checkpoint" f" {checkpoint}."
        )
        return None


def is_checkpoint_compatible(
    checkpoint_model_args: Union[AbsSumModelArguments, ExtSumModelArguments],
    model_args: Union[AbsSumModelArguments, ExtSumModelArguments],
    return_uncompatible_args: bool = True,
) -> Union[bool, Tuple[bool, List[str]]]:
    """ Checks whether a checkpoint is compatible with a config instance. To be
    used when loading a pretrained model with new configs. As general rule,
    the model_config cannot be different in this case, as they influence the
    weights, inputs, outputs and/or behavior of the model.
    
    Parameters
    ----------
    checkpoint_model_args:
        Model arguments of the checkpoint
    model_args:
        The model arguments to check them against
    return_uncompatible_args:
        Whether to return the parameters that do not match
    
    Returns
    -------
    Whether the paremeters are compatible
    If `return_uncompatible_args=True` a list of parameters that are not \
    compatbile (if none an empty list)
    """
    checkpoint_dict = vars(checkpoint_model_args)
    args_dict = vars(model_args)
    is_compatible = True
    uncompatible_args = []
    for i in checkpoint_dict:
        is_current_compatible = checkpoint_dict[i] == args_dict[i]
        if not is_current_compatible:
            uncompatible_args.append(i)
        is_compatible = is_compatible and is_current_compatible
    if return_uncompatible_args:
        return is_compatible, uncompatible_args
    else:
        return is_compatible


def verify_checkpoint(
    checkpoint: str, model_args: Union[AbsSumModelArguments, ExtSumModelArguments]
):
    """
    Verifies whether a checkpoint is valid to use. Wrapper for
    `is_checkpoint_compatible` that includes the logic for parsing the
    checkpoint's config and making assertions.

    Parameters
    ----------
    checkpoint:
        Path to a checkpoint (and it's config)
    model_args:
        The model arguments to check them against
    """
    checkpoint_config = get_run_config_from_checkpoint(checkpoint)
    hf_parser = ArgumentParser(([AbsSumModelArguments, ExtSumModelArguments]))
    checkpoint_model_args = hf_parser.parse_json_file(checkpoint_config)[0]
    if checkpoint_model_args:
        is_compatible, uncompatible_args = is_checkpoint_compatible(
            checkpoint_model_args, model_args
        )
        if not is_compatible:
            raise RuntimeError(
                "The checkpoint is not compatible with the"
                "current APE_ARGS. Verify arguments {}.".format(uncompatible_args)
            )
    else:
        raise ValueError(
            f"Could not find {CONFIG_FILE_NAME} in checkpoint" "to check compatibility."
        )


def get_last_checkpoint(
    training_args: Seq2SeqTrainingArguments,
    model_args: Union[AbsSumModelArguments, ExtSumModelArguments] = None,
    verify: bool = True,
) -> str:
    """Checks whether there is a previous checkpoint and returns it. Also
    checks whether no files would be overwritten and whether the checkpoint
    is compatible.

    Parameters
    ----------
    training_args:
        Training arguments
    model_args:
        Model arguments to check possible checkpoint against

    Returns
    -------
    The path to the previous checkpoint if there exists one - else None
    """
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint_hf(training_args.output_dir)
        files = os.listdir(training_args.output_dir)
        if LOG_FILE_NAME in files:
            files.remove(LOG_FILE_NAME)
        if CONFIG_FILE_NAME in files:
            files.remove(CONFIG_FILE_NAME)
        if last_checkpoint is None and len(files) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists "
                "and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    if verify and last_checkpoint is not None:
        assert model_args, "Provide model args to verify."
        verify_checkpoint(checkpoint=last_checkpoint, model_args=model_args)

    return last_checkpoint


def save_config(
    output_dir: str,
    cli_args: Optional[Namespace] = None,
    hypersearch_args: Optional[Dict] = None,
    config_file_name_prepend: str = "",
):
    """Saves the configs to a file.

    Parameters
    ----------
    output_dir:
        Directory in which to save the config file.
    cli_args:
        A Namespace from argparse with cli arguments. Overrides arguments
        from `config_paths`.
    hypersearch_args:
        Best set of parameters found by hyperparameter search
    config_file_name_prepend:
        Prefix to the config file name
    """

    # Parser
    parser = argparse.ArgumentParser(description="config_paths")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        nargs="*",
        help="Cofig file with additional arguments.",
    )
    config_paths = [os.path.abspath(i) for i in parser.parse_known_args()[0].config]

    if config_paths is not None:
        json_objects = []
        for config_path in config_paths:
            json_file = open(config_path, "r")
            json_object = json.load(json_file)
            json_file.close()
            json_objects.append(json_object)
        json_object = json_objects[0]
        for i in json_objects[1:]:
            json_object.update(i)
    else:
        json_object = {}

    if cli_args:
        # CLI args overwrite config args
        args_dict = vars(cli_args)
        for key, obj in args_dict.items():
            json_object[key] = obj

    if hypersearch_args:
        for key, obj in hypersearch_args.items():
            json_object[key] = obj

    save_file = open(
        os.path.join(output_dir, config_file_name_prepend + CONFIG_FILE_NAME), "w"
    )
    json.dump(json_object, save_file, indent=4)
    save_file.close()


def reports_to(tag: str, trainer: Trainer):
    """
    Checks whether a trainer using `training_args` report to specific
    integration.

    Parameters
    ----------
    tag:
        Integration to check (e.g. 'tensorboard' or 'wandb')
    training_args:
        Trainer considered

    Returns
    -------
    A boolean of whether the trainer report to the integration
    """
    return (
        (tag in trainer.args.report_to)
        or (tag == trainer.args.report_to)
        or ("all" == trainer.args.report_to)
        or ("all" in trainer.args.report_to)
    )
