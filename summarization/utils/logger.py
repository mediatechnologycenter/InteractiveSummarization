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
import sys
import logging

import tqdm
import transformers
from transformers.trainer_utils import is_main_process

from typing import Union

from utils.constants import LOG_FILE_NAME
from utils.arguments import (
    SumArguments,
    AbsSumTrainingArguments,
    ExtSumTrainingArguments,
)

logger = logging.getLogger(__name__)
# LOG_FORMAT = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s,%(msecs)d >> %(message)s"
LOG_FORMAT = "[%(levelname)s|%(filename)s:%(lineno)d] %(asctime)s >> %(message)s"
DATE_FORMAT = "%d-%m-%Y %H:%M:%S"


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    args: SumArguments,
    training_args: Union[AbsSumTrainingArguments, ExtSumTrainingArguments],
) -> logging.Logger:
    """Initiates a logger.

    Parameters
    ----------
    training_args:
        Training arguments that provide certain arguments to the logger setup.

    Returns
    -------
    A logger instance.
    """
    if args.log_to_file:
        if not os.path.isdir(training_args.output_dir):
            os.makedirs(training_args.output_dir)

        filename = os.path.join(training_args.output_dir, LOG_FILE_NAME)
        logging.basicConfig(
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            handlers=[logging.FileHandler(filename, mode="w"), TqdmLoggingHandler()],
            level=logging.getLevelName(args.logging_level),
        )
    else:
        logging.basicConfig(
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            handlers=[TqdmLoggingHandler()],
            level=logging.getLevelName(args.logging_level),
        )

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, \
          device: {training_args.device}, n_gpu: {training_args.n_gpu}, \
          distributed training: {bool(training_args.local_rank != -1)}, \
          16-bits training: {training_args.fp16}"
    )

    # Set the verbosity of the logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(
            logging.getLevelName(args.logging_level)
        )
        transformers.utils.logging.disable_default_handler()
        transformers.utils.logging.enable_propagation()
        transformers.utils.logging.enable_explicit_format()

    return logger
