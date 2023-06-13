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

""" Data sampler, data collator and tokenize methods. """

import torch
import random
import logging
import numpy as np

from typing import List

logger = logging.getLogger(__name__)


class TokenBatchSampler:

    """Data sampler that allows to batch by tokens, instead of sentences."""

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        lengths: List[int],
        batch_size: int = 4,
        token_batching: bool = False,
        batch_size_includes_padding: bool = True,
        precompute: bool = True,
        reshuffle_token_batched_samples: bool = False,
    ):
        self.sampler = sampler
        self.lengths = lengths
        self.batch_size = batch_size
        self.token_batching = token_batching
        self.batch_size_includes_padding = batch_size_includes_padding
        self.reshuffle_token_batched_samples = reshuffle_token_batched_samples
        self.batches = None
        self.n_batches = None

        # Precompute
        if precompute:
            self.batches = self.__make_batches(
                sampler, batch_size, token_batching, batch_size_includes_padding
            )
            self.n_batches = len(self.batches)
            assert (self.n_batches) > 0, (
                "Please choose bigger batch_size. "
                + f"No batches could be formed with batch size {self.batch_size}."
            )

    def __len__(self):
        """Get amount of batches."""
        if not hasattr(self, "length") or self.n_batches is None:
            if hasattr(self, "batches"):
                self.n_batches = len(self.batches)
            else:
                self.n_batches = sum(1 for _ in iter(self))
        return self.n_batches

    def __iter__(self):
        """Iterate over all batches."""

        if not hasattr(self, "batches") or self.batches is None:
            self.batches = self.__make_batches(
                self.sampler,
                self.batch_size,
                self.token_batching,
                self.batch_size_includes_padding,
            )
        return iter(self.batches)

    def __make_batches(
        self,
        sampler: torch.utils.data.Sampler,
        batch_size: int = 4,
        token_batching: bool = False,
        batch_size_includes_padding: bool = True,
    ):
        """
        Creates smart batches. In order to optimize the use provide a
        sampler that gets samples of similar length.
        """

        batches = []
        if not token_batching:
            batch = []
            for sample in sampler:
                batch.append(sample)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []

            if len(batch) != 0:
                batches.append(batch)

        else:
            indices = list(iter(sampler))
            lens = [np.array(i)[indices] for i in self.lengths]
            batch = []
            ignored = 0

            if batch_size_includes_padding:
                max_len = [0 for _ in range(len(lens))]
                els = 0

                for el in zip(*lens, indices):
                    l = el[:-1]
                    sample = el[-1]

                    len_if_added = 0
                    for new_l, max_l in zip(l, max_len):
                        len_if_added += max(new_l, max_l) * (els + 1)

                    if sum(l) > batch_size:
                        ignored += 1
                        continue  # Ignore

                    elif len_if_added <= batch_size:
                        batch.append(sample)
                        new_max_len = []
                        for new_l, max_l in zip(l, max_len):
                            new_max_len.append(max(new_l, max_l))
                        max_len = new_max_len
                        els += 1
                    else:
                        batches.append(batch)
                        batch = [sample]
                        max_len = list(l)
                        els = 1

                if len(batch) != 0:
                    batches.append(batch)

            else:
                curr_len = 0

                for el in zip(*lens, sampler):
                    l = el[:-1]
                    sample = el[-1]

                    if sum(l) > batch_size:
                        ignored += 1
                        continue  # Ignore
                    elif sum(l) + curr_len <= batch_size:
                        batch.append(sample)
                        curr_len += sum(l)
                    else:
                        batches.append(batch)
                        batch = [sample]
                        curr_len = sum(l)

                if len(batch) != 0:
                    batches.append(batch)

            logger.info(
                f"Ignored {ignored}/{len(indices)} samples, as they "
                f"are too long for the current batch size "
                f"({self.batch_size})."
            )

        if self.reshuffle_token_batched_samples:
            random.shuffle(batches)

        return batches
