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
import datasets

from torch.utils.data import DataLoader

from transformers.tokenization_utils_base import BatchEncoding
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard

from typing import Optional, List

from utils.typing import HFDataset
from dataset.sampler import TokenBatchSampler


class TokenBatchMixin:
    """Enables smart batching for train dataset."""

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement
        :obj:`__len__`, a random sampler (adapted to distributed training if
        necessary) otherwise.

        Subclass and override this method if you want to inject some
        custom behavior.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        if not self.args.token_batching:
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            batch_train_sampler = self._get_batch_train_sampler(train_sampler)
            return DataLoader(
                train_dataset,
                batch_sampler=batch_train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def _get_feature_length(
        self, dataset: HFDataset, column_name: str, length_column_name: str
    ):
        """Get the lengths of a given feature for each sample in a dataset."""

        if (
            is_datasets_available()
            and isinstance(dataset, datasets.Dataset)
            and length_column_name in dataset.column_names
        ):
            lengths = dataset[length_column_name]

        else:
            if (
                not (
                    isinstance(dataset[0], dict)
                    or isinstance(dataset[0], BatchEncoding)
                )
                or column_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose "
                    "items are dictionaries with an "
                    f"'{column_name}' key."
                )
            lengths = [len(feature[column_name]) for feature in dataset]

        return lengths

    def _get_input_output_lengths(
        self, train_sampler: Optional[torch.utils.data.Sampler]
    ) -> List[int]:
        """Return the length of each input_id & label."""

        if hasattr(train_sampler, "lengths"):
            lengths = train_sampler.lengths

        else:
            model_input_name = (
                self.tokenizer.model_input_names[0]
                if self.tokenizer is not None
                else "input_ids"
            )
            lengths = self._get_feature_length(
                dataset=self.train_dataset,
                column_name=model_input_name,
                length_column_name=self.args.length_column_name,
            )

        if not self.args.token_batching_with_output:
            output_lengths = [0 for _ in range(len(lengths))]

        else:
            output_lengths = self._get_feature_length(
                dataset=self.train_dataset,
                column_name=self.args.model_output_name,
                length_column_name=self.args.output_length_column_name,
            )

        return [lengths, output_lengths]

    def _get_batch_train_sampler(
        self, train_sampler: Optional[torch.utils.data.Sampler]
    ) -> TokenBatchSampler:
        """Get a train batch sampler that batches by token and not samples."""

        lengths = self._get_input_output_lengths(train_sampler=train_sampler)
        return TokenBatchSampler(
            sampler=train_sampler,
            lengths=lengths,
            batch_size=self.args.train_batch_size,
            token_batching=self.args.token_batching,
            batch_size_includes_padding=self.args.batch_size_includes_padding,
            precompute=True,
            reshuffle_token_batched_samples=self.args.reshuffle_token_batched_samples,
        )


class GuidedTokenBatchMixin(TokenBatchMixin):
    def _get_batch_train_sampler(
        self, train_sampler: Optional[torch.utils.data.Sampler]
    ) -> TokenBatchSampler:
        """Get a train batch sampler that batches by token and not samples."""

        lengths = self._get_input_output_lengths(train_sampler=train_sampler)

        if self.args.model_guidance_name in self.train_dataset.features:
            guidance_lengths = self._get_feature_length(
                dataset=self.train_dataset,
                column_name=self.args.model_guidance_name,
                length_column_name=self.args.length_column_name,
            )
            lengths.append(guidance_lengths)

        return TokenBatchSampler(
            sampler=train_sampler,
            lengths=lengths,
            batch_size=self.args.train_batch_size,
            token_batching=self.args.token_batching,
            batch_size_includes_padding=self.args.batch_size_includes_padding,
            precompute=True,
            reshuffle_token_batched_samples=self.args.reshuffle_token_batched_samples,
        )
