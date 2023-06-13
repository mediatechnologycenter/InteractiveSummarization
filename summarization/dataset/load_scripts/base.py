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

import datasets
from enum import Enum
from dataclasses import dataclass

from typing import Union, Optional, Iterable, Dict


class ConfigNames(Enum):
    NOT_GUIDED = "not_guided"
    SENT_GUIDED = "sent_guided"


@dataclass
class BaseConfig(datasets.BuilderConfig):
    _id_counter: int = 0

    data_dir: Optional[str] = None

    src_file_extension: str = ".src"
    tgt_file_extension: str = ".tgt"

    src_field: str = "article"
    tgt_field: str = "summary"
    src_path_field: str = "article_paths"

    start_train_index: Union[int, float] = 0
    start_valid_index: Union[int, float] = 0
    start_test_index: Union[int, float] = 0

    train_size: Optional[Union[int, float]] = None
    valid_size: Optional[Union[int, float]] = None
    test_size: Optional[Union[int, float]] = None

    testing: Optional[bool] = None
    max_src_sample_len: Optional[int] = None
    max_tgt_sample_len: Optional[int] = None

    def get_max_size(self, split):
        if split == "train":
            return self.train_size
        elif split == "dev":
            return self.valid_size
        elif split == "test":
            return self.test_size
        else:
            raise NotImplementedError

    def get_start_index(self, split):
        if split == "train":
            return self.start_train_index
        elif split == "dev":
            return self.start_valid_index
        elif split == "test":
            return self.start_test_index
        else:
            raise NotImplementedError


@dataclass
class GuidanceConfig(BaseConfig):
    guidance_file_extension: str = None
    guidance_field: Optional[str] = "guidance"

    def __post_init__(self):
        super().__post_init__()
        if self.name == "default":
            self.name = ConfigNames.NOT_GUIDED.name


class BaseBuilder:
    def read_file(self, path: str, safe=False):
        if safe:
            try:
                return open(path, "r").read().strip()
            except:
                return None
        else:
            return open(path, "r").read().strip()

    def _generate_examples(self, samples: Iterable[Dict], split: str):
        """Yields samples"""

        # Start index
        start_index = self.config.get_start_index(split)
        assert start_index >= 0
        if (0.0 < start_index) and (start_index < 1.0):
            start_index = int(len(samples) * start_index)

        # Max samples
        size = self.config.get_max_size(split)
        if size and size < 0.0:
            size = None
        elif size and (0.0 < size) and (size < 1.0):
            size = int(len(samples) * size)

        for i, sample in enumerate(samples):
            # Skip until start_index
            if i < start_index:
                continue

            yield self.config._id_counter, sample

            self.config._id_counter += 1

            # Stop if size reached
            if size and i == size + start_index - 1:
                break
