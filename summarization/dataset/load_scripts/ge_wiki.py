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

""" APE dataset. """

import os
import datasets
from dataclasses import dataclass

from .base import GuidanceConfig, BaseBuilder, ConfigNames

logger = datasets.logging.get_logger(__name__)

_CITATION = """ TODO """
_DESCRIPTION = """ "German Wikipedia Summarization Dataset" """
HOMEPAGE = "TODO"
SEED = 2022


@dataclass
class GeWikiConfig(GuidanceConfig):
    pass


class GeWiki(BaseBuilder, datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = GeWikiConfig

    NOT_GUIDED = GeWikiConfig(
        name=ConfigNames.NOT_GUIDED.value,
        version=datasets.Version("1.0.0", ""),
        description="German Wiki Dataset without guidance.",
    )

    SENT_GUIDED = GeWikiConfig(
        name=ConfigNames.SENT_GUIDED.value,
        version=datasets.Version("1.0.0", ""),
        description="German Wiki Dataset with sentence guidance.",
    )

    BUILDER_CONFIGS = [NOT_GUIDED, SENT_GUIDED]

    def _info(self):
        """Sets the own info based on self.config.name."""

        assert self.config.data_dir, "Please specify data path."

        if self.config.name == ConfigNames.NOT_GUIDED.value:
            features = datasets.Features(
                {
                    self.config.src_field: datasets.Value("string"),
                    self.config.tgt_field: datasets.Value("string"),
                }
            )
        else:
            features = datasets.Features(
                {
                    self.config.src_field: datasets.Value("string"),
                    self.config.tgt_field: datasets.Value("string"),
                    self.config.guidance_field: datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Creates the different dataset split for train/validation/test."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(self.config.data_dir, "train"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(self.config.data_dir, "eval"),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(self.config.data_dir, "test"),
                    "split": "test",
                },
            ),
        ]

    def _generate_guidances(self, samples):
        """Method that retreives either from file with extensions or computes
        it on the go."""

        try:
            guide = self.config.guidance_file_extension
            return [self.read_file(i + guide) for i in samples]
        except:
            if self.config.name == ConfigNames.SENT_GUIDED.value:
                import sys

                sys.path.append("dataset")
                sys.path.append("dataset/guidance")
                from guidance.sents import guidance_sents_extract
            else:
                raise NotImplementedError

            srcs = [self.read_file(i + ".src") for i in samples]
            tgts = [self.read_file(i + ".tgt") for i in samples]
            return guidance_sents_extract(srcs=srcs, tgts=tgts)

    def _generate_examples(self, path, split):
        """Yields samples"""

        files = set([i for i in os.listdir(path)])
        sample_files_canididates = set([i[:-4] for i in files])
        sample_files = [
            i
            for i in sample_files_canididates
            if i + ".src" in files and i + ".tgt" in files
        ]
        sample_files = [os.path.join(path, i) for i in sample_files]
        sample_files.sort()

        if self.config.name == ConfigNames.NOT_GUIDED.value:

            def samples_iter():
                for ex in sample_files:
                    yield {
                        self.config.src_field: self.read_file(ex + ".src"),
                        self.config.tgt_field: self.read_file(ex + ".tgt"),
                    }

        else:

            def samples_iter():
                for ex in sample_files:
                    yield {
                        self.config.src_field: self.read_file(ex + ".src"),
                        self.config.tgt_field: self.read_file(ex + ".tgt"),
                        self.config.guidance_field: self._generate_guidances([ex])[
                            0
                        ].strip(),
                    }

        samples = samples_iter()
        for i in super()._generate_examples(samples=samples, split=split):
            yield i
