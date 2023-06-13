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
from typing import Callable, List, Union, Optional


def get_common_files(
    dir1: str, dir2: str, file_extension1: str = "", file_extension2: str = ""
) -> List[str]:
    """Get common files with corresponding file extensions."""
    dir1_files = set([i for i in os.listdir(dir1)])
    dir2_files = set([i for i in os.listdir(dir2)])
    sample_candidates = set([i[: -len(file_extension1)] for i in dir1_files])
    samples = [
        i
        for i in sample_candidates
        if i + file_extension1 in dir1_files and i + file_extension2 in dir2_files
    ]
    samples.sort()
    return samples


def extract_guidance(
    func: Callable[[List[str], List[str]], List[str]],
    src_path: str,
    tgt_path: str,
    output_path: str = None,
    src_file_extension: str = ".src",
    tgt_file_extension: str = ".tgt",
    guidance_file_extension: str = ".guide",
    **kwargs
) -> Optional[Union[str, List[str]]]:
    # Check whether single file or multiple files
    if os.path.isdir(src_path):
        # Get paths
        files = get_common_files(
            dir1=src_path,
            dir2=tgt_path,
            file_extension1=src_file_extension,
            file_extension2=tgt_file_extension,
        )
        out_files = [
            os.path.join(output_path, i + guidance_file_extension) for i in files
        ]

        # Extract guidance signal
        src_files = [os.path.join(src_path, i + src_file_extension) for i in files]
        tgt_files = [os.path.join(tgt_path, i + tgt_file_extension) for i in files]
        srcs = [open(i).read() for i in src_files]
        tgts = [open(i).read() for i in tgt_files]
        outs = func(srcs=srcs, tgts=tgts, **kwargs)

        # Write guidance signal
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            for sample, file in zip(outs, out_files):
                with open(file, "w") as f:
                    f.write(sample)
    else:
        # Extract guidance signal
        srcs = open(src_path).readlines()
        tgts = open(tgt_path).readlines()
        outs = func(srcs=srcs, tgts=tgts, **kwargs)

        # Write guidance signal
        if output_path:
            with open(output_path, "w") as out:
                n_line = False
                for sample in outs:
                    out.write(("\n" if n_line else "") + sample)
                    n_line = True

    return outs
