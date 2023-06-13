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

import json
import logging
from typing import Any, List, Dict


def read_lines(file_name: str):
    with open(file_name, "r", encoding="utf8") as file:
        lines = file.readlines()
    return [line[:-1] for line in lines]


def write_lines(lines: List[str], file_name: str):
    with open(file_name, "w", encoding="utf8") as file:
        file.write(lines[0].replace("\n", " "))
        for line in lines[1:]:
            file.write("\n" + line.replace("\n", " "))


def dict_to_file(dictionnary: Dict, file_path: str, as_json: bool = True):
    """Stores dict to file."""
    with open(file_path, "w") as file:
        if as_json:
            file.write(json.dumps(dictionnary, indent=4, sort_keys=True))
        else:
            for key, value in sorted(dictionnary.items()):
                file.write(f"{key} = {value}\n")


def log_dict(dictionnary: Dict, info: str):
    """Log content from dict."""
    logging.info(f"***** {info} *****")
    for key, value in sorted(dictionnary.items()):
        logging.info(f"  {key} = {value}")
