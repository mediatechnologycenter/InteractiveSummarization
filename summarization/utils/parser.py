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
import json
import dataclasses
import argparse
from pathlib import Path

from typing import Tuple, List
from argparse import Namespace, SUPPRESS, _UNRECOGNIZED_ARGS_ATTR, ArgumentError
from transformers.hf_argparser import HfArgumentParser, DataClass

"""

Main method: `parse_args_with_json_into_dataclasses_with_default`

Arguments can be provided through one or several .json config files - where
latter ones overwrite values of earlier ones. In addition command line arguments
can be provided, which have the priority on all given arguments
(cli/json/default). Entries (keys) starting with `_` in the json are ignored
and can be used as comments within a json.

The order of priority in which the provided arguments are considered is:

- cli arguments
- last config file
- ...
- first config file
- dataclass defaults

#### Example json config files

(example1.json)

{
    src_lang: "it"
}

(example2.json)

{
    input_file: "hi.txt",
    output_file: "bye.txt"
}

#### Example command

python3 code.py --config example1.json example2.json --tgt_lang ch --input_file "hello.txt"

#### Example Code

@dataclass
class DataclassArguments1:
    model: str = field(default="bert", metadata={"help": "Model"})
    src_lang: str = field(default="fr", metadata={"help": "Source language"})
    tgt_lang: str = field(default="en", metadata={"help": "Target language"})

@dataclass
class DataclassArguments2:
    input_file: str = field(default="in.txt", metadata={"help": "Input_file"})
    output_file: str = field(default="out.txt", metadata={"help": "Output file"})

parser = ArgumentParser((DataclassArguments1, DataclassArguments2))
args1, args2, remaining = parser.parse_args_with_json_into_dataclasses_with_default(
                            return_remaining_strings=True)

print(args1.model, args1.src_lang, args2.src_lang)
print(args2.input_file, args2.output_file)

#### Prints

bert it ch
hello.txt bye.txt

"""


class ArgumentParser(HfArgumentParser):
    """Extensions to the Huggingface **HfArgumentParser** parser. It allows
    for multiple config files, where the latter ones overwrite the previous
    ones.
    """

    def parse_known_args(
        self,
        args: List[str] = None,
        namespace: Namespace = None,
        with_default: bool = True,
        ignore_required: bool = False,
    ):
        """Based on parse_known_args `ArgumentParser.parse_known_args`.
        In addition to the initial method it allow for ignoring default
        values - i.e. if the default value nothing is set instead of the
        default value.

        Parameters
        ----------
        args:
            The arguments.
        namespace:
            The namespace to store the parsed arguments to.
        with_deafult:
            Whether to set default values if these are not present.
        ignore_required:
            Whether to ignore if fields are required.
        """
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        if with_default:
            for action in self._actions:
                if action.dest is not SUPPRESS:
                    if not hasattr(namespace, action.dest):
                        if action.default is not SUPPRESS:
                            setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        if with_default:
            for dest in self._defaults:
                if not hasattr(namespace, dest):
                    setattr(namespace, dest, self._defaults[dest])

        to_reset = []
        for action in self._actions:
            if hasattr(namespace, action.dest) or ignore_required:
                to_reset.append((action, action.required))
                action.required = False

        # parse the arguments and exit if there are any errors
        try:
            namespace, args = self._parse_known_args(args, namespace)
            if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
                args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
                delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
            return namespace, args
        except ArgumentError:
            err = sys.exc_info()[1]
            self.error(str(err))
        finally:
            for act, req in to_reset:
                act.required = req

    def parse_args_into_dataclasses_with_default(
        self,
        args: List[str] = None,
        return_remaining_strings: bool = False,
        json_default_files: List[str] = None,
        check_no_remaining_args: bool = False,
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`.
        See the docat:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        This method combines parse_args_into_dataclasses and parse_json_file
        from the Huggingface parser and allow additionaly to provide several
        json files.

        The order of priority in which the provided arguments are considered is:

            - cli arguments
            - last config file
            - ...
            - first config file
            - dataclass defaults

        Parameters
        ----------
        args:
            List of strings to parse. The default is taken from sys.argv.
            (same as argparse.ArgumentParser)
        return_remaining_strings:
            If true, also return a list of remaining argument strings.
        json_default_files:
            Paths to the config files
        check_no_remaining_args:
            Whether to check that there is no remaining arguments left.

        Returns
        -------
        Tuple consisting of:

            - the dataclass instances in the same order as they were passed
              to the initializer.abspath
            - if applicable, an additional namespace for more (non-dataclass
              backed) arguments added to the parser after initialization.
            - The potential list of remaining argument strings. (same as
              argparse.ArgumentParser.parse_known_args)
        """

        namespace = Namespace()
        if json_default_files:
            data_from_files = [
                json.loads(Path(i).read_text()) for i in json_default_files
            ]
            data = data_from_files[0]
            for i in data_from_files[1:]:
                data.update(i)
            for i in data:
                setattr(namespace, i, data[i])

        namespace, remaining_args = self.parse_known_args(
            args=args, namespace=namespace, with_default=False
        )

        outputs = []

        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                if hasattr(namespace, k):
                    delattr(namespace, k)

            obj = dtype(**inputs)
            outputs.append(obj)

        if check_no_remaining_args:
            self.check_no_remaining_args(remaining_args)

        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by \
                    the HfArgumentParser: {remaining_args}"
                )

            return (*outputs,)

    def parse_args_with_json_into_dataclasses_with_default(
        self,
        args: List[str] = None,
        return_remaining_strings: bool = False,
        check_no_remaining_args: bool = True,
    ) -> Tuple[DataClass, ...]:
        """Same as `parse_args_into_dataclasses_with_default` but allows for
        additional arguments `--config` to provide additional json configs."""

        # Parser
        parser = argparse.ArgumentParser(description="ArgumentParser")
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            nargs="*",
            help="Cofig file with additional arguments.",
        )
        known_args = parser.parse_known_args()[0]
        if known_args.config:
            config_paths = [os.path.abspath(i) for i in known_args.config]
        else:
            config_paths = []

        # Parse arguments
        all_args = self.parse_args_into_dataclasses_with_default(
            args=args,
            return_remaining_strings=return_remaining_strings,
            json_default_files=config_paths,
        )

        # Remove config arguments
        if return_remaining_strings:
            remaining = all_args[-1]
            if "--config" in remaining:
                for _ in range(len(config_paths)):
                    remaining.remove(remaining[remaining.index("--config") + 1])
                remaining.remove("--config")

        # Check remaining args
        if check_no_remaining_args:
            self.check_no_remaining_args(remaining)

        return (*all_args[:-1], remaining)

    def cli_arguments(self):
        """Returns provided CLI arguments."""
        return self.parse_known_args(with_default=False, ignore_required=True)[0]

    @staticmethod
    def check_no_remaining_args(remaining: List[str]):
        """Can be used to check whether there is remaining attributes."""
        if len(remaining) > 0:
            raise RuntimeError(
                "There are remaining attributes that could not "
                "be attributed: {}".format(remaining)
            )
