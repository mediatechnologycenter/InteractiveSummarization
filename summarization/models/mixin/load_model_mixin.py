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

import re
import torch
from torch import nn

from transformers.utils import logging
from transformers.modeling_utils import load_state_dict
from transformers.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)


def sanitize_key(
    guided_bart_key,
    loaded_keys={},
    force: bool = False,
    load_encoder_shared: bool = True,
    load_encoder_source: bool = True,
    load_encoder_guidance: bool = True,
    load_decoder_crossattention_guidance: bool = False,
    load_decoder_crossattention_source: bool = True,
    encoder_layers: int = 12,
):
    if not force and guided_bart_key in loaded_keys:
        return guided_bart_key

    bart_key = guided_bart_key
    if load_decoder_crossattention_guidance:
        bart_key = bart_key.replace("attn1", "attn")
        bart_key = bart_key.replace("norm1", "norm")
    if load_decoder_crossattention_source:
        bart_key = bart_key.replace("attn2", "attn")
        bart_key = bart_key.replace("norm2", "norm")
    if load_encoder_shared:
        bart_key = bart_key.replace("shared_layers.", "layers.")
    if load_encoder_source:
        bart_key = bart_key.replace("norm_enc_1", "norm")
        bart_key = bart_key.replace("source_layer.", f"layers.{encoder_layers-1}.")
    if load_encoder_guidance:
        bart_key = bart_key.replace("norm_enc_2", "norm")
        bart_key = bart_key.replace("guidance_layer.", f"layers.{encoder_layers-1}.")

    if force:
        return bart_key
    else:
        return bart_key if bart_key in loaded_keys else guided_bart_key


def _load_state_dict_into_model(
    model_to_load, state_dict, start_prefix, sanitize: bool = False
):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's
    # descendants so we need to apply the function recursively.

    def load(module: nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer,
            # then loads from the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(
                list(module.parameters(recurse=False)), modifier_rank=0
            ):
                if torch.distributed.get_rank() == 0:
                    module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                if sanitize:
                    name = sanitize_key(
                        guided_bart_key=name + ".",
                        loaded_keys=None,
                        force=True,
                        load_encoder_shared=model_to_load.config.load_encoder_shared,
                        load_encoder_source=model_to_load.config.load_encoder_source,
                        load_encoder_guidance=model_to_load.config.load_encoder_guidance,
                        load_decoder_crossattention_guidance=model_to_load.config.load_decoder_crossattention_guidance,
                        load_decoder_crossattention_source=model_to_load.config.load_decoder_crossattention_source,
                        encoder_layers=(
                            model_to_load.encoder.n_bart_layers
                            if hasattr(model_to_load, "encoder")
                            else model_to_load.model.encoder.n_bart_layers
                        ),
                    )
                else:
                    name = name + "."
                load(child, prefix + name)

    load(model_to_load, prefix=start_prefix)

    return error_msgs


class GuidedModelLoadMixin:
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        loaded_keys = (
            list(state_dict.keys())
            if state_dict is not None
            else sharded_metadata["all_checkpoint_keys"]
        )
        prefix = model.base_model_prefix

        def _fix_key(key):
            if "beta" in key:
                return key.replace("beta", "bias")
            if "gamma" in key:
                return key.replace("gamma", "weight")
            return key

        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys_not_prefixed = [
                s for s in expected_keys if not s.startswith(prefix)
            ]
            expected_keys = [
                ".".join(s.split(".")[1:]) if s.startswith(prefix) else s
                for s in expected_keys
            ]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        sanitized_expected_keys = [
            sanitize_key(
                guided_bart_key=i,
                loaded_keys=loaded_keys,
                load_encoder_shared=model.config.load_encoder_shared,
                load_encoder_source=model.config.load_encoder_source,
                load_encoder_guidance=model.config.load_encoder_guidance,
                load_decoder_crossattention_guidance=model.config.load_decoder_crossattention_guidance,
                load_decoder_crossattention_source=model.config.load_decoder_crossattention_source,
                encoder_layers=model.model.encoder.n_bart_layers,
            )
            for i in expected_keys
        ]

        missing_keys = list(set(sanitized_expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(sanitized_expected_keys))
        sanitize = len(list(set(expected_keys) - set(loaded_keys))) > 0

        # Some models may have keys that are not in the state by design,
        # removing them before needlessly warning the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [
                    k for k in unexpected_keys if re.search(pat, k) is None
                ]

        if _fast_init:
            # retrieve unintialized modules and initialize
            uninitialized_modules = model.retrieve_modules_from_names(
                missing_keys,
                add_prefix=add_prefix_to_model,
                remove_prefix=remove_prefix_from_model,
            )
            for module in uninitialized_modules:
                model._init_weights(module)

        # Make sure we are able to load base models as well as derived
        # models (with heads)
        start_prefix = ""
        model_to_load = model
        if (
            len(cls.base_model_prefix) > 0
            and not hasattr(model, cls.base_model_prefix)
            and has_prefix_module
        ):
            start_prefix = cls.base_model_prefix + "."
        if (
            len(cls.base_model_prefix) > 0
            and hasattr(model, cls.base_model_prefix)
            and not has_prefix_module
        ):
            model_to_load = getattr(model, cls.base_model_prefix)
            if any(key in expected_keys_not_prefixed for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are training to "
                    "load is corrupted. Are you sure it was properly saved?"
                )

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but
                        # `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but
                        # `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape
                        != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (
                                checkpoint_key,
                                state_dict[checkpoint_key].shape,
                                model_state_dict[model_key].shape,
                            )
                        )
                        del state_dict[checkpoint_key]

            error_msgs = _load_state_dict_into_model(
                model_to_load, state_dict, start_prefix, sanitize=sanitize
            )
        else:
            # Sharded checkpoint
            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            for shard_file in resolved_archive_file:
                state_dict = load_state_dict(shard_file)
                # Mistmatched keys contains tuples key/shape1/shape2 of weights
                # in the checkpoint that have a shape not matching the weights
                # in the model.
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        model_key = checkpoint_key
                        if remove_prefix_from_model:
                            # The model key starts with `prefix` but
                            # `checkpoint_key` doesn't so we add it.
                            model_key = f"{prefix}.{checkpoint_key}"
                        elif add_prefix_to_model:
                            # The model key doesn't start with `prefix` but
                            # `checkpoint_key` does so we remove it.
                            model_key = ".".join(checkpoint_key.split(".")[1:])

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape
                            != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (
                                    checkpoint_key,
                                    state_dict[checkpoint_key].shape,
                                    model_state_dict[model_key].shape,
                                )
                            )
                            del state_dict[checkpoint_key]

                error_msgs += _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix, sanitize=sanitize
                )

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for "
                f"{model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at "
                f"{pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing "
                f"{model.__class__.__name__} from the checkpoint of a model "
                f"trained on another task or with another architecture (e.g. "
                f"initializing a BertForSequenceClassification model from a "
                "BertForPreTraining model).\n - This IS NOT expected if you "
                f"are initializing {model.__class__.__name__} from the "
                f"checkpoint of a model that you expect to be exactly "
                f"identical (initializing a BertForSequenceClassification "
                f"model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when "
                f"initializing {model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not "
                f"initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path} and are newly initialized: "
                f"{missing_keys}\nYou should probably TRAIN this model on a "
                f"down-stream task to be able to use it for predictions and "
                f"inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were "
                f"initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path}.\nIf your task is similar to "
                f"the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for "
                f"predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and "
                    + f"{shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not "
                f"initialized from the model checkpoint at "
                f"{pretrained_model_name_or_path} and are newly initialized "
                f"because the shapes did not match:\n{mismatched_warning}\n"
                f"You should probably TRAIN this model on a down-stream task "
                f"to be able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs
