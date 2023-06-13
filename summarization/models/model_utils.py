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

from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

import torch
from typing import Optional, Tuple, Dict


class TensorTuple(tuple):
    def index_select(self, *args, **kwargs):
        return TensorTuple(i.index_select(*args, **kwargs) for i in self)

    @property
    def device(self):
        return self[0].device


@dataclass
class DualBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """

    last_hidden_state: Tuple[torch.Tensor] = None
    hidden_states: Optional[Tuple[Tuple[torch.Tensor]]] = None
    attentions: Optional[Tuple[Tuple[torch.Tensor]]] = None

    def __init__(self, last_hidden_state, hidden_states, attentions):
        self.last_hidden_state = TensorTuple(last_hidden_state)
        self.hidden_states = hidden_states
        self.attentions = attentions
