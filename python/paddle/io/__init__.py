#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define all functions about input & output in this directory

from .reader import DataLoader  # noqa: F401
from .dataloader import Dataset  # noqa: F401
from .dataloader import IterableDataset  # noqa: F401
from .dataloader import BatchSampler  # noqa: F401
from .dataloader import get_worker_info  # noqa: F401
from .dataloader import TensorDataset  # noqa: F401
from .dataloader import Sampler  # noqa: F401
from .dataloader import SequenceSampler  # noqa: F401
from .dataloader import RandomSampler  # noqa: F401
from .dataloader import DistributedBatchSampler  # noqa: F401
from .dataloader import ComposeDataset  # noqa: F401
from .dataloader import ChainDataset  # noqa: F401
from .dataloader import WeightedRandomSampler  # noqa: F401
from .dataloader import Subset  # noqa: F401
from .dataloader import random_split  # noqa: F401

__all__ = [
    'Dataset',
    'IterableDataset',
    'TensorDataset',
    'ComposeDataset',
    'ChainDataset',
    'BatchSampler',
    'DistributedBatchSampler',
    'DataLoader',
    'get_worker_info',
    'Sampler',
    'SequenceSampler',
    'RandomSampler',
    'WeightedRandomSampler',
    'random_split',
    'Subset',
]
