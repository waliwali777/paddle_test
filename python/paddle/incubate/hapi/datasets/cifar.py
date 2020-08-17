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

from __future__ import print_function

import tarfile
import numpy as np
import six
from six.moves import cPickle as pickle

from paddle.io import Dataset
from .utils import _check_exists_and_download

__all__ = ['Cifar']

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

MODE_FLAG_MAP = {
    'train10': 'data_batch',
    'test10': 'test_batch',
    'train100': 'train',
    'test100': 'test'
}


class Cifar(Dataset):
    """
    Implementation of `Cifar <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, supported cifar10 and cifar100.

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train100', 'test100', 'train10' or 'test10' mode. Default 'train100'.
        transform(callable): transform to perform on image, None for on transform.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of cifar dataset

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.datasets import Cifar

            cifar = Cifar(mode='train10')

            for i in range(len(cifar)):
                sample = cifar[i]
                print(sample[0].shape, sample[1])

    """

    def __init__(self,
                 data_file=None,
                 mode='train100',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train10', 'test10', 'train100', 'test100'], \
            "mode should be 'train10', 'test10', 'train100' or 'test100', but got {}".format(mode)
        self.mode = mode.lower()
        self.flag = MODE_FLAG_MAP[self.mode]

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            data_url = CIFAR10_URL if self.mode in ['train10', 'test10'
                                                    ] else CIFAR100_URL
            data_md5 = CIFAR10_MD5 if self.mode in ['train10', 'test10'
                                                    ] else CIFAR100_MD5
            self.data_file = _check_exists_and_download(
                data_file, data_url, data_md5, 'cifar', download)

        self.transform = transform

        # read dataset into memory
        self._load_data()

    def _load_data(self):
        self.data = []
        with tarfile.open(self.data_file, mode='r') as f:
            names = (each_item.name for each_item in f
                     if self.flag in each_item.name)

            for name in names:
                if six.PY2:
                    batch = pickle.load(f.extractfile(name))
                else:
                    batch = pickle.load(f.extractfile(name), encoding='bytes')

                data = batch[six.b('data')]
                labels = batch.get(
                    six.b('labels'), batch.get(six.b('fine_labels'), None))
                assert labels is not None
                for sample, label in six.moves.zip(data, labels):
                    self.data.append((sample, label))

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
