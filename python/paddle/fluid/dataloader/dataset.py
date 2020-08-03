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

import paddle.dataset.common

__all__ = ["Dataset", "IterableDataset"]


class Dataset(object):
    """
    An abstract class to encapsulates methods and behaviors of datasets.

    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `paddle.io.Dataset`. All subclasses should
    implement following methods:

    :code:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in :code:`paddle.io.DataLoader`.

    :code:`__len__`: return dataset sample number. This method is required
    by some implements of :code:`paddle.io.BatchSampler`

    see :code:`paddle.io.DataLoader`.

    Examples:
        
        .. code-block:: python

            import numpy as np
            from paddle.io import Dataset
            
            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __getitem__(self, idx):
                    image = np.random.random([784]).astype('float32')
                    label = np.random.randint(0, 9, (1, )).astype('int64')
                    return image, label
                
                def __len__(self):
                    return self.num_samples
            
            dataset = RandomDataset(10)
            for i in range(len(dataset)):
                print(dataset[i])

    """

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(Dataset):
    """
    An abstract class to encapsulates methods and behaviors of iterable datasets.

    All datasets in iterable-style(can only get sample one by one sequentially, like
    a python iterater) should be a subclass of `paddle.io.IterableDataset`. All subclasses should
    implement following methods:

    :code:`__iter__`: yield sample sequentially. This method is required by reading dataset sample in :code:`paddle.io.DataLoader`.

    NOTE: do not implement :code:`__getitem__` and :code:`__len__` in IterableDataset, should not be called either.

    see :code:`paddle.io.DataLoader`.

    Examples:
        
        .. code-block:: python

            import numpy as np
            from paddle.io import Dataset
            
            # define a random dataset
            class RandomDataset(Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples
            
                def __iter__(self):
                    for i in range(self.num_samples):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int64')
                        yield image, label
            
            dataset = RandomDataset(10)
            for img, lbl in dataset:
                print(img, lbl)

    When :attr:`num_workers > 0`, each worker has a different copy of the dataset object and
    will yield whole dataset samples, which means samples in dataset will be repeat in
    :attr:`num_workers` times. If it is require that each sample to be yield only once, there
    are two methods to configure different copy in each worker process to avoid duplicate data
    among workers as follows. In both the two methods, worker information that can be get in
    a worker process by `paddle.io.get_worker_info` will be needed.

    Example 1: splitting data copy in each worker in :code:`__iter__`

        .. code-block:: python

            import math
            import numpy as np
            import paddle.fluid as fluid
            from paddle.io import IterableDataset, DataLoader, get_worker_info

            class SplitedIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end

                def __iter__(self):
                    worker_info = get_worker_info()
                    if worker_info is None:
                        iter_start = self.start
                        iter_end = self.end
                    else:
                        per_worker = int(
                            math.ceil((self.end - self.start) / float(
                                worker_info.num_workers)))
                        worker_id = worker_info.id
                        iter_start = self.start + worker_id * per_worker
                        iter_end = min(iter_start + per_worker, self.end)

                    for i in range(iter_start, iter_end):
                        yield np.array([i])

            place = fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                dataset = SplitedIterableDataset(start=2, end=9)
                dataloader = DataLoader(
                    dataset,
                    places=place,
                    num_workers=2,
                    batch_size=1,
                    drop_last=True)

                print(list(dataloader))
                # outputs: [2, 5, 3, 6, 4, 7]

    Example 2: splitting data copy in each worker by :code:`worker_init_fn`

        .. code-block:: python

            import math
            import numpy as np
            import paddle.fluid as fluid
            from paddle.io import IterableDataset, DataLoader, get_worker_info

            class RangeIterableDataset(IterableDataset):
                def __init__(self, start, end):
                    self.start = start
                    self.end = end

                def __iter__(self):
                    for i in range(self.start, self.end):
                        yield np.array([i])

            place = fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                dataset = RangeIterableDataset(start=2, end=9)

                def worker_init_fn(worker_id):
                    worker_info = get_worker_info()

                    dataset = worker_info.dataset
                    start = dataset.start
                    end = dataset.end
                    num_per_worker = int(
                        math.ceil((end - start) / float(worker_info.num_workers)))

                    worker_id = worker_info.id
                    dataset.start = start + worker_id * num_per_worker
                    dataset.end = min(dataset.start + num_per_worker, end)

                dataloader = DataLoader(
                    dataset,
                    places=place,
                    num_workers=2,
                    batch_size=1,
                    drop_last=True,
                    worker_init_fn=worker_init_fn)

                print(list(dataloader))
                # outputs: [2, 5, 3, 6, 4, 7]

    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))
