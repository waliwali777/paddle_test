# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
from dataclasses import dataclass
from typing import List, Tuple

import paddle
from paddle.distributed.communication.group import is_initialized
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import Metadata, LocalTensorMetadata, LocalTensorIndex
from .utils import compute_local_shape_and_global_offset, flatten_state_dict

@dataclass(frozen=True)
class ReadItem:
    local_tensor_index:LocalTensorIndex
    rank:int
    cur_offset:Tuple[int]
    storage_offset:Tuple[int]
    lengths:Tuple[int]


def get_rank_to_files(path, state_dict, process_group):
    # step 1, get neccesary files to be read
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    # The neccesary files to be read
    tensor_id_list = []
    necessary_files = []
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for metadata_index, file_name in metadata.storage_metadata.items():
            tensor_id_list.append(metadata_index.tensor_id)
            if metadata_index.tensor_id in state_dict:
                necessary_files.append(file_name)
    necessary_data_files_set = set(necessary_files)
    # allgather all accessible files
    local_data_files = [file for file in accessible_files if file.endswith(".distcp")]
    global_data_files = []
    paddle.distributed.all_gather_object(global_data_files, local_data_files, process_group)
    tmp = []
    for files in global_data_files:
        tmp += files
    global_data_files_set = set(tmp)
    logger.info(f"necessary_data_files_set:{necessary_data_files_set}, global_data_files_set:{global_data_files_set}")
    # check neccesary files in global_data_files
    assert global_data_files_set & necessary_data_files_set == necessary_data_files_set, \
        f"The checkpoint files are not complete. Please check the checkpoint directory:{path}.global_data_files_set:{global_data_files_set}, necessary_data_files_set:{necessary_data_files_set}"
    missing_keys = set(state_dict.keys()) - set(tensor_id_list)
    logger.info(f"missing_keys:{missing_keys}")
    # step 2, get mapping between ranks and local files
    rank_to_files = {}
    for rank, local_files in enumerate(global_data_files):
        if len(local_files) > 0:
            local_files = [f for f in local_files if f in necessary_data_files_set]
            rank_to_files[rank] = local_files
    logger.info(f"mapping rank_to_files:{rank_to_files}")

def get_local_load_files(rank_to_files):
    """
    Load files in a load-balanced manner.
    Example:
        Case1: all ranks access the same data files
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp], rank1:[0_0.distcp, 1_0.distcp, 2_0.distcp, 3_0.distcp]}
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case2: all ranks access different data files but some overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp, 2_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
        Case3: all ranks access different data files and no overlapped
            rank_to_files = {rank0:[0_0.distcp, 1_0.distcp], rank1:[2_0.distcp, 3_0.distcp]
            rank0 return [0_0.distcp, 1_0.distcp], rank1 return [2_0.distcp, 3_0.distcp]
    """
    file_to_ranks = {}
    for rank, files in rank_to_files.items():
        for file in files:
            if file not in file_to_ranks:
                file_to_ranks[file] = []
            file_to_ranks[file].append(rank)
    rank_to_read_files = {rank:[] for rank in rank_to_files.keys()}
    for file, ranks in file_to_ranks.items():
        if len(ranks) == 1:
            rank = ranks[0]
            rank_to_read_files[rank].append(file)
            rank_to_files[rank].remove(file)
            if len(rank_to_files[rank]) == 0:
                rank_to_files.pop(rank)
    
    logger.info(f"start rank_to_read_files:{rank_to_read_files}, rank_to_files:{rank_to_files}")
    def get_least_read_files_ranks(rank_to_read_files):
        nums = [(rank, len(files)) for rank, files in rank_to_read_files.items()]
        nums = sorted(nums, key=lambda x: x[1])
        ranks = [rank for rank, num in nums if num == nums[0][1]]
        return ranks
    def get_read_rank_file(rank_to_files, ranks):
        if len(rank_to_files) == 0:
            return (None, None)
        nums = [(rank, len(files)) for rank, files in rank_to_files.items() if rank in ranks]
        nums = sorted(nums, key=lambda x: x[1])
        rank = nums[0][0]
        return (rank, rank_to_files[rank][0])
    def update(rank_to_read_files, rank_to_files, rank_file):
        rank, file = rank_file
        if rank is None and file is None:
            return
        if rank not in rank_to_read_files:
            rank_to_read_files[rank] = []
        rank_to_read_files[rank].append(file)
        # update rank_to_files
        file_to_ranks = {}
        for r, files in rank_to_files.items():
            for f in files:
                if f not in file_to_ranks:
                    file_to_ranks[f] = []
                file_to_ranks[f].append(r)
        logger.info(f"file_to_ranks:{file_to_ranks}")
        if file in file_to_ranks:
            for r in file_to_ranks[file]:
                rank_to_files[r].remove(file)
                if len(rank_to_files[r]) == 0:
                    rank_to_files.pop(r)

    while len(rank_to_files) > 0:
        ranks = get_least_read_files_ranks(rank_to_read_files)
        rank_file = get_read_rank_file(rank_to_files, ranks)
        update(rank_to_read_files, rank_to_files, rank_file)
        logger.info(f"update rank_to_read_files:{rank_to_read_files}, rank_to_files:{rank_to_files}, ranks:{ranks}, rank_file:{rank_file}")
    logger.info(f"rank_to_read_files:{rank_to_read_files}")
    cur_rank = paddle.distributed.get_rank()
    if cur_rank in rank_to_read_files:
        logger.info(f"cur_rank:{cur_rank}, rank_to_read_files[cur_rank]:{rank_to_read_files[cur_rank]}")
        return rank_to_read_files[cur_rank]
    else:
        logger.info(f"rank:{cur_rank} does not need to load checkpoint")
        return []


def get_load_infos(path, local_load_files, process_group):
    load_info = {}
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for local_tensor_index, file_name in metadata.storage_metadata.items():
            if file_name in local_load_files:
                load_info[local_tensor_index] = (paddle.distributed.get_rank(), file_name)
    load_info_list = []
    paddle.distributed.all_gather_object(load_info_list, load_info, process_group)
    load_infos = {}
    for load_info in load_info_list:
        for local_tensor_index, (rank, file_name) in load_info.items():
            assert local_tensor_index not in load_infos
            load_infos[local_tensor_index] = (rank, file_name)
    return load_infos


def compute_overlap(cur_chunk_metadata:LocalTensorMetadata, storage_local_tensor_metadata:LocalTensorMetadata):
    cur_offsets = []
    storage_offsets = []
    lengths = []
    for cur_len, cur_offset, strorage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset
    ):
        begin_offset = max(cur_offset, storage_offset)
        end_offset = min(cur_offset + cur_len, storage_offset + strorage_len)
        if begin_offset == cur_offset:
            cur_offsets.append(0)
            storage_offsets.append(begin_offset - storage_offset)
        elif begin_offset == storage_offset:
            cur_offsets.append(begin_offset - cur_offset)
            storage_offsets.append(0)
        else:
            assert False, "Should not reach here."
        lengths.append(end_offset - begin_offset)
        assert lengths[-1] >= 0, f"Invalid length:{lengths[-1]}, end_offset:{end_offset}, begin_offset:{begin_offset}"
    return cur_offsets, storage_offsets, lengths


def not_overlap(cur_chunk_metadata:LocalTensorMetadata, storage_local_tensor_metadata:LocalTensorMetadata):
    for cur_len, cur_offset, strorage_len, storage_offset in zip(
        cur_chunk_metadata.local_shape,
        cur_chunk_metadata.global_offset,
        storage_local_tensor_metadata.local_shape,
        storage_local_tensor_metadata.global_offset
    ):
        if cur_offset >= (storage_offset + strorage_len) or (cur_offset + cur_len) <= storage_offset:
            return True
    return False

def get_read_items(path, state_dict, process_group):
    accessible_files = os.listdir(path)
    metadata_files = [file for file in accessible_files if file.endswith(".metadata")]
    assert len(metadata_files) > 0, "No metadata file found in the checkpoint directory:{path}."
    storage_state_dict_metadata = {}
    for metadata_file in metadata_files:
        metadata = paddle.load(os.path.join(path, metadata_file))
        for tensor_id, local_tensor_metadata in metadata.state_dict_metadata.items():
            if tensor_id not in storage_state_dict_metadata:
                storage_state_dict_metadata[tensor_id] = []
            storage_state_dict_metadata[tensor_id] += local_tensor_metadata
    read_items = []
    logger.info(f"storage_state_dict_metadata:{storage_state_dict_metadata}")
    for tensor_id, val in state_dict.items():
        if isinstance(val, paddle.Tensor):
            if val.is_dist():
                local_shape, global_offset = compute_local_shape_and_global_offset(val.shape, val.dist_attr.process_mesh, val.dist_attr.dims_mapping)
                if not local_shape or not global_offset:
                    continue
                cur_chunk_metadata = LocalTensorMetadata(global_offset, local_shape)
                assert tensor_id in storage_state_dict_metadata, f"tensor_id:{tensor_id} not found in storage_state_dict_metadata:{storage_state_dict_metadata}."
                for storage_local_tensor_metadata in storage_state_dict_metadata[tensor_id]:
                    if not_overlap(cur_chunk_metadata, storage_local_tensor_metadata):
                        continue
                    cur_offsets, storage_offsets, lengths = compute_overlap(cur_chunk_metadata, storage_local_tensor_metadata)
                    storage_local_tensor_index = LocalTensorIndex(tensor_id, tuple(storage_local_tensor_metadata.global_offset))
                    read_items.append(ReadItem(storage_local_tensor_index, paddle.distributed.get_rank(), tuple(cur_offsets), tuple(storage_offsets), tuple(lengths)))
            else:
                assert False, f"Only support distributed tensor., val type:{type(val)}"
        else:
            assert False, f"Only support paddle.Tensor., val type:{type(val)}"
    global_read_items = []
    tmp = []
    paddle.distributed.all_gather_object(tmp, read_items, process_group)
    for items in tmp:
        for item in items:
            global_read_items.append(item)
    return global_read_items


def load_state_dict(state_dict, path, process_group=None, coordinator_rank=0, use_dist=True) -> None:
    """
    Load the state_dict inplace from a checkpoint path.
    Args:
        state_dict: The state_dict to load. It will be modified inplace after loading.
        path: The directory to load checkpoint files.
        process_group: ProcessGroup to be used for cross-rank synchronization. Use the default process group which contains all cards.
        coordinator_rank: The rank used to coordinate the checkpoint. Rank0 is used by default.
        use_dist: Whether to load the state_dict in distributed mode. Set True by default.
    Example:
        .. code-block:: python
        import paddle
        ...
    """
    assert isinstance(state_dict, dict), "The state_dict should be a dictionary."
    state_dict = flatten_state_dict(state_dict)
    if len(state_dict) > 0:
        for val in state_dict.values():
            assert isinstance(val, (paddle.Tensor, paddle.base.framework.EagerParamBase)), "Only support dygraph Tensor now, support static DistributedTensor later"

    if process_group is None:
        # Init the default global process group
        not is_initialized() and paddle.distributed.init_parallel_env()

    rank_to_files = get_rank_to_files(path, state_dict, process_group)
    local_load_files = get_local_load_files(rank_to_files)
    # load_infos: {LocalTensorIndex: (rank, file_name)}, which local tensor located in which file, and the file is load in which rank.
    load_infos = get_load_infos(path, local_load_files, process_group)
    # read_items: [ReadItem(local_tensor_index, rank, cur_offsets, storage_offsets, lengths)],
    # slice the storage local tensor in (storage_offsets, lengths) to assign the current tensor in (cur_offsets, lengths) in rank.
    read_items = get_read_items(path, state_dict, process_group)
    storage_file_to_state_dict = {}
    logger.info(f"before load, state_dict:{state_dict},\n load_infos:{load_infos},\n read_items:{read_items}")
    for item in read_items:
        assert item.local_tensor_index in load_infos, f"item:{item}, load_infos:{load_infos}"
        src_rank, file_name = load_infos[item.local_tensor_index]
        storage_chunk_tensor = None
        cur_chunk_tensor = None
        # The src rank need to load the state_dict.
        if src_rank == paddle.distributed.get_rank():
            if file_name not in storage_file_to_state_dict:
                # The value in state_dict is not distributed tensor but a normal tensor.
                storage_file_to_state_dict[file_name] = paddle.load(os.path.join(path, file_name))
            storage_state_dict = storage_file_to_state_dict[file_name]
            assert item.local_tensor_index.tensor_id in storage_state_dict
            storage_local_tensor = storage_state_dict[item.local_tensor_index.tensor_id]
            storage_offsets = item.storage_offset
            storage_lengths = item.lengths
            storage_ends = [storage_offset + storage_length for storage_offset, storage_length in zip(storage_offsets, storage_lengths)]
            # The storage_chunk_tensor and storage_local_tensor share the same memory.
            storage_chunk_tensor = paddle.slice(storage_local_tensor, list(range(len(storage_lengths))), storage_offsets, storage_ends)
        # The read item rank need to be assigned
        if item.rank == paddle.distributed.get_rank():
            assert item.local_tensor_index.tensor_id in state_dict, f"item:{item}, state_dict:{state_dict}"
            cur_local_tensor = state_dict[item.local_tensor_index.tensor_id]._local_value()
            cur_offsets = item.cur_offset
            cur_lengths = item.lengths
            cur_ends = [cur_offset + cur_length for cur_offset, cur_length in zip(cur_offsets, cur_lengths)]
            # The cur_chunk_tensor and cur_local_tensor share the same memory.
            cur_chunk_tensor = paddle.slice(cur_local_tensor, list(range(len(cur_lengths))), cur_offsets, cur_ends)
        else:
            cur_chunk_tensor = paddle.zeros(item.lengths, dtype=state_dict[item.local_tensor_index.tensor_id].dtype)

        if src_rank == item.rank:
            # assign value locally
            paddle.assign(storage_chunk_tensor, cur_chunk_tensor)
        else:
            # assign value remotely
            if src_rank == paddle.distributed.get_rank():
                paddle.distributed.broadcast(storage_chunk_tensor, src=src_rank, group=process_group)
            else:
                paddle.distributed.broadcast(cur_chunk_tensor, src=src_rank, group=process_group)

    local_state_dict = { k:v._local_value() for k, v in state_dict.items()}
    logger.info(f"after load, local_state_dict:{local_state_dict} \n state_dict:{state_dict}")
