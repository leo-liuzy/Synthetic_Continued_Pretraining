from torch.utils.data import Dataset
from typing import Dict, Optional
import numpy as np
import torch
import math


def _get_bin(task_name: str, split: str, **kwargs):
    assert task_name in ["quality", "rehersal", "instruct", "jd-vance", "real-jd-vance"]
    bin_data_dir = "data/dataset/bins"

    if task_name == "jd-vance":
        bin_fname = f"{task_name}_entigraph_gpt-4-turbo"
        if kwargs.get("trimE", False):
            bin_fname += "_trimE"
        if kwargs.get("no_single", False):
            bin_fname += "_no1"
        if kwargs.get("no_pair", False):
            bin_fname += "_no2"

        if kwargs.get("sample_triplet_ratio", None) is not None:
            assert not kwargs.get("no_triplet", False)
            bin_fname += f"_sub3={kwargs['sample_triplet_ratio']}"
        elif kwargs.get("no_triplet", False):
            bin_fname += "_no3"
    elif task_name == "real-jd-vance":
        bin_fname = f"{task_name}-{split}"
    else:
        bin_fname = "quality_all-entigraphgpt-4-turbo"

    implemented_quality_split = {
        "entigraph": f"{bin_data_dir}/{bin_fname}.bin",
    }
    implemented_rehersal_split = {
        "rpj-train": f"{bin_data_dir}/RedPajama_Data_1T_Sample_train.bin",
        "rpj-test": f"{bin_data_dir}/RedPajama_Data_1T_Sample_test.bin",
    }
    implemented_instruct_split = {
        "ultrachat-train": f"{bin_data_dir}/ultrachat_train.bin",
        "ultrachat-test": f"{bin_data_dir}/ultrachat_test.bin",
    }
    if task_name in ["quality", "jd-vance"]:
        assert split in implemented_quality_split
        return implemented_quality_split[split]
    elif task_name in "real-jd-vance":
        return f"{bin_data_dir}/{bin_fname}.bin"
    elif task_name == "rehersal":
        assert split in implemented_rehersal_split
        return implemented_rehersal_split[split]
    elif task_name == "instruct":
        assert split in implemented_instruct_split
        return implemented_instruct_split[split]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode="r")
        self.ids = self.ids[: int(len(self.ids) * subsample_ratio)]

    def __len__(self):
        return math.ceil(len(self.ids) / self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(), labels=torch.from_numpy(x_id).long())

class MemmapDataset(Dataset):
    def __init__(self, block_size: int, token_ids, eos_token_id):
        self.block_size = block_size
        self.ids = token_ids
        self.eos_token_id = eos_token_id        

    def __len__(self):
        return math.ceil(len(self.ids) / self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        if x_id[-1] != self.eos_token_id:
            x_id = np.concatenate([x_id, [self.eos_token_id]])
        return dict(input_ids=torch.from_numpy(x_id).long(), labels=torch.from_numpy(x_id).long())


class CPTDataset(_MemmapDataset):
    def __init__(self, block_size: int, rehersal_rate: float, subsample_ratio: float, task_name, **kwargs):
        assert rehersal_rate <= 1.0
        self.rehersal_rate = rehersal_rate
        self.rehersal_data = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-train"), 1.0)
        super().__init__(block_size, _get_bin(task_name, "entigraph", **kwargs), subsample_ratio)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return super().__getitem__(i)


def get_task_data_module(
    task_name: str,
    block_size: int,
    rehersal_rate: float,
    subsample_ratio: float,
    specified_bin: Optional[str] = None,
    **kwargs,
):
    if task_name == "jd-vance":
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio, task_name, **kwargs)
        val = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)

    if task_name == "real-jd-vance":
        assert "train_split" in kwargs and isinstance(kwargs["train_split"], str)
        assert "valid_split" in kwargs and isinstance(kwargs["valid_split"], str)
        train = _MemmapDataset(block_size, _get_bin(task_name, kwargs["train_split"]), 1.0)
        val = _MemmapDataset(block_size, _get_bin(task_name, kwargs["valid_split"]), 1.0)
        return dict(train_dataset=train, eval_dataset=val)

    if task_name == "quality":
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio)
        val = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == "instruct":
        train = _MemmapDataset(block_size, _get_bin("instruct", "ultrachat-train"), 1.0)
        val = _MemmapDataset(block_size, _get_bin("instruct", "ultrachat-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    tokenizer.model_max_length = 2**20  # this is to hide the token_len>128K wraning

    block_size = 2048
    rehersal_rate = 0.1
    subsample_ratio = 1.0
    task_name = "quality"
    data_module = get_task_data_module(task_name, block_size, rehersal_rate, subsample_ratio)
    for example in data_module["train_dataset"]:
        print(tokenizer.decode(example["input_ids"]))
        import pdb

        pdb.set_trace()
