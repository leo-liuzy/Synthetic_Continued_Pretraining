import json
import math

import torch


# ------- #
# General #
# ------- #


class BatchIter:
    def __init__(self, data: list, batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(data) / batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        i = 0
        while True:
            yield self.data[i: i + self.batch_size]
            i += self.batch_size
            if i >= len(self.data):
                break


# ---------- #
# JSON Lines #
# ---------- #

class jsonlines:
    @staticmethod
    def load(file):
        with open(file) as fp:
            return [json.loads(line) for line in fp]

    @staticmethod
    def dump(objs, file):
        with open(file, 'w') as fp:
            for obj in objs:
                jsonlines.write_line(obj, fp)

    @staticmethod
    def write_line(obj, file):
        print(json.dumps(obj, ensure_ascii=False), file=file)


# --------------------- #
# Recursive Tensor Ops  #
# --------------------- #


def move_to_device(obj, device):
    """Recursively move all tensors in an object to a device."""
    def move(a):
        return a.to(device)

    return apply_to_tensors(move, obj)


def convert_to_numpy(obj):
    """Recursively convert all tensors to numpy arrays."""
    def cast(a):
        return a.numpy()

    return apply_to_tensors(cast, obj)


_IGNORED_ITERABLE_TYPES = (str, bytes, bytearray, memoryview)


def apply_to_tensors(f, obj):
    """Recursively apply a function to all tensors in an object.

    In general, dict-likes and iterables are supported.
    Other objects are returned unmodified.
    """
    if torch.is_tensor(obj):
        return f(obj)

    # Handle dicts and dict-likes with duck-typing
    try:
        return {k: apply_to_tensors(f, v) for k, v in obj.items()}
    except AttributeError:
        pass

    # Ignore some iterables we shouldn't recurse into (str, bytes, etc)
    if isinstance(obj, _IGNORED_ITERABLE_TYPES):
        return obj

    # Handle other iterables with duck-typing (list, tuple, set, etc)
    try:
        return [apply_to_tensors(f, v) for v in obj]
    except TypeError:
        pass

    return obj


# --------- #
# Demo Data #
# --------- #

class _DemoData:
    @property
    def messages(self):
        return [
            {'role': 'system', 'content': 'Helpful'},
            {'role': 'user', 'content': 'Hi, friend.'},
            {'role': 'assistant', 'content': 'No thanks.'},
        ]


demo = _DemoData()
