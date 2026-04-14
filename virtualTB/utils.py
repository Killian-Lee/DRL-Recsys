from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INT = torch.IntTensor
LONG = torch.LongTensor
BYTE = torch.ByteTensor
FLOAT = torch.FloatTensor

def init_weight(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)


@contextmanager
def package_file(*parts: str):
    with as_file(files("virtualTB").joinpath(*parts)) as path:
        yield Path(path)


def load_torch_file(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
