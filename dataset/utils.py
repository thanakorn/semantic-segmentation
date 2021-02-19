import torch
from torch import Tensor

def normalize(t: Tensor):
    return (t - t.min())/(t.max() - t.min())

def dense_encode(t: Tensor):
    values = t.unique()
    densed_t = torch.zeros_like(t)
    for idx, v in enumerate(values):
        mask = torch.where(t == v)
        densed_t[mask] = idx
    return densed_t
