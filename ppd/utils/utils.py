import torch


def has_native_bf16():
    return torch.cuda.get_device_capability() >= (8, 0)