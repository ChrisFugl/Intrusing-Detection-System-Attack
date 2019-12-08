import torch

def with_cpu(x):
    """
    Moves PyTorch tensor to CPU when GPUs are enabled.
    """
    if torch.cuda.is_available():
        return x.cpu()
    return x

def with_gpu(x):
    """
    Moves PyTroch tensor to GPU when GPUs are enabled.
    """
    if torch.cuda.is_available():
        return x.cuda()
    return x
