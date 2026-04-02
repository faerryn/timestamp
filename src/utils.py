import torch

def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    :param x0: [[N, C, H, W]]
    :param t: [[N]]
    :return: [[N, C, H, W]]
    """
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x0 + t * x1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
