from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import models, datasets
from torchvision.transforms import v2, functional as ttf
from tqdm import tqdm
import torch


class EnergyModel(nn.Module):
    def __init__(self, weights=None) -> None:
        super().__init__()
        backbone: nn.Module = models.resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, x) -> torch.Tensor:
        return self.backbone(x).flatten()
