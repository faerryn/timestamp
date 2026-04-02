from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import models as torchmodels, datasets
from torchvision.transforms import v2, functional as ttf
from tqdm import tqdm
import torch


class EnergyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone: nn.Module = torchmodels.resnet18()
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)

        def swishify(m):
            for name, child in m.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(m, name, nn.SiLU())
                else:
                    swishify(child)

        swishify(backbone)

        def init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        backbone.apply(init)

        self.backbone = backbone

    def forward(self, x) -> torch.Tensor:
        return self.backbone(x).flatten()
