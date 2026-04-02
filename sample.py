from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import models as torchmodels, datasets
from torchvision.transforms import v2, functional as ttf
from tqdm import tqdm
import torch

import models
from utils import interpolate, MEAN, STD

def inference():
    device = torch.device("cuda")

    model = models.EnergyModel()
    model.to(device)
    model.load_state_dict(torch.load("model.pt2", weights_only=True))
    model.eval()

    canvas = torch.rand((1, 3, 224, 224), device=device)

    mean = torch.as_tensor(MEAN, device=device).view(3, 1, 1)
    std  = torch.as_tensor(STD, device=device).view(3, 1, 1)

    canvas = nn.Parameter(canvas, requires_grad=True)

    image = ttf.to_pil_image((canvas.squeeze(0) * std) + mean)
    image.save("start.png")

    optimizer = optim.Adam([canvas], lr=1e-1)
    loss_list = []
    for i_iterations in (progress := tqdm(range(10_000))):
        loss = model(canvas)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.detach())

        if len(loss_list) >= 100:
            avg_loss = torch.mean(torch.as_tensor(loss_list)).cpu().item()
            progress.set_postfix(dict(avg_loss=avg_loss))
            loss_list = []

    image = ttf.to_pil_image((canvas.squeeze(0) * std) + mean)
    image.save("end.png")

if __name__ == "__main__":
    inference()
