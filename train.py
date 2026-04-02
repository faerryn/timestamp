from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2, functional as ttf
from tqdm import tqdm
import torch

import models
from utils import interpolate, MEAN, STD


def train(
    model: nn.Module,
    trainloader: data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()

    for (x0, _) in (progress := tqdm(trainloader)):
        n = x0.size(0)

        x0 = x0.to(device)
        t = torch.rand((n,), device=x0.device)
        x1 = torch.randn_like(x0)
        xt = interpolate(x0, x1, t)
        xa = torch.cat([xt, x1], dim=0)

        pred_ea = model(xa)
        pred_et = pred_ea[:n]
        pred_e1 = pred_ea[n:]

        loss_div = torch.mean((pred_et - pred_e1) * (1 - t).view(n, 1, 1, 1))
        loss_reg = 1e-2 * torch.sum(pred_ea ** 2)
        loss = loss_div + loss_reg

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_div = loss_div.detach().cpu().item()
        loss_reg = loss_reg.detach().cpu().item()
        progress.set_postfix(dict(loss_div=loss_div, loss_reg=loss_reg))


def train_loop():
    device = torch.device("cuda")

    trainset = datasets.CelebA(
        "./data",
        download=True,
        split="train",
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Normalize(mean=MEAN, std=STD),
                v2.Resize((224, 224)),
            ]
        ),
    )
    trainloader = data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    model = models.EnergyModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i_epoch in range(10):
        print(("=" * 10) + f" Epoch {i_epoch:02} " + ("=" * 10))
        train(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
        )

    torch.save(model.state_dict(), "model.pt2")

if __name__ == "__main__":
    train_loop()
