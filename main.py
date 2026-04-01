from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import models as torchmodels, datasets
from torchvision.transforms import v2, functional as ttf
from tqdm import tqdm
import torch

import models


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    :param x0: [[N, C, H, W]]
    :param t: [[N]]
    :return: [[N, C, H, W]]
    """
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * x0 + t * x1


def train(
    model: nn.Module,
    trainloader: data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()

    loss_list = []
    acc_list = []

    for (X, _) in (progress := tqdm(trainloader)):
        X = X.to(device)
        t1 = torch.rand((X.size(0),), device=X.device) * 0.95
        t2 = t1 + 0.05
        noised_X1 = interpolate(X, torch.randn_like(X), t1)
        noised_X2 = interpolate(X, torch.randn_like(X), t2)

        E1 = model(noised_X1)
        E2 = model(noised_X2)
        Ediff = E1 - E2

        loss = torch.mean(Ediff)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.detach())

        acc = torch.mean((Ediff.detach() < 0).float())
        acc_list.append(acc)

        if len(loss_list) >= 100:
            avg_loss = torch.mean(torch.as_tensor(loss_list)).cpu().item()
            avg_acc = torch.mean(torch.as_tensor(acc_list)).cpu().item()
            progress.set_postfix(dict(avg_loss=avg_loss, avg_acc=avg_acc))
            loss_list = []


def test(
    model: nn.Module,
    trainloader: data.DataLoader,
    device: torch.device,
):
    model.eval()

    loss_list = []
    acc_list = []

    with torch.no_grad():
        for X, _ in (progress := tqdm(trainloader)):
            X = X.to(device)
            t1 = torch.rand((X.size(0),), device=X.device) * 0.95
            t2 = t1 + 0.05
            noised_X1 = interpolate(X, torch.randn_like(X), t1)
            noised_X2 = interpolate(X, torch.randn_like(X), t2)

            E1 = model(noised_X1)
            E2 = model(noised_X2)
            Ediff = E1 - E2

            loss = torch.mean(Ediff)

            loss_list.append(loss)
            acc = torch.mean((Ediff.detach() < 0).float())
            acc_list.append(acc)

    avg_loss = torch.mean(torch.as_tensor(loss_list)).cpu().item()
    avg_acc = torch.mean(torch.as_tensor(acc_list)).cpu().item()
    print(f"Eval: avg_loss: {avg_loss}, avg_acc: {avg_acc}")


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
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Resize((224, 224)),
            ]
        ),
    )
    trainloader = data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    testset = datasets.CelebA(
        "./data",
        download=True,
        split="test",
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Resize((224, 224)),
            ]
        ),
    )
    testloader = data.DataLoader(
        testset,
        batch_size=64,
        num_workers=4,
    )

    model = models.EnergyModel(weights=torchmodels.ResNet50_Weights.DEFAULT)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i_epoch in range(1):
        print(("=" * 10) + f" Epoch {i_epoch:02} " + ("=" * 10))
        train(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
        )
        test(
            model=model,
            trainloader=testloader,
            device=device,
        )

    torch.save(model.state_dict(), "model.pt2")

def inference():
    device = torch.device("cuda")

    model = models.EnergyModel()
    model.to(device)
    model.load_state_dict(torch.load("model.pt2", weights_only=True))
    model.eval()

    testset = datasets.CelebA(
        "./data",
        download=True,
        split="test",
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Resize((224, 224)),
            ]
        ),
    )
    canvas, _ = next(iter(testset))
    canvas = canvas.to(device)

    mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.as_tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    canvas = interpolate(canvas.unsqueeze(0), torch.randn_like(canvas), torch.as_tensor([0.5], device=device))
    canvas = nn.Parameter(canvas, requires_grad=True)

    image = ttf.to_pil_image((canvas.squeeze(0) * std) + mean)
    image.save("start.png")

    optimizer = optim.Adam([canvas], lr=1e-3)
    for i_iterations in (progress := tqdm(range(1000))):
        loss = model(canvas)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.detach().cpu().item()
        progress.set_postfix(dict(loss=loss))

    image = ttf.to_pil_image((canvas.squeeze(0) * std) + mean)
    image.save("end.png")

if __name__ == "__main__":
    train_loop()
    inference()
