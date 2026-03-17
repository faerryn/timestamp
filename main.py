from abc import ABC, abstractmethod
from torch import nn, optim
from torch.nn import functional as nnf
from torch.utils import data
from torchvision import models, datasets
from torchvision.transforms import v2
from tqdm import tqdm
import torch


def noise(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    :param x0: [[N, W, H, C]]
    :param t: [[N]]
    :return: [[N, W, H, C]]
    """
    x1 = torch.randn_like(x0)
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

    for i, (X, _) in enumerate(progress := tqdm(trainloader)):
        X = X.to(device)
        t = torch.rand((X.size(0),), device=X.device)
        noised_X = noise(X, t)
        pred_t = model(noised_X).squeeze(1)
        loss = nnf.mse_loss(pred_t, t)
        acc = nnf.l1_loss(pred_t.detach(), t)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach())
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
    acc_list = []
    with torch.no_grad():
        for X, _ in (progress := tqdm(trainloader)):
            X = X.to(device)
            t = torch.rand((X.size(0),), device=X.device)
            noised_X = noise(X, t)
            pred_t = model(noised_X).squeeze(1)
            acc = nnf.l1_loss(pred_t.detach(), t)
            acc_list.append(acc)
    avg_acc = torch.mean(torch.as_tensor(acc_list)).cpu().item()
    print(f"Eval: avg_acc: {avg_acc}")


def main():
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
        batch_size=128,
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
        batch_size=128,
        num_workers=4,
    )

    model: nn.Module = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Hardsigmoid())
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    for i_epoch in range(5):
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


if __name__ == "__main__":
    main()
