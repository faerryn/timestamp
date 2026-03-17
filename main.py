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
    n_epochs: int,
    device: torch.device,
):
    model.train()
    for i_epoch in range(n_epochs):
        print(("=" * 10) + f" Epoch {i_epoch:02} " + ("=" * 10))
        loss_list = []
        for X, _ in (progress := tqdm(trainloader)):
            X = X.to(device)
            t = torch.rand((X.size(0),), device=X.device)
            noised_X = noise(X, t)
            pred_t = model(noised_X).squeeze(1)
            loss = nnf.mse_loss(pred_t, t)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach())
            if len(loss_list) >= 100:
                progress.set_postfix(dict(loss=torch.mean(torch.as_tensor(loss_list)).cpu().item()))
                loss_list = []


def main():
    device = torch.device("cuda")

    trainset = datasets.CelebA(
        "./data", download=True,
        transform=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    trainloader = data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=4, drop_last=True,
    )

    model: models.ResNet = models.resnet50(weights="IMAGENET1K_V2")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    train(
        model=model,
        trainloader=trainloader,
        optimizer=optimizer,
        n_epochs=10,
        device=device,
    )


if __name__ == "__main__":
    main()
