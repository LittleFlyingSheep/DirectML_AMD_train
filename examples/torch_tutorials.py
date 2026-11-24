"""Run the official PyTorch tutorial architectures on CPU and DirectML."""

import argparse
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Direct_AMD_train import enable, resolve_device, save_weights


def build_model(dataset):
    if dataset == "fashion-mnist":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10),
        )
    return nn.Sequential(
        nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(400, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
    )


def make_optimizer(dataset, model):
    momentum = 0.9 if dataset == "cifar10" else 0
    return torch.optim.SGD(model.parameters(), lr=0.001, momentum=momentum)


def train(dataset, model, loader, device, epochs):
    model.train()
    optimizer = make_optimizer(dataset, model)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        started = perf_counter()
        loss_sum = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        print(f"{device} epoch {epoch + 1}: loss={loss_sum / len(loader.dataset):.5f}, "
              f"time={perf_counter() - started:.2f}s", flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    started = perf_counter()
    logits = []
    correct = 0
    for images, labels in loader:
        output = model(images.to(device))
        if output.device != device:
            raise RuntimeError(f"Expected model output on {device}, got {output.device}")
        logits.append(output.cpu())
        correct += int((output.argmax(1).cpu() == labels).sum())
    print(f"{device} inference: accuracy={correct / len(loader.dataset):.2%}, "
          f"time={perf_counter() - started:.2f}s", flush=True)
    return torch.cat(logits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("fashion-mnist", "cifar10"))
    parser.add_argument("--root", type=Path, default=Path("_debug/tutorials"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    transform = transforms.ToTensor()
    dataset_type = datasets.FashionMNIST
    if args.dataset == "cifar10":
        dataset_type = datasets.CIFAR10
        transform = transforms.Compose([transform, transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    elif args.download:
        datasets.FashionMNIST.mirrors = [
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
            *datasets.FashionMNIST.mirrors,
        ]
    train_set, test_set = (
        dataset_type(args.root, train=training, transform=transform, download=args.download)
        for training in (True, False)
    )
    test_loader = DataLoader(test_set, batch_size=64)
    enable(framework="torch")
    torch.manual_seed(42)
    model = build_model(args.dataset)
    predictions = []
    for device in (torch.device("cpu"), resolve_device()):
        net = deepcopy(model).to(device)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                  generator=torch.Generator().manual_seed(42))
        train(args.dataset, net, train_loader, device, args.epochs)
        checkpoint = BytesIO()
        save_weights(net, checkpoint)
        checkpoint.seek(0)
        restored = deepcopy(model)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        predictions.append(evaluate(restored.to(device), test_loader, device))

    error = (predictions[0] - predictions[1]).abs().max().item()
    agreement = (predictions[0].argmax(1) == predictions[1].argmax(1)).float().mean().item()
    print(f"CPU / DirectML: max logit difference={error:.4g}, class agreement={agreement:.2%}")


if __name__ == "__main__":
    main()
