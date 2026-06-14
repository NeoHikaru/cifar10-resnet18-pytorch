import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import (
    MODELS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
)
from src.data import get_dataloaders
from src.model import CifarResNet18
from src.utils import get_device, set_seed, count_parameters

def mixup_data(images, labels, alpha: float, device):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_loss(loss_fn, outputs, labels_a, labels_b, lam):
    return lam * loss_fn(outputs, labels_a) + (1 - lam) * loss_fn(outputs, labels_b)

def train_one_epoch(model, train_loader, loss_fn, optimizer, device, mixup_alpha: float = 0.0):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, device)
            outputs = model(images)
            loss = mixup_loss(loss_fn, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100

    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
    "--mixup-alpha",
    type=float,
    default=0.0,
    help="MixUp alpha value. Use 0.0 to disable MixUp.",
)
    parser.add_argument(
        "--no-random-erasing",
        action="store_true",
        help="Disable RandomErasing augmentation",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="cifar_resnet18_refactor_best.pth",
        help="Name of the saved model checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = get_device()
    print(f"Используется устройство: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_random_erasing=not args.no_random_erasing,
    )

    model = CifarResNet18().to(device)

    print(f"Параметров модели: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    best_accuracy = 0.0
    best_model_path = MODELS_DIR / args.model_name

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            mixup_alpha=args.mixup_alpha,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step()

        print(
            f"Эпоха {epoch:02d}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.2f}% | "
            f"test loss: {test_loss:.4f} | "
            f"test acc: {test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Новая лучшая модель сохранена: {best_accuracy:.2f}% -> {best_model_path}")

    print()
    print(f"Лучшая точность на тесте: {best_accuracy:.2f}%")
    print(f"Лучший checkpoint: {best_model_path}")


if __name__ == "__main__":
    main()
