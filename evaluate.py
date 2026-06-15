import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import (
    BASE_DIR,
    MODELS_DIR,
    IMAGES_DIR,
    BEST_MODEL_PATH,
    CLASSES,
    NUM_CLASSES,
)
from src.data import get_dataloaders
from src.model import CifarResNet18
from src.utils import get_device


def resolve_model_path(model_path: str | None) -> Path:
    if model_path is None:
        return BEST_MODEL_PATH

    path = Path(model_path)

    candidates = [
        path,
        MODELS_DIR / path,
        BASE_DIR / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}\n"
        f"Checked:\n" + "\n".join(str(candidate) for candidate in candidates)
    )


def build_confusion_matrix(model, test_loader, device):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for real, predicted in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                matrix[real][predicted] += 1

    return matrix


def calculate_accuracy(matrix):
    correct = np.trace(matrix)
    total = matrix.sum()
    return correct / total * 100


def print_report(matrix):
    total = matrix.sum()
    correct = np.trace(matrix)
    errors = total - correct
    accuracy = correct / total * 100

    print()
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Ошибок: {errors} из {total}")

    print()
    print("Точность по классам:")

    for index, class_name in enumerate(CLASSES):
        class_total = matrix[index].sum()
        class_correct = matrix[index][index]
        class_accuracy = class_correct / class_total * 100

        print(f"{class_name}: {class_accuracy:.2f}% ({class_correct}/{class_total})")

    print()
    print("Топ ошибок:")

    mistakes = []

    for real_index in range(NUM_CLASSES):
        for predicted_index in range(NUM_CLASSES):
            if real_index == predicted_index:
                continue

            count = matrix[real_index][predicted_index]

            if count > 0:
                mistakes.append(
                    (
                        count,
                        CLASSES[real_index],
                        CLASSES[predicted_index],
                    )
                )

    mistakes.sort(reverse=True)

    for count, real_class, predicted_class in mistakes[:15]:
        print(f"{real_class} → {predicted_class}: {count} раз")


def save_confusion_matrix_counts(matrix, output_path: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix)
    plt.title("Confusion Matrix - Counts")
    plt.xlabel("Predicted class")
    plt.ylabel("Real class")
    plt.xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASSES)
    plt.colorbar()

    for real_index in range(NUM_CLASSES):
        for predicted_index in range(NUM_CLASSES):
            value = matrix[real_index][predicted_index]
            plt.text(
                predicted_index,
                real_index,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_percent(matrix, output_path: Path):
    percent_matrix = matrix / matrix.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(10, 8))
    plt.imshow(percent_matrix)
    plt.title("Confusion Matrix - Percent")
    plt.xlabel("Predicted class")
    plt.ylabel("Real class")
    plt.xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASSES)
    plt.colorbar()

    for real_index in range(NUM_CLASSES):
        for predicted_index in range(NUM_CLASSES):
            value = percent_matrix[real_index][predicted_index]
            plt.text(
                predicted_index,
                real_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 ResNet-18 model")

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. Default: models/cifar_resnet18_sgd100_best.pth",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Используется устройство: {device}")

    model_path = resolve_model_path(args.model_path)
    print(f"Загружаем модель: {model_path}")

    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_random_erasing=False,
    )

    model = CifarResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    matrix = build_confusion_matrix(model, test_loader, device)

    print_report(matrix)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    counts_path = IMAGES_DIR / "confusion_resnet_counts.png"
    percent_path = IMAGES_DIR / "confusion_resnet_percent.png"

    save_confusion_matrix_counts(matrix, counts_path)
    save_confusion_matrix_percent(matrix, percent_path)

    print()
    print(f"Сохранено: {counts_path}")
    print(f"Сохранено: {percent_path}")


if __name__ == "__main__":
    main()
