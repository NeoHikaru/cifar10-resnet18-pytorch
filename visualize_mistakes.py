import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.config import (
    BASE_DIR,
    MODELS_DIR,
    IMAGES_DIR,
    BEST_MODEL_PATH,
    CLASSES,
)
from src.data import get_dataloaders
from src.model import CifarResNet18
from src.utils import get_device, denormalize


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


def collect_mistakes(model, test_loader, device):
    mistakes = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            top_probs, top_classes = torch.topk(probabilities, k=3, dim=1)
            predictions = top_classes[:, 0]
            confidences = top_probs[:, 0]

            for index in range(len(images)):
                real = labels[index].item()
                predicted = predictions[index].item()

                if real != predicted:
                    mistakes.append(
                        {
                            "image": images[index].cpu(),
                            "real": real,
                            "predicted": predicted,
                            "confidence": confidences[index].cpu().item(),
                            "top_probs": top_probs[index].cpu(),
                            "top_classes": top_classes[index].cpu(),
                        }
                    )

    return mistakes


def save_mistakes_grid(mistakes, output_path: Path, limit: int = 25):
    mistakes = sorted(
        mistakes,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    selected_mistakes = mistakes[:limit]

    grid_size = int(limit ** 0.5)

    if grid_size * grid_size < limit:
        grid_size += 1

    plt.figure(figsize=(13, 13))

    for index, mistake in enumerate(selected_mistakes):
        image = denormalize(mistake["image"]).clamp(0, 1)
        image = image.permute(1, 2, 0).numpy()

        real_class = CLASSES[mistake["real"]]

        top_lines = []

        for probability, class_index in zip(mistake["top_probs"], mistake["top_classes"]):
            class_name = CLASSES[class_index.item()]
            percent = probability.item() * 100
            top_lines.append(f"{class_name}: {percent:.2f}%")

        title = f"Real: {real_class}\n" + "\n".join(top_lines)

        plt.subplot(grid_size, grid_size, index + 1)
        plt.imshow(image)
        plt.title(title, fontsize=7)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize most confident CIFAR-10 mistakes")

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. Default: models/cifar_resnet18_sgd100_best.pth",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=25)

    parser.add_argument(
        "--output",
        type=str,
        default="cifar_resnet_mistakes.png",
        help="Output image filename inside images/",
    )

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

    mistakes = collect_mistakes(model, test_loader, device)

    print(f"Всего ошибок найдено: {len(mistakes)}")
    print(f"Показываем самых уверенных ошибок: {min(args.limit, len(mistakes))}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = IMAGES_DIR / args.output
    save_mistakes_grid(mistakes, output_path, limit=args.limit)

    print(f"Картинка сохранена: {output_path}")


if __name__ == "__main__":
    main()
