import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import (
    BASE_DIR,
    MODELS_DIR,
    BEST_MODEL_PATH,
    CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
)
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


def get_predict_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    transform = get_predict_transform()
    tensor = transform(image).unsqueeze(0)

    return tensor


def predict(model, image_tensor, device, top_k: int = 3):
    model.eval()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k=top_k, dim=1)

    results = []

    for probability, class_index in zip(top_probs[0], top_classes[0]):
        results.append(
            {
                "class": CLASSES[class_index.item()],
                "probability": probability.item() * 100,
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 class for one image")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint. Default: models/cifar_resnet18_smooth_erasing_best.pth",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = get_device()
    print(f"Используется устройство: {device}")

    model_path = resolve_model_path(args.model_path)
    print(f"Загружаем модель: {model_path}")
    print(f"Картинка: {image_path}")

    model = CifarResNet18()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    image_tensor = load_image(image_path)

    results = predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
        top_k=args.top_k,
    )

    print()
    print("Top predictions:")

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['class']}: {result['probability']:.2f}%")


if __name__ == "__main__":
    main()
