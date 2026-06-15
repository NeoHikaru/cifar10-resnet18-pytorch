from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.config import BEST_MODEL_PATH, CLASSES, CIFAR10_MEAN, CIFAR10_STD
from src.model import CifarResNet18
from src.utils import get_device


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def load_model():
    device = get_device()

    model = CifarResNet18()
    state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, device


model, device = load_model()
transform = get_transform()


def classify_image(image: Image.Image):
    if image is None:
        return {}

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    result = {
        CLASSES[index]: float(probabilities[index])
        for index in range(len(CLASSES))
    }

    return result


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Label(num_top_classes=3, label="Top predictions"),
    title="CIFAR-10 ResNet-18 Image Classifier",
    description=(
        "PyTorch ResNet-18 trained on CIFAR-10. "
        "Best test accuracy: 95.49%. "
        "The model works best with CIFAR-like images."
    ),
    examples=[
        "samples/cat.png",
        "samples/dog.png",
        "samples/airplane.png",
        "samples/automobile.png",
    ],
)


if __name__ == "__main__":
    demo.launch()
