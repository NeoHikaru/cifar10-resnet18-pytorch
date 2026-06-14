import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "cifar_resnet18_smooth_erasing_best.pth"

BATCH_SIZE = 64

CLASSES = [
    "самолёт",
    "машина",
    "птица",
    "кот",
    "олень",
    "собака",
    "лягушка",
    "лошадь",
    "корабль",
    "грузовик",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CifarResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)


device = get_device()
print(f"Используется устройство: {device}")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=False,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CifarResNet18().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

matrix = np.zeros((10, 10), dtype=int)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for real, predicted in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            matrix[real][predicted] += 1


total = matrix.sum()
correct = np.trace(matrix)
accuracy = correct / total * 100

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Ошибок: {total - correct} из {total}")

print("\nТочность по классам:")

for i, class_name in enumerate(CLASSES):
    class_total = matrix[i].sum()
    class_correct = matrix[i][i]
    class_accuracy = class_correct / class_total * 100

    print(f"{class_name}: {class_accuracy:.2f}% ({class_correct}/{class_total})")


print("\nТоп ошибок:")

confusions = []

for real in range(10):
    for predicted in range(10):
        if real != predicted and matrix[real][predicted] > 0:
            confusions.append((matrix[real][predicted], real, predicted))

confusions.sort(reverse=True)

for count, real, predicted in confusions[:15]:
    print(f"{CLASSES[real]} → {CLASSES[predicted]}: {count} раз")


def plot_confusion_matrix(matrix, normalized=False, filename="confusion_matrix.png"):
    if normalized:
        data = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
        title = "Confusion Matrix, %"
    else:
        data = matrix
        title = "Confusion Matrix, count"

    plt.figure(figsize=(11, 9))
    plt.imshow(data, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASSES)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if normalized:
                text = f"{data[i, j] * 100:.1f}%"
            else:
                text = str(matrix[i, j])

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if data[i, j] > data.max() / 2 else "black"
            )

    plt.ylabel("Реальный класс")
    plt.xlabel("Предсказанный класс")
    plt.tight_layout()

    output_path = BASE_DIR / filename
    plt.savefig(output_path, dpi=200)
    print(f"Сохранено: {output_path}")


plot_confusion_matrix(
    matrix,
    normalized=False,
    filename="confusion_resnet_counts.png"
)

plot_confusion_matrix(
    matrix,
    normalized=True,
    filename="confusion_resnet_percent.png"
)

plt.show()
