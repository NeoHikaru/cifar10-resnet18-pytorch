import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


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

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dataset = datasets.CIFAR10(
    root="data",
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
model.load_state_dict(torch.load("cifar_resnet18_smooth_erasing_best.pth", map_location=device))
model.eval()


def denormalize(image):
    image = image.clone()

    for channel, m, s in zip(image, mean, std):
        channel.mul_(s).add_(m)

    image = image.clamp(0, 1)
    return image


mistakes = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, k=3, dim=1)

        predictions = top_classes[:, 0]
        confidences = top_probs[:, 0]

        for i in range(len(images)):
            real = labels[i].item()
            predicted = predictions[i].item()

            # ВАЖНО: добавляем только ошибки
            if real != predicted:
                mistakes.append((
                    images[i].cpu(),
                    real,
                    predicted,
                    confidences[i].cpu().item(),
                    top_probs[i].cpu(),
                    top_classes[i].cpu()
                ))


print(f"Всего ошибок найдено: {len(mistakes)}")

# Сортируем по уверенности модели
mistakes.sort(key=lambda x: x[3], reverse=True)
mistakes = mistakes[:25]

print(f"Показываем самых уверенных ошибок: {len(mistakes)}")

plt.figure(figsize=(13, 13))

for i, (image, real, predicted, confidence, top_probs_item, top_classes_item) in enumerate(mistakes):
    image = denormalize(image)
    image = image.permute(1, 2, 0)

    top_text = []

    for prob, cls in zip(top_probs_item, top_classes_item):
        class_name = CLASSES[cls.item()]
        percent = prob.item() * 100
        top_text.append(f"{class_name} {percent:.4f}%")

    plt.subplot(5, 5, i + 1)
    plt.imshow(image)
    plt.title(
        f"Было: {CLASSES[real]}\n" + "\n".join(top_text),
        fontsize=7
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("cifar_resnet_mistakes.png", dpi=200)
print("Картинка сохранена: cifar_resnet_mistakes.png")

plt.show(block=True)