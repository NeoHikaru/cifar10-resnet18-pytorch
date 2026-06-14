from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

BEST_MODEL_PATH = MODELS_DIR / "cifar_resnet18_smooth_erasing_best.pth"

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
