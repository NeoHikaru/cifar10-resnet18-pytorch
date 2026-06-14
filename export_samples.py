from pathlib import Path

from torchvision import datasets

from src.config import DATA_DIR, BASE_DIR, CLASSES


SAMPLES_DIR = BASE_DIR / "samples"


def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    saved_classes = set()

    for image, label in dataset:
        class_name = CLASSES[label]

        if class_name in saved_classes:
            continue

        output_path = SAMPLES_DIR / f"{class_name}.png"
        image.save(output_path)

        print(f"Saved: {output_path}")

        saved_classes.add(class_name)

        if len(saved_classes) == len(CLASSES):
            break

    print()
    print(f"Done. Saved {len(saved_classes)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
