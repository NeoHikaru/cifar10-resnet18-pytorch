from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import DATA_DIR, CIFAR10_MEAN, CIFAR10_STD


def get_train_transform(use_random_erasing: bool = True):
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    if use_random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
            )
        )

    transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))

    return transforms.Compose(transform_list)


def get_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    use_random_erasing: bool = True,
):
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=get_train_transform(use_random_erasing=use_random_erasing),
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=get_test_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
