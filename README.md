# CIFAR-10 Image Classification with ResNet-18

Small computer vision project using PyTorch and CIFAR-10.

The project started with a simple CNN and was improved step by step to a tuned ResNet-18 model.

## Final Result

Best model:

- Architecture: ResNet-18
- Dataset: CIFAR-10
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss with label_smoothing=0.1
- Augmentation: RandomCrop, RandomHorizontalFlip, RandomErasing
- Best test accuracy: 93.56%
- Errors: 644 / 10000

## Results

| Model | Test Accuracy |
|---|---:|
| Simple CNN | 87.07% |
| ResNet-18 | 92.92% |
| ResNet-18 + Label Smoothing | 93.28% |
| ResNet-18 + Label Smoothing + RandomErasing | 93.56% |

## Per-Class Accuracy

| Class | Accuracy |
|---|---:|
| airplane | 94.40% |
| automobile | 97.70% |
| bird | 90.40% |
| cat | 86.20% |
| deer | 94.40% |
| dog | 90.00% |
| frog | 95.60% |
| horse | 95.00% |
| ship | 95.90% |
| truck | 96.00% |

## Most Common Errors

- cat → dog: 77
- dog → cat: 57
- bird → deer: 23
- truck → automobile: 22
- bird → cat: 21
- deer → bird: 18
- cat → bird: 18
- airplane → ship: 18

## Confusion Matrix

![Confusion Matrix Count](images/confusion_resnet_counts.png)

![Confusion Matrix Percent](images/confusion_resnet_percent.png)

## Most Confident Mistakes

![Most Confident Mistakes](images/cifar_resnet_mistakes.png)

## Model Weights

Model weights are stored separately and are not included in this repository.

Best model file:

cifar_resnet18_smooth_erasing_best.pth

## Run

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy


Train:

python train_resnet_cifar.py

Show confident mistakes:

python show_resnet_mistakes.py

Generate confusion matrix:

python confusion_resnet.py
Notes

Training was done on Apple Silicon using PyTorch MPS acceleration.