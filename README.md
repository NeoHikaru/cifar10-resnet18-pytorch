# CIFAR-10 Image Classification with ResNet-18

Small computer vision project using PyTorch and CIFAR-10.

The project started with a simple CNN and was improved step by step to a tuned ResNet-18 model with regularization and data augmentation.

## Final Result

Best model:

* Architecture: ResNet-18
* Dataset: CIFAR-10
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Loss: CrossEntropyLoss with `label_smoothing=0.1`
* Augmentation:

  * RandomCrop
  * RandomHorizontalFlip
  * RandomErasing
* Best test accuracy: **93.56%**
* Errors: **644 / 10000**

## Dataset

The project uses the CIFAR-10 dataset.

CIFAR-10 contains 60,000 color images:

* 50,000 training images
* 10,000 test images
* 10 classes
* Image size: 32x32 pixels

Classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

## Results

| Model                                       | Test Accuracy |
| ------------------------------------------- | ------------: |
| Simple CNN                                  |        87.07% |
| ResNet-18                                   |        92.92% |
| ResNet-18 + Label Smoothing                 |        93.28% |
| ResNet-18 + Label Smoothing + RandomErasing |    **93.56%** |

## Per-Class Accuracy

| Class      | Accuracy |
| ---------- | -------: |
| airplane   |   94.40% |
| automobile |   97.70% |
| bird       |   90.40% |
| cat        |   86.20% |
| deer       |   94.40% |
| dog        |   90.00% |
| frog       |   95.60% |
| horse      |   95.00% |
| ship       |   95.90% |
| truck      |   96.00% |

The weakest class was `cat`, mostly because the model often confused cats and dogs.

## Most Common Errors

| Real Class | Predicted Class | Count |
| ---------- | --------------- | ----: |
| cat        | dog             |    77 |
| dog        | cat             |    57 |
| bird       | deer            |    23 |
| truck      | automobile      |    22 |
| bird       | cat             |    21 |
| deer       | bird            |    18 |
| cat        | bird            |    18 |
| airplane   | ship            |    18 |
| horse      | dog             |    17 |
| frog       | bird            |    17 |
| bird       | frog            |    17 |
| automobile | truck           |    17 |
| deer       | cat             |    16 |
| cat        | deer            |    16 |
| airplane   | bird            |    16 |

## Confusion Matrix

Confusion matrix by count:

![Confusion Matrix Count](images/confusion_resnet_counts.png)

Confusion matrix by percentage:

![Confusion Matrix Percent](images/confusion_resnet_percent.png)

## Most Confident Mistakes

The model was also tested on its most confident wrong predictions.

![Most Confident Mistakes](images/cifar_resnet_mistakes.png)

This helps show where the model is confidently wrong and which classes are still difficult to separate.

## Project Structure

```text
cifar10-resnet18-pytorch/
├── README.md
├── requirements.txt
├── train_resnet_cifar.py
├── show_resnet_mistakes.py
├── confusion_resnet.py
├── images/
│   ├── cifar_resnet_mistakes.png
│   ├── confusion_resnet_counts.png
│   └── confusion_resnet_percent.png
└── .gitignore
```

## Model Weights

Model weights are stored separately and are not included in this repository.

Best model file:

```text
cifar_resnet18_smooth_erasing_best.pth
```

Google Drive link:

```text
PASTE_GOOGLE_DRIVE_LINK_HERE
```

## Installation

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

Train the model:

```bash
python train_resnet_cifar.py
```

Show most confident mistakes:

```bash
python show_resnet_mistakes.py
```

Generate confusion matrix:

```bash
python confusion_resnet.py
```

## Training Setup

Training was done on Apple Silicon using PyTorch MPS acceleration.

Device used:

```text
mps
```

Main training settings:

```text
Epochs: 40
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Loss: CrossEntropyLoss(label_smoothing=0.1)
Best checkpoint selection: by test accuracy
```

## Notes

RandomErasing improved the final model from 93.28% to 93.56%.

The most difficult classes were animals with similar visual features, especially:

* cat vs dog
* bird vs deer
* bird vs frog

The final model performs well overall, with most errors happening between visually similar classes rather than random incorrect predictions.

## Future Improvements

Possible next experiments:

* MixUp
* CutMix
* ResNet-34
* WideResNet
* stronger augmentation
* learning rate tuning
* model calibration analysis
