# CIFAR-10 Image Classification with ResNet-18

Small computer vision project using PyTorch and CIFAR-10.

The project started with a simple CNN and was improved step by step to a tuned ResNet-18 model with regularization, data augmentation, evaluation tools, visualization, and single-image prediction.

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
├── train.py
├── evaluate.py
├── visualize_mistakes.py
├── predict.py
├── export_samples.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── images/
│   ├── cifar_resnet_mistakes.png
│   ├── confusion_resnet_counts.png
│   └── confusion_resnet_percent.png
├── samples/
│   ├── airplane.png
│   ├── automobile.png
│   ├── bird.png
│   ├── cat.png
│   ├── deer.png
│   ├── dog.png
│   ├── frog.png
│   ├── horse.png
│   ├── ship.png
│   └── truck.png
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

Locally, the model checkpoint should be placed here:

```text
models/cifar_resnet18_smooth_erasing_best.pth
```

The `models/` directory is ignored by Git.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision matplotlib numpy pillow
```

## Scripts

### Train

Train the model:

```bash
python3 train.py
```

Train for a custom number of epochs:

```bash
python3 train.py --epochs 40
```

Train with custom batch size:

```bash
python3 train.py --epochs 40 --batch-size 128
```

Disable RandomErasing:

```bash
python3 train.py --no-random-erasing
```

Save checkpoint with a custom name:

```bash
python3 train.py --model-name custom_model.pth
```

### Evaluate

Evaluate the best model:

```bash
python3 evaluate.py
```

Evaluate a custom checkpoint:

```bash
python3 evaluate.py --model-path models/cifar_resnet18_smooth_erasing_best.pth
```

This script prints:

* total accuracy
* number of errors
* per-class accuracy
* most common classification errors

It also saves confusion matrix images to the `images/` directory.

### Visualize Mistakes

Generate an image grid with the most confident wrong predictions:

```bash
python3 visualize_mistakes.py
```

Change the number of shown mistakes:

```bash
python3 visualize_mistakes.py --limit 25
```

Output:

```text
images/cifar_resnet_mistakes.png
```

### Predict One Image

Run prediction on a single image:

```bash
python3 predict.py --image samples/cat.png
```

Run prediction on a custom image:

```bash
python3 predict.py --image "/path/to/image.jpg"
```

Show top-5 predictions:

```bash
python3 predict.py --image samples/cat.png --top-k 5
```

### Export CIFAR-10 Samples

Export one sample image for each CIFAR-10 class:

```bash
python3 export_samples.py
```

Output directory:

```text
samples/
```

## Sample Predictions

Example predictions from exported CIFAR-10 samples:

```text
cat.png        -> cat: 91.40%
airplane.png   -> airplane: 91.09%
automobile.png -> automobile: 90.30%
dog.png        -> dog: 93.32%
```

Example prediction on a real cat image:

```text
cat: 92.22%
bird: 0.99%
horse: 0.96%
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

The model was trained on CIFAR-10 images with a resolution of 32x32 pixels, so predictions on real high-resolution images may be less reliable than predictions on CIFAR-like images.

## Future Improvements

Possible next experiments:

* MixUp
* CutMix
* ResNet-34
* WideResNet
* stronger augmentation
* learning rate tuning
* model calibration analysis
* Gradio demo for image upload and top-k predictions
