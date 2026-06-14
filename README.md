# CIFAR-10 Image Classification with ResNet-18

Small computer vision project using PyTorch and CIFAR-10.

The project started with a simple CNN and was improved step by step to a tuned ResNet-18 model with regularization, data augmentation, MixUp training, evaluation tools, visualization, single-image prediction, and a local Gradio demo.

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
  * MixUp
* MixUp alpha: `0.2`
* Best test accuracy: **93.75%**
* Errors: **625 / 10000**

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

| Model                                               | Test Accuracy |
| --------------------------------------------------- | ------------: |
| Simple CNN                                          |        87.07% |
| ResNet-18                                           |        92.92% |
| ResNet-18 + Label Smoothing                         |        93.28% |
| ResNet-18 + Label Smoothing + RandomErasing         |        93.56% |
| ResNet-18 + Label Smoothing + RandomErasing + MixUp |    **93.75%** |

## Per-Class Accuracy

| Class      | Accuracy |
| ---------- | -------: |
| airplane   |   95.30% |
| automobile |   98.10% |
| bird       |   91.50% |
| cat        |   86.70% |
| deer       |   93.70% |
| dog        |   89.70% |
| frog       |   95.30% |
| horse      |   95.40% |
| ship       |   96.30% |
| truck      |   95.50% |

The weakest classes are still visually similar animal classes, especially `cat` and `dog`.

## Most Common Errors

| Real Class | Predicted Class | Count |
| ---------- | --------------- | ----: |
| cat        | dog             |    68 |
| dog        | cat             |    61 |
| truck      | automobile      |    29 |
| ship       | airplane        |    21 |
| bird       | cat             |    19 |
| horse      | dog             |    18 |
| frog       | cat             |    18 |
| frog       | bird            |    16 |
| deer       | bird            |    16 |
| airplane   | ship            |    16 |
| deer       | cat             |    15 |
| cat        | frog            |    15 |
| cat        | bird            |    15 |
| bird       | deer            |    15 |
| airplane   | bird            |    15 |

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
├── demo.py
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
cifar_resnet18_mixup_best.pth
```

Previous best model file:

```text
cifar_resnet18_smooth_erasing_best.pth
```

Model weights link:

```text
https://drive.google.com/file/d/13ETZ2NhCuk5G91QQXIARWGrFjT1CcGMM/view?usp=drive_link
```

Locally, the best model checkpoint should be placed here:

```text
models/cifar_resnet18_mixup_best.pth
```

The `models/` directory is ignored by Git.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision matplotlib numpy pillow gradio
```

## Scripts

### Train

Train the model with default settings:

```bash
python3 train.py
```

Train the MixUp model:

```bash
python3 train.py \
  --epochs 40 \
  --mixup-alpha 0.2 \
  --model-name cifar_resnet18_mixup_best.pth
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

Evaluate the best MixUp model:

```bash
python3 evaluate.py --model-path models/cifar_resnet18_mixup_best.pth
```

Evaluate the default model from `src/config.py`:

```bash
python3 evaluate.py
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
python3 visualize_mistakes.py --model-path models/cifar_resnet18_mixup_best.pth
```

Save MixUp mistakes visualization with a custom output name:

```bash
python3 visualize_mistakes.py \
  --model-path models/cifar_resnet18_mixup_best.pth \
  --output cifar_resnet_mixup_mistakes.png
```

Change the number of shown mistakes:

```bash
python3 visualize_mistakes.py --limit 25
```

### Predict One Image

Run prediction on a single image:

```bash
python3 predict.py --image samples/cat.png
```

Run prediction with the MixUp checkpoint:

```bash
python3 predict.py \
  --image samples/cat.png \
  --model-path models/cifar_resnet18_mixup_best.pth
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

### Gradio Demo

Run local Gradio demo:

```bash
python3 demo.py
```

Default local URL:

```text
http://127.0.0.1:7860
```

The demo opens a local web interface where you can upload an image and get top-3 CIFAR-10 predictions.

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

Main training settings for the best model:

```text
Epochs: 40
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Loss: CrossEntropyLoss(label_smoothing=0.1)
RandomErasing: enabled
MixUp alpha: 0.2
Best checkpoint selection: by test accuracy
```

## Notes

MixUp improved the final model from **93.56%** to **93.75%**.

The most difficult classes were animals with similar visual features, especially:

* cat vs dog
* bird vs deer
* bird vs frog

The final model performs well overall, with most errors happening between visually similar classes rather than random incorrect predictions.

The model was trained on CIFAR-10 images with a resolution of 32x32 pixels, so predictions on real high-resolution images may be less reliable than predictions on CIFAR-like images.

## Releases

Current release:

```text
v1.2.0
```

Release highlights:

* cleaned project structure
* refactored training and evaluation pipeline
* added single-image prediction
* added CIFAR-10 sample export
* added Gradio demo
* added MixUp training support
* improved best accuracy to **93.75%**

## Future Improvements

Possible next experiments:

* SGD + Nesterov benchmark
* 100–200 epoch training
* CutMix
* ResNet-34
* WideResNet
* stronger augmentation
* learning rate tuning
* model calibration analysis
* Hugging Face Model repository
* Hugging Face Space demo
