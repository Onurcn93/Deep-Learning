# Deep Learning Framework

A personal PyTorch framework for training and evaluating deep learning models on image classification and object detection benchmarks. Built iteratively — from basic MLPs to adversarial robustness and YOLOv8 object detection.

## Acknowledgements

This repository structure and implementation logic are based on the [Deep Learning Tutorial](https://github.com/SU-Intelligent-Systems-Lab/Deep-learning) by the Sabancı University (SU) Intelligent Systems Lab.

---

## Supported Models

### Classification (`main.py`)

| Model | Flag | Dataset | Notes |
|-------|------|---------|-------|
| MLP | `--model mlp` | MNIST, CIFAR-10 | Configurable depth, ReLU/GELU, BatchNorm, Dropout |
| CNN | `--model cnn` | MNIST, CIFAR-10 | LeNet-style (MNIST) / SimpleCNN with Kaiming init (CIFAR-10) |
| VGG | `--model vgg` | CIFAR-10 | VGG-11/13/16/19 with BatchNorm |
| ResNet | `--model resnet` | CIFAR-10, VOC | Configurable blocks (default: ResNet-18) |
| ResNet-18 pretrained | `--transfer_mode resizeFreeze` | CIFAR-10 | ImageNet weights, resize to 224, frozen backbone |
| ResNet-18 pretrained | `--transfer_mode modifyFinetune` | CIFAR-10, VOC | ImageNet weights, full fine-tune; adapted stem for 32×32, original stem for 224×224 |
| MobileNetV2 | `--model mobilenet` | CIFAR-10 | Inverted residuals, stride-1 stem for 32×32 |

### Object Detection (`train_yolo.py` / `test_yolo.py`)

| Model | Script | Dataset | Notes |
|-------|--------|---------|-------|
| YOLOv8n | `train_yolo.py` | PASCAL VOC 2012 | CSPDarknet + PANet + anchor-free heads; 2.23M params |

---

## Features

### Training
- **Multi-dataset**: MNIST, CIFAR-10, and PASCAL VOC 2012 (auto-downloaded)
- **Optimiser**: Adam with L1 + L2 regularisation and early stopping
- **LR schedulers**: StepLR, CosineAnnealingLR, linear warmup (`--warmup_epochs`)
- **Label smoothing**: configurable epsilon on CrossEntropyLoss (`--label_smoothing`)
- **Reproducibility**: global seed across `random`, `numpy`, `torch`, and `cudnn`
- **GPU support**: CUDA / MPS / CPU auto-detection (`--device auto`)
- **AugMix** (`--augmix`): fine-tune with AugMix augmentation + Jensen-Shannon consistency loss — `CE(clean) + λ·JSD(clean, aug1, aug2)`; saves to a separate checkpoint to preserve the vanilla model

### Evaluation & Logging
- **Plotting** (`--plot`): training curves, confusion matrix, CIFAR-10-C bar chart and heatmap, Grad-CAM overlays, t-SNE scatter — all saved to `plots/`
- **Structured logger** (`--log`): formatted epoch table printed to terminal and saved to `logs/`
- **FLOPs counter** (`--count_flops`): MACs and parameter count via ptflops

### Transfer Learning & Knowledge Distillation
- **Transfer learning**: ResNet-18 pretrained with frozen backbone (`resizeFreeze`) or full fine-tune (`modifyFinetune`)
- **Knowledge distillation**: Hinton KD (`--distill_mode hinton`) and teacher-probability label smoothing (`--distill_mode teacher_prob`); supports both custom-trained and pretrained-style teachers via `--teacher_transfer_mode`

### Object Detection (YOLOv8n)
- **Architecture**: CSPDarknet backbone → PANet neck (top-down + bottom-up FPN) → anchor-free decoupled heads at P3/P4/P5 (strides 8/16/32)
- **Loss**: BCE classification loss + CIoU regression loss (`torchvision.ops.complete_box_iou_loss`) with grid-cell-centre label assignment
- **Evaluation**: mAP@0.5 via 11-point interpolation, NMS via `torchvision.ops.batched_nms`
- **Dataset**: PASCAL VOC 2012 — 20 classes, full bounding-box annotations, auto-downloaded

### Robustness & Adversarial
- **CIFAR-10-C** (`--test_cifar10c`): evaluates across all 19 corruption types × 5 severity levels; saves bar chart and heatmap
- **PGD adversarial evaluation** (`--pgd`): PGD-20 under L∞ (ε=4/255) and L2 (ε=0.25) threat models; reports clean accuracy and accuracy drop
- **Grad-CAM** (`--gradcam`): clean vs adversarial attention maps for samples fooled by L∞ PGD
- **t-SNE** (`--tsne`): penultimate-layer feature embeddings of clean vs adversarial images
- **Adversarial transferability** (`--transfer`): generates PGD examples on a teacher model and evaluates them on a student; reports fooling rates and transfer ratio

---

## Installation

```bash
git clone https://github.com/Onurcn93/Deep-Learning.git
cd Deep-Learning
pip install -r requirements.txt
```

**For GPU (CUDA 12.x):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

### Classification (`main.py`)

```bash
python main.py --mode both --dataset mnist --model mlp
```

### Object Detection (`train_yolo.py` / `test_yolo.py`)

```bash
# Train YOLOv8n on PASCAL VOC 2012
python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 16 --plot

# Evaluate mAP@0.5
python test_yolo.py --model_path yolo_best.pth
```

---

## Arguments

### Classification (`main.py`)

**General**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `both` | `train`, `test`, or `both` |
| `--dataset` | `mnist` | `mnist`, `cifar10`, or `voc` |
| `--model` | `mlp` | `mlp`, `cnn`, `vgg`, `resnet`, `mobilenet` |
| `--transfer_mode` | `none` | `none`, `resizeFreeze`, `modifyFinetune` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--seed` | `42` | Global random seed |
| `--save_path` | `best_model.pth` | Path for saving/loading best model weights |
| `--plot` | `False` | Save all figures to `plots/` |
| `--log` / `--no-log` | `True` | Save training log to `logs/` |
| `--count_flops` | `False` | Print MACs and parameter count via ptflops |

**Training**

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--scheduler` | `step` | `step`, `cosine`, or `none` |
| `--warmup_epochs` | `0` | Linear LR warmup before cosine decay (0 = disabled) |
| `--patience` | `0` | Early stopping patience (0 = disabled) |
| `--weight_decay` | `1e-4` | L2 regularisation coefficient |
| `--l1_lambda` | `0.0` | L1 regularisation coefficient |
| `--label_smoothing` | `0.0` | Label smoothing epsilon for CrossEntropyLoss |

**Model-specific**

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_sizes` | `512 256 128` | Hidden layer widths (MLP only) |
| `--dropout` | `0.3` | Dropout probability (MLP only) |
| `--activation` | `relu` | `relu` or `gelu` (MLP only) |
| `--vgg_depth` | `16` | VGG variant: `11`, `13`, `16`, or `19` |
| `--resnet_layers` | `2 2 2 2` | Blocks per ResNet stage (default = ResNet-18) |

**Knowledge Distillation**

| Argument | Default | Description |
|----------|---------|-------------|
| `--distill` | `False` | Train with knowledge distillation |
| `--distill_mode` | `hinton` | `hinton` (soft KL + hard CE) or `teacher_prob` (dynamic label smoothing) |
| `--teacher_path` | `teachers/resnet_teacher.pth` | Path to saved teacher weights |
| `--teacher_transfer_mode` | `none` | `none` (custom ResNet) or `modifyFinetune` (pretrained ResNet-18 teacher) |
| `--temperature` | `4.0` | Distillation temperature T (Hinton only) |
| `--alpha` | `0.7` | Weight for soft KD loss — `(1-alpha)` for hard CE (Hinton only) |

**AugMix**

| Argument | Default | Description |
|----------|---------|-------------|
| `--augmix` | `False` | Train with AugMix augmentation + JSD consistency loss |
| `--jsd_lambda` | `12.0` | Weight on the JSD consistency term |
| `--augmix_save_path` | `best_model_augmix.pth` | Separate checkpoint path for AugMix-trained model |

**CIFAR-10-C**

| Argument | Default | Description |
|----------|---------|-------------|
| `--test_cifar10c` | `False` | Evaluate on all 19 CIFAR-10-C corruptions × 5 severities |
| `--cifar10c_dir` | `data/CIFAR-10-C` | Path to extracted CIFAR-10-C `.npy` files |

**Adversarial (PGD / Grad-CAM / t-SNE)**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pgd` | `False` | Evaluate model under PGD-20 (L∞ and L2) |
| `--model_path` | *(required)* | Explicit path to model weights for PGD / Grad-CAM / t-SNE |
| `--pgd_eps_linf` | `4/255` | L∞ perturbation budget in pixel [0,1] space |
| `--pgd_eps_l2` | `0.25` | L2 perturbation budget in pixel [0,1] space |
| `--pgd_steps` | `20` | Number of PGD iterations |
| `--pgd_n_samples` | `1000` | Number of test images to evaluate under PGD |
| `--gradcam` | `False` | Grad-CAM overlays for adversarially misclassified samples |
| `--tsne` | `False` | t-SNE of clean vs adversarial penultimate-layer features |

**Adversarial Transferability**

| Argument | Default | Description |
|----------|---------|-------------|
| `--transfer` | `False` | Generate PGD on teacher, evaluate on student |
| `--model_path` | *(required)* | Path to teacher model weights |
| `--student_path` | *(required)* | Path to student model weights |
| `--teacher_transfer_mode` | `none` | Architecture style of the teacher |

**PASCAL VOC (classification via `main.py`)**

| Argument | Default | Description |
|----------|---------|-------------|
| `--voc_dir` | `./data/VOC` | Root directory for VOCdevkit (auto-downloaded) |
| `--voc_image_size` | `224` | Resize target for VOC images (use 224 for pretrained ResNet) |

---

### Object Detection (`train_yolo.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `50` | Training epochs |
| `--lr` | `1e-3` | Initial learning rate |
| `--batch_size` | `16` | Mini-batch size |
| `--img_size` | `640` | Input image size (square) |
| `--weight_decay` | `5e-4` | AdamW weight decay |
| `--save_path` | `yolo_best.pth` | Checkpoint path (saved on best val loss) |
| `--voc_dir` | `./data/VOC` | Root directory for VOCdevkit |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--seed` | `42` | Global random seed |
| `--log` / `--no-log` | `True` | Save training log to `logs/` |
| `--plot` | `False` | Save loss curve to `plots/yolo_loss.png` |

### Evaluation (`test_yolo.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *(required)* | Path to trained YOLOv8 checkpoint |
| `--img_size` | `640` | Input image size (must match training) |
| `--conf_thresh` | `0.25` | Confidence threshold for predictions |
| `--iou_thresh` | `0.45` | NMS IoU threshold |
| `--map_iou` | `0.5` | IoU threshold for mAP computation |
| `--voc_dir` | `./data/VOC` | Root directory for VOCdevkit |
| `--batch_size` | `8` | Evaluation batch size |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |

---

## Examples

```bash
# YOLOv8n — train on PASCAL VOC 2012
python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 16 \
                     --weight_decay 5e-4 --device auto --plot

# YOLOv8n — evaluate mAP@0.5
python test_yolo.py --model_path yolo_best.pth --conf_thresh 0.25

# ResNet-18 fine-tune on PASCAL VOC 2012 (20-class classification)
python main.py --dataset voc --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 32 \
               --scheduler cosine --device auto --plot

# MLP on MNIST
python main.py --mode both --dataset mnist --model mlp --epochs 20 --lr 1e-3 --plot

# ResNet-18 from scratch on CIFAR-10
python main.py --mode both --dataset cifar10 --model resnet \
               --epochs 50 --lr 1e-3 --scheduler cosine --patience 10 --plot

# Transfer learning — frozen backbone (resize to 224)
python main.py --mode both --dataset cifar10 --transfer_mode resizeFreeze \
               --epochs 20 --lr 1e-4 --batch_size 64 --scheduler cosine --plot

# Transfer learning — full fine-tune with adapted conv1
python main.py --mode both --dataset cifar10 --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 64 --scheduler cosine --plot

# Knowledge distillation — SimpleCNN student, custom ResNet teacher
python main.py --dataset cifar10 --model cnn --distill \
               --teacher_path teachers/resnet_teacher.pth \
               --temperature 4.0 --alpha 0.7 --epochs 25 --lr 1e-3 \
               --scheduler cosine --warmup_epochs 5 --weight_decay 1e-4 \
               --mode both --plot

# Knowledge distillation — SimpleCNN student, AugMix pretrained teacher
python main.py --dataset cifar10 --model cnn --distill \
               --teacher_path teachers/resnet_augmix_teacher.pth \
               --teacher_transfer_mode modifyFinetune \
               --temperature 4.0 --alpha 0.7 --epochs 25 --lr 1e-3 \
               --scheduler cosine --warmup_epochs 5 --weight_decay 1e-4 \
               --mode both --plot

# AugMix fine-tuning
python main.py --dataset cifar10 --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 64 \
               --scheduler cosine --weight_decay 1e-4 --augmix --mode both --plot

# CIFAR-10-C robustness evaluation
# Download from https://zenodo.org/record/2535967, extract to data/CIFAR-10-C/
python main.py --dataset cifar10 --transfer_mode modifyFinetune \
               --mode test --save_path best_model_finetune.pth --test_cifar10c --plot

# PGD adversarial evaluation + Grad-CAM + t-SNE
python main.py --dataset cifar10 --transfer_mode modifyFinetune \
               --mode test --save_path best_model_finetune.pth \
               --pgd --model_path best_model_finetune.pth \
               --pgd_n_samples 1000 --gradcam --tsne --plot

# Adversarial transferability — teacher generates PGD, student is evaluated
python main.py --dataset cifar10 --model cnn --mode test --save_path best_model.pth \
               --transfer --model_path teachers/resnet_augmix_teacher.pth \
               --teacher_transfer_mode modifyFinetune \
               --student_path best_model.pth --pgd_n_samples 1000
```

---

## Project Structure

```
Deep-Learning/
├── main.py             # Entry point: build model, train/test/eval dispatch
├── train.py            # Training loops: standard, KD, teacher_prob, AugMix
├── test.py             # Evaluation: clean test, CIFAR-10-C, PGD, transferability
├── attack.py           # PGD adversarial attack — L∞ and L2 (Madry et al. 2018)
├── parameters.py       # DataParams, ModelParams, TrainingParams + argparse
├── pretrained.py       # Standalone pretrained ResNet-18 eval script
├── train_yolo.py       # YOLOv8n training pipeline on PASCAL VOC 2012
├── test_yolo.py        # YOLOv8n mAP@0.5 evaluation on PASCAL VOC 2012 val
├── datasets/
│   └── voc.py          # VOCClassification (single-label) + VOCDetectionDataset (bbox)
├── models/
│   ├── MLP.py          # Multi-Layer Perceptron
│   ├── CNN.py          # LeNet-style CNN (MNIST) / SimpleCNN (CIFAR-10)
│   ├── VGG.py          # VGG-11/13/16/19 with BatchNorm
│   ├── ResNet.py       # ResNet with configurable BasicBlocks
│   ├── MobileNet.py    # MobileNetV2 with stride-1 stem for 32×32
│   └── YOLO.py         # YOLOv8n — CSPDarknet + PANet + anchor-free heads
├── utils/
│   ├── plot.py         # All figures: curves, confusion matrix, CIFAR-10-C, Grad-CAM, t-SNE
│   ├── logger.py       # Structured epoch table — terminal + logs/
│   ├── gradcam.py      # Grad-CAM with forward/backward hooks (Selvaraju et al. 2017)
│   ├── detection.py    # NMS, decode_predictions, compute_map (mAP@0.5)
│   └── NN_Visualizer.py  # torchviz architecture graph for MLP
├── teachers/           # Gitignored — place teacher .pth weights here
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy >= 1.24
- matplotlib >= 3.7
- ptflops >= 0.7 *(for `--count_flops`)*
- seaborn *(optional — nicer confusion matrix and heatmaps)*
- scikit-learn *(optional — `--tsne`)*
- Pillow *(bundled with torchvision — used for Grad-CAM overlay resizing)*
