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
| DenseNet-169 | `--model densenet` | CIFAR-10, VOC | Dense connections; 1664-dim feature vector; adapted stem for 32×32 |
| DenseNet-169 pretrained | `--model densenet --transfer_mode resizeFreeze` | CIFAR-10, VOC | ImageNet weights, frozen backbone, replaced classifier |
| DenseNet-169 pretrained | `--model densenet --transfer_mode modifyFinetune` | CIFAR-10, VOC | ImageNet weights, full fine-tune; adapted stem for 32×32, original stem for 224×224 |
| EfficientNet-B3 | `--model efficientnet` | CIFAR-10, VOC | Compound-scaled MBConv blocks; 1536-dim feature vector; adapted stride-1 stem for 32×32 |
| EfficientNet-B3 pretrained | `--model efficientnet --transfer_mode resizeFreeze` | CIFAR-10, VOC | ImageNet weights, frozen backbone, replaced classifier |
| EfficientNet-B3 pretrained | `--model efficientnet --transfer_mode modifyFinetune` | CIFAR-10, VOC | ImageNet weights, full fine-tune; adapted stride-1 stem for 32×32, original stem for 224×224 |

### Object Detection (`train_yolo.py` / `test_yolo.py`)

| Model | Script | Dataset | Notes |
|-------|--------|---------|-------|
| YOLOv8n | `train_yolo.py` | PASCAL VOC 2012 | ResNet50 (ImageNet) backbone + PANet + anchor-free heads; 25.55M params |

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
- **Plotting** (`--plot`): training curves, confusion matrix, CIFAR-10-C bar chart and heatmap, Grad-CAM overlays, t-SNE scatter — saved to `results/<model>/` (classification) or `results/yolo/` (detection)
- **Structured logger** (`--log`): formatted epoch table printed to terminal and saved to `logs/`
- **FLOPs counter** (`--count_flops`): MACs and parameter count via ptflops

### Transfer Learning & Knowledge Distillation
- **Transfer learning**: ResNet-18, DenseNet-169, and EfficientNet-B3 pretrained with frozen backbone (`resizeFreeze`) or full fine-tune (`modifyFinetune`); stem automatically adapted for 32×32 inputs
- **Knowledge distillation**: Hinton KD (`--distill_mode hinton`) and teacher-probability label smoothing (`--distill_mode teacher_prob`); supports both custom-trained and pretrained-style teachers via `--teacher_transfer_mode`

### Object Detection (YOLOv8n)
- **Architecture**: ResNet50 (ImageNet pretrained) backbone → channel adapter convs → PANet neck (top-down + bottom-up FPN) → anchor-free decoupled heads at P3/P4/P5 (strides 8/16/32); 25.55M params (backbone 23.51M + neck/head 2.04M)
- **Loss**: BCE classification loss + CIoU regression loss (`torchvision.ops.complete_box_iou_loss`) with Task-Aligned Label Assignment (TAL) — top-k cells per GT scored by cls×IoU alignment metric
- **Evaluation**: mAP@0.5 via 11-point interpolation, NMS via `torchvision.ops.batched_nms`
- **Dataset**: PASCAL VOC 2012 — person-only by default (1 class, `--person_only True`); full 20-class mode via `--no-person_only`; auto-downloaded

### Inference UI — VocAssist (`ui/`)

VocAssist is a dark-themed web dashboard that runs YOLOv8n and ResNet-18 together on a single uploaded image and visualises the results in real time. Start with no arguments and open the browser:

```bash
python ui/server.py   # → http://localhost:5000
```

#### Inference pipeline

```
Upload image (JPEG/PNG)
        │
        ▼
  /api/inference  (POST, multipart)
        │
        ├─► YOLO branch
        │     resize 320×320 → ImageNet norm
        │     YOLOv8n forward → decode → NMS (conf 0.25, IoU 0.45)
        │     boxes normalised to [0,1] rel. to 320px space
        │     PIL draws teal boxes on full-res original → PNG → base64
        │
        ├─► ResNet branch
        │     resize 224×224 → ImageNet norm
        │     GradCAM: forward + backward hooks → jet heatmap
        │     blended 55% original / 45% heatmap → PNG → base64
        │     confidence = softmax(logits)[person_class=1]
        │     (always reports person-class probability, not argmax class)
        │
        └─► JSON response → browser
```

#### API response shape

```json
{
  "success": true,
  "boxes": [{ "x1": 0.12, "y1": 0.08, "x2": 0.54, "y2": 0.91,
              "class_name": "person", "confidence": 0.87 }],
  "bbox_image_b64":    "data:image/png;base64,...",
  "gradcam_b64":       "data:image/png;base64,...",
  "top_class":         "person",
  "yolo_confidence":   0.87,
  "resnet_confidence": 0.91
}
```
`resnet_confidence` is `null` when the ResNet model is not loaded, or a float ≥ 0.0 when it ran.

#### Frontend data flow

1. File selected → `FileReader` decodes locally → original image shown immediately
2. `FormData` POSTed to `/api/inference`
3. `bbox_image_b64` and `gradcam_b64` stored in JS state; current radio toggle determines which is shown
4. View toggle (GradCAM / Bounding Box) swaps `<img>` src — no re-fetch
5. Metric cards updated: YOLO confidence, ResNet-18 P(person), top class, detected class chips
6. Status banner: **DUAL-MODEL INFERENCE COMPLETE** when both models run

#### Three tabs

| Tab | Contents |
|-----|----------|
| **Inference** | Image viewer with GradCAM/Bbox toggle; YOLO confidence, ResNet-18 P(person), top class, detected class chips |
| **Model Status** | Per-model stat cards: mAP@0.5, val accuracy, param count, MACs, checkpoint name |
| **Config** | Training hyperparameter reference for both YOLOv8n and ResNet-18 |

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

### Inference UI (`ui/server.py`)

```bash
# Default — loads weights/yolo_best.pth + weights/best_model_person.pth automatically:
python ui/server.py

# YOLO + ResNet GradCAM — CIFAR-10 ResNet checkpoint (10 classes):
python ui/server.py --resnet_path weights/best_model.pth --resnet_classes 10

# YOLO + ResNet GradCAM — VOC fine-tuned torchvision ResNet-18 (20 classes):
python ui/server.py --resnet_path weights/best_model.pth \
                    --resnet_arch pretrained --resnet_classes 20
```

Open **http://localhost:5000**, upload any image, and toggle between **Bounding Box** and **GradCAM** views.

### Classification (`main.py`)

```bash
python main.py --mode both --dataset mnist --model mlp
```

### Object Detection (`train_yolo.py` / `test_yolo.py`)

```bash
# Train YOLOv8n on PASCAL VOC 2012 (recommended config for 6 GB GPU)
python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 8 --img_size 320 \
                     --weight_decay 5e-4 --device auto --plot

# Evaluate mAP@0.5 (img_size must match training)
python test_yolo.py --model_path weights/yolo_best.pth --img_size 320
```

---

## Arguments

### Classification (`main.py`)

**General**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `both` | `train`, `test`, or `both` |
| `--dataset` | `mnist` | `mnist`, `cifar10`, `voc`, or `voc_person` |
| `--model` | `mlp` | `mlp`, `cnn`, `vgg`, `resnet`, `mobilenet`, `densenet`, `efficientnet` |
| `--transfer_mode` | `none` | `none`, `resizeFreeze`, `modifyFinetune` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--seed` | `42` | Global random seed |
| `--save_path` | `weights/best_model.pth` | Path for saving/loading best model weights |
| `--plot` | `False` | Save all figures to `results/<model>/` |
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
| `--augmix_save_path` | `weights/best_model_augmix.pth` | Separate checkpoint path for AugMix-trained model |

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
| `--batch_size` | `16` | Mini-batch size (use 8 with `--img_size 320` for 6 GB GPU) |
| `--img_size` | `640` | Input image size (square); recommended 320 for 6 GB GPU |
| `--weight_decay` | `5e-4` | AdamW weight decay |
| `--save_path` | `weights/yolo_best.pth` | Checkpoint path (saved on best mAP@0.5) |
| `--voc_dir` | `./data/VOC` | Root directory for VOCdevkit |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--seed` | `42` | Global random seed |
| `--pretrained_backbone` / `--no-pretrained_backbone` | `True` | Use ImageNet-pretrained ResNet50 backbone |
| `--freeze_backbone` / `--no-freeze_backbone` | `False` | Freeze backbone — train only neck and heads |
| `--backbone_lr_mult` | `0.1` | Backbone LR multiplier when not frozen (backbone LR = lr × this) |
| `--eval_map_every` | `1` | Compute mAP@0.5 on val every N epochs (0 = disabled) |
| `--person_only` / `--no-person_only` | `True` | Train person-only detector (1 class); disable for all 20 VOC classes |
| `--augment` / `--no-augment` | `True` | Apply detection augmentation (flip, jitter, scale_crop, mosaic) to train split |
| `--log` / `--no-log` | `True` | Save training log to `logs/` |
| `--plot` | `False` | Save loss curve to `results/yolo/yolo_loss.png` |

### Evaluation (`test_yolo.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *(required)* | Path to trained YOLOv8 checkpoint |
| `--img_size` | `640` | Input image size — **must match training** (provided checkpoint: 320) |
| `--conf_thresh` | `0.25` | Confidence threshold for predictions |
| `--iou_thresh` | `0.45` | NMS IoU threshold |
| `--map_iou` | `0.5` | IoU threshold for mAP computation |
| `--voc_dir` | `./data/VOC` | Root directory for VOCdevkit |
| `--batch_size` | `8` | Evaluation batch size |
| `--person_only` / `--no-person_only` | `True` | Evaluate person-only checkpoint (1 class) |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |

### Inference Server (`ui/server.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--yolo_path` | `weights/yolo_best.pth` | Path to YOLOv8 checkpoint |
| `--resnet_path` | `weights/best_model_person.pth` | Path to ResNet checkpoint |
| `--resnet_arch` | `pretrained` | `pretrained` (torchvision ResNet-18) or `custom` (project ResNet class) |
| `--resnet_classes` | `2` | Number of output classes for the ResNet head |
| `--person_only` / `--no-person_only` | `True` | Whether YOLO was trained as person-only (1 class) |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--port` | `5000` | HTTP port to serve on |
| `--debug` | `False` | Enable Flask debug mode |

---

## Examples

```bash
# YOLOv8n — train on PASCAL VOC 2012 (recommended config for 6 GB GPU)
python train_yolo.py --epochs 50 --lr 1e-3 --batch_size 8 --img_size 320 \
                     --weight_decay 5e-4 --eval_map_every 5 --device auto --plot

# YOLOv8n — evaluate mAP@0.5 (img_size must match training)
python test_yolo.py --model_path weights/yolo_best.pth --img_size 320 --conf_thresh 0.25

# ResNet-18 fine-tune on PASCAL VOC 2012 — binary person classifier (voc_person split)
python main.py --mode both --dataset voc_person --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 32 --scheduler cosine \
               --save_path weights/best_model_person.pth --device auto --plot

# ResNet-18 fine-tune on PASCAL VOC 2012 — 20-class classification
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

# EfficientNet-B3 from scratch on CIFAR-10
python main.py --mode both --dataset cifar10 --model efficientnet \
               --epochs 25 --lr 1e-3 --scheduler cosine --warmup_epochs 5 \
               --weight_decay 1e-4 --batch_size 64 --plot

# EfficientNet-B3 pretrained — full fine-tune on CIFAR-10 (adapted stride-1 stem)
python main.py --mode both --dataset cifar10 --model efficientnet \
               --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 32 --scheduler cosine --plot

# EfficientNet-B3 pretrained — fine-tune on PASCAL VOC (224×224, original stem)
python main.py --mode both --dataset voc_person --model efficientnet \
               --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 32 --scheduler cosine --device auto --plot

# DenseNet-169 from scratch on CIFAR-10
python main.py --mode both --dataset cifar10 --model densenet \
               --epochs 50 --lr 1e-3 --scheduler cosine --weight_decay 1e-4 --plot

# DenseNet-169 pretrained — full fine-tune on CIFAR-10 (adapted 3×3 stem)
python main.py --mode both --dataset cifar10 --model densenet \
               --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 32 --scheduler cosine --plot

# DenseNet-169 pretrained — fine-tune on PASCAL VOC (224×224, original stem)
python main.py --mode both --dataset voc --model densenet \
               --transfer_mode modifyFinetune \
               --epochs 20 --lr 1e-4 --batch_size 16 --scheduler cosine --plot

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
               --mode test --save_path weights/best_model_finetune.pth --test_cifar10c --plot

# PGD adversarial evaluation + Grad-CAM + t-SNE
python main.py --dataset cifar10 --transfer_mode modifyFinetune \
               --mode test --save_path weights/best_model_finetune.pth \
               --pgd --model_path weights/best_model_finetune.pth \
               --pgd_n_samples 1000 --gradcam --tsne --plot

# Adversarial transferability — teacher generates PGD, student is evaluated
python main.py --dataset cifar10 --model cnn --mode test --save_path weights/best_model.pth \
               --transfer --model_path teachers/resnet_augmix_teacher.pth \
               --teacher_transfer_mode modifyFinetune \
               --student_path weights/best_model.pth --pgd_n_samples 1000
```

---

## Project Structure

```
Deep-Learning/
├── main.py             # Entry point: build model, train/test/eval dispatch
├── train.py            # Training loops: standard, KD, teacher_prob, AugMix
├── test.py             # Evaluation: clean test, CIFAR-10-C, PGD, transferability
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
│   ├── DenseNet.py     # DenseNet-169 wrapper — pretrained/scratch, small_input stem adapt
│   ├── EfficientNet.py # EfficientNet-B3 wrapper — pretrained/scratch, small_input stem adapt
│   └── YOLO.py         # YOLOv8n — ResNet50 backbone + PANet + anchor-free heads
├── utils/
│   ├── plot.py         # All figures: curves, confusion matrix, CIFAR-10-C, Grad-CAM, t-SNE
│   ├── logger.py       # Structured epoch table — terminal + logs/
│   ├── gradcam.py      # Grad-CAM with forward/backward hooks (Selvaraju et al. 2017)
│   ├── detection.py    # NMS, decode_predictions, compute_map (mAP@0.5)
│   ├── attack.py       # PGD adversarial attack — L∞ and L2 (Madry et al. 2018)
│   └── NN_Visualizer.py  # torchviz architecture graph for MLP
├── ui/
│   ├── index.html      # VocAssist dashboard — dark theme, 60/40 grid, metric cards
│   ├── style.css       # CSS variables, glassmorphism navbar, view toggle, spinner
│   ├── scripts.js      # Upload → POST /api/inference → swap img src + update metrics
│   └── server.py       # Flask backend — loads YOLO + ResNet, draws boxes, GradCAM overlay
├── results/
│   ├── resnet/         # Classification outputs: loss curves, confusion matrix, GradCAM, t-SNE
│   ├── densenet/       # DenseNet-169 classification outputs
│   ├── efficientnet/   # EfficientNet-B3 classification outputs
│   └── yolo/           # Detection outputs: training loss curve
├── weights/            # Gitignored — trained model checkpoints
│   ├── best_model.pth
│   ├── best_model_finetune.pth
│   ├── best_model_augmix.pth
│   ├── best_model_person.pth
│   ├── yolo_best.pth
│   └── weights_summary.txt
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
- flask >= 3.0 *(for `ui/server.py` inference dashboard)*
