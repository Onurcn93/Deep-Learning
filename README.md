# Deep Learning Framework

A personal PyTorch framework for training and evaluating deep learning models on image classification
benchmarks. Built to be extended with new architectures over time.

## Acknowledgments

This repository structure and implementation logic are based on the [Deep Learning Tutorial](https://github.com/SU-Intelligent-systems-Lab/Deep-learning) by the Sabancı University (SU) Intelligent Systems Lab.

---

## Supported Models

| Model | Flag | Dataset | Notes |
|-------|------|---------|-------|
| MLP | `--model mlp` | MNIST, CIFAR-10 | Configurable depth, ReLU/GELU, BatchNorm, Dropout |
| CNN | `--model cnn` | MNIST, CIFAR-10 | LeNet-style (MNIST) / SimpleCNN with Kaiming init (CIFAR-10) |
| VGG | `--model vgg` | CIFAR-10 | VGG-11/13/16/19 with BatchNorm |
| ResNet | `--model resnet` | CIFAR-10 | Configurable blocks (default: ResNet-18) |
| ResNet-18 pretrained | `--transfer_mode resizeFreeze` | CIFAR-10 | ImageNet weights, resize to 224, frozen backbone, FC only |
| ResNet-18 pretrained | `--transfer_mode modifyFinetune` | CIFAR-10 | ImageNet weights, adapted conv1 for 32×32, full fine-tune |

---

## Features

- **Multi-dataset**: MNIST and CIFAR-10 (auto-downloaded)
- **Training utilities**: Adam optimizer, L1 + L2 regularization, early stopping
- **LR schedulers**: StepLR, CosineAnnealingLR
- **Reproducibility**: global seed for `random`, `numpy`, `torch`, and `cudnn`
- **GPU support**: CUDA / MPS / CPU auto-detection (`--device auto`)
- **Plotting** (`--plot`): saves training curves and confusion matrix to `plots/`
- **Structured logger** (`--log`): formatted epoch table saved to `logs/`
- **Transfer learning**: ResNet-18 pretrained with freeze or full fine-tune modes
- **Knowledge distillation**: Hinton KD — soft + hard loss with temperature scaling (`--distill`)

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

### Train + Test
```bash
python main.py --mode both --dataset mnist --model mlp
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `both` | `train`, `test`, or `both` |
| `--dataset` | `mnist` | `mnist` or `cifar10` |
| `--model` | `mlp` | `mlp`, `cnn`, `vgg`, `resnet` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--transfer_mode` | `none` | `none`, `resizeFreeze`, `modifyFinetune` |
| `--log` / `--no-log` | `True` | Save training log to `logs/` |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--scheduler` | `step` | `step`, `cosine`, or `none` |
| `--warmup_epochs` | `0` | Linear LR warmup epochs before cosine decay (0 = disabled) |
| `--patience` | `0` | Early stopping patience (0 = disabled) |
| `--weight_decay` | `1e-4` | L2 regularization coefficient |
| `--l1_lambda` | `0.0` | L1 regularization coefficient |
| `--label_smoothing` | `0.0` | Label smoothing epsilon for CrossEntropyLoss |
| `--plot` | `False` | Save training curves and confusion matrix to `plots/` |
| `--seed` | `42` | Global random seed |
| `--distill` | `False` | Train with Hinton knowledge distillation |
| `--teacher_path` | `teachers/resnet_teacher.pth` | Path to saved teacher weights |
| `--temperature` | `4.0` | Distillation temperature T |
| `--alpha` | `0.7` | Weight for soft KD loss (1-alpha for hard CE) |
| `--count_flops` | `False` | Print MACs and param count via ptflops |

### Model-specific Arguments

**MLP:**
```bash
--hidden_sizes 512 256 128   # hidden layer widths
--dropout 0.3
--activation relu             # relu or gelu
```

**VGG:**
```bash
--vgg_depth 16                # 11, 13, 16, or 19
```

**ResNet:**
```bash
--resnet_layers 2 2 2 2       # blocks per stage (default = ResNet-18)
```

---

## Examples

```bash
# MLP on MNIST with GPU and plots
python main.py --mode both --dataset mnist --model mlp \
               --epochs 20 --lr 1e-3 --plot

# ResNet-18 on CIFAR-10 with cosine scheduler and early stopping
python main.py --mode both --dataset cifar10 --model resnet \
               --epochs 50 --lr 1e-3 --scheduler cosine \
               --patience 10 --plot

# VGG-16 on CIFAR-10
python main.py --mode both --dataset cifar10 --model vgg \
               --vgg_depth 16 --epochs 30 --plot

# Transfer learning — freeze backbone, train FC only (resize CIFAR to 224)
python main.py --dataset cifar10 --transfer_mode resizeFreeze \
               --epochs 10 --batch_size 128 --plot --log

# Transfer learning — adapt first conv for 32x32, fine-tune all layers
python main.py --dataset cifar10 --transfer_mode modifyFinetune \
               --epochs 10 --batch_size 128 --lr 1e-4 --plot --log

# Knowledge distillation — SimpleCNN student, ResNet teacher
# (copy best ResNet weights to teachers/resnet_teacher.pth first)
python main.py --dataset cifar10 --model cnn --distill \
               --teacher_path teachers/resnet_teacher.pth \
               --temperature 4.0 --alpha 0.7 \
               --epochs 20 --lr 1e-3 --batch_size 64 \
               --scheduler cosine --weight_decay 1e-4 \
               --mode both --plot --count_flops
```

---

## Project Structure

```
Deep-Learning/
├── main.py           # Entry point: model build, transfer learning, train/test dispatch
├── train.py          # Training loop, validation, LR schedulers, data loaders, transforms
├── test.py           # Test evaluation with per-class accuracy
├── plot.py           # Training curves and confusion matrix (saved to plots/)
├── logger.py         # Structured epoch table logger (terminal + logs/)
├── parameters.py     # Dataclasses and argparse for all hyperparameters
├── pretrained.py     # Standalone pretrained ResNet-18 eval script
├── NN_Visualizer.py  # torchviz architecture graph for MLP
├── models/
│   ├── MLP.py        # Multi-Layer Perceptron
│   ├── CNN.py        # LeNet-style CNN (MNIST) / SimpleCNN (CIFAR-10)
│   ├── VGG.py        # VGG-11/13/16/19
│   └── ResNet.py     # ResNet with BasicBlock
├── teachers/         # Gitignored — place teacher .pth weights here
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy >= 1.24
- matplotlib >= 3.7
- ptflops >= 0.7 *(for FLOPs counting)*
- seaborn *(optional, for nicer confusion matrix)*
```


