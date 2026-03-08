# Deep-learning

```markdown
# Deep Learning Framework

A personal PyTorch framework for training and evaluating deep learning models on image classification benchmarks. Built to be extended with new architectures over time.

---

## Supported Models

| Model | Flag | Dataset | Notes |
|-------|------|---------|-------|
| MLP | `--model mlp` | MNIST, CIFAR-10 | Configurable depth, ReLU/GELU, BatchNorm, Dropout |
| CNN | `--model cnn` | MNIST, CIFAR-10 | LeNet-style (MNIST) / SimpleCNN with Kaiming init (CIFAR-10) |
| VGG | `--model vgg` | CIFAR-10 | VGG-11/13/16/19 with BatchNorm |
| ResNet | `--model resnet` | CIFAR-10 | Configurable blocks (default: ResNet-18) |

---

## Features

- **Multi-dataset**: MNIST and CIFAR-10 (auto-downloaded)
- **Training utilities**: Adam optimizer, L1 + L2 regularization, early stopping
- **LR schedulers**: StepLR, CosineAnnealingLR
- **Reproducibility**: global seed for `random`, `numpy`, `torch`, and `cudnn`
- **GPU support**: CUDA / MPS / CPU auto-detection
- **Plotting** (`--plot`): saves training curves and confusion matrix to `plots/`
- **Pretrained models**: `pretrained.py` for fine-tuning torchvision models

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
| `--device` | `cpu` | `cuda`, `mps`, or `cpu` |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--scheduler` | `step` | `step`, `cosine`, or `none` |
| `--patience` | `0` | Early stopping patience (0 = disabled) |
| `--weight_decay` | `1e-4` | L2 regularization coefficient |
| `--l1_lambda` | `0.0` | L1 regularization coefficient |
| `--plot` | `False` | Save training curves and confusion matrix to `plots/` |
| `--seed` | `42` | Global random seed |

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
               --epochs 20 --lr 1e-3 --device cuda --plot

# ResNet-18 on CIFAR-10 with cosine scheduler and early stopping
python main.py --mode both --dataset cifar10 --model resnet \
               --epochs 50 --lr 1e-3 --scheduler cosine \
               --patience 10 --device cuda --plot

# VGG-16 on CIFAR-10
python main.py --mode both --dataset cifar10 --model vgg \
               --vgg_depth 16 --epochs 30 --device cuda --plot
```

---

## Project Structure

```
Deep-Learning/
├── main.py           # Entry point: argument parsing, model build, train/test dispatch
├── train.py          # Training loop, validation, LR schedulers, data loaders
├── test.py           # Test evaluation with per-class accuracy
├── plot.py           # Training curves and confusion matrix (saved to plots/)
├── parameters.py     # Dataclasses and argparse for all hyperparameters
├── pretrained.py     # Fine-tuning with torchvision pretrained models
├── models/
│   ├── MLP.py        # Multi-Layer Perceptron
│   ├── CNN.py        # LeNet-style CNN (MNIST) / SimpleCNN (CIFAR-10)
│   ├── VGG.py        # VGG-11/13/16/19
│   └── ResNet.py     # ResNet with BasicBlock
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy >= 1.24
- matplotlib
- seaborn *(optional, for nicer confusion matrix)*
```
