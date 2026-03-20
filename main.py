import random
import ssl

import numpy as np
import torch
import torch.nn as nn

from torchvision import models as tv_models

from parameters import get_params, DataParams, ModelParams, TrainingParams
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.MobileNet import MobileNetV2
from train import run_training
from test  import run_test
from logger import TrainLogger


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_pretrained_model(
    model_params: ModelParams,
    num_classes:  int = 10,
) -> nn.Module:
    """Load a pretrained ResNet-18 and adapt it for transfer learning.

    Args:
        model_params: Architecture parameters (transfer_mode).
        num_classes:  Number of output classes.

    Returns:
        Adapted ``nn.Module`` ready for training.
    """
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    if model_params.transfer_mode == "resizeFreeze":
        # Freeze entire backbone — only the new FC head will train
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_params.transfer_mode == "modifyFinetune":
        # Replace first conv to accept 32×32 input (no aggressive downsampling)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(model.fc.in_features, num_classes)
        # All layers fine-tune (requires_grad=True by default)

    return model


def build_model(
    data_params:  DataParams,
    model_params: ModelParams,
) -> nn.Module:
    """Instantiate the requested model architecture.

    Args:
        data_params:  Dataset parameters (input size, num classes, dataset name).
        model_params: Architecture parameters.

    Returns:
        An ``nn.Module`` instance ready for training.

    Raises:
        ValueError: If the model/dataset combination is unsupported.
    """
    name = model_params.model
    nc   = data_params.num_classes

    if name == "mlp":
        return MLP(
            input_size   = data_params.input_size,
            hidden_sizes = model_params.hidden_sizes,
            num_classes  = nc,
            dropout      = model_params.dropout,
            activation   = model_params.activation,
        )

    if name == "cnn":
        # MNIST_CNN expects 1-channel 28×28; SimpleCNN expects 3-channel 32×32
        if data_params.dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        return SimpleCNN(num_classes=nc)

    if name == "vgg":
        if data_params.dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=model_params.vgg_depth, num_class=nc)

    if name == "resnet":
        if data_params.dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, model_params.resnet_layers, num_classes=nc)

    if name == "mobilenet":
        if data_params.dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10 with mobilenet.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {name}")


def build_config_title(
    data_params:     DataParams,
    model_params:    ModelParams,
    training_params: TrainingParams,
) -> str:
    """Build a short human-readable title describing the current experiment setup."""
    parts = []

    if model_params.transfer_mode != "none":
        parts.append(f"ResNet18-pretrained | {model_params.transfer_mode}")
    else:
        name = model_params.model
        if name == "mlp":
            arch = "\u00d7".join(str(h) for h in model_params.hidden_sizes)
            parts.append(f"MLP {arch}")
            parts.append(f"drop={model_params.dropout}")
            parts.append(model_params.activation)
        elif name == "cnn":
            parts.append("CNN")
        elif name == "vgg":
            parts.append(f"VGG-{model_params.vgg_depth}")
        elif name == "resnet":
            parts.append(f"ResNet {model_params.resnet_layers}")
        elif name == "mobilenet":
            parts.append("MobileNetV2")
    parts.append(data_params.dataset)
    parts.append(f"lr={training_params.learning_rate}")
    parts.append(f"bs={training_params.batch_size}")
    sched_str = training_params.scheduler
    if training_params.warmup_epochs > 0:
        sched_str += f"+warmup{training_params.warmup_epochs}"
    parts.append(f"sched={sched_str}")
    if training_params.weight_decay > 0:
        parts.append(f"wd={training_params.weight_decay}")
    if training_params.distill:
        parts.append(f"KD | T={training_params.temperature} | alpha={training_params.alpha}")

    return " | ".join(parts)


def main() -> None:
    """Entry point: parse parameters, build model, run training and/or testing."""
    data_params, model_params, training_params = get_params()

    set_seed(training_params.seed)
    print(f"Seed set to: {training_params.seed}")
    print(f"Dataset: {data_params.dataset}  |  Model: {model_params.model}")

    if training_params.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(training_params.device)
    print(f"Using device: {device}")

    if model_params.transfer_mode != "none":
        model = build_pretrained_model(model_params, data_params.num_classes).to(device)
    else:
        model = build_model(data_params, model_params).to(device)
    print(model)

    config_title = build_config_title(data_params, model_params, training_params)
    logger = TrainLogger(experiment=config_title, enabled=training_params.log)

    # FLOPs count (ptflops) — logged to file and terminal
    if training_params.count_flops:
        try:
            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(
                model, (3, 32, 32), as_strings=True, print_per_layer_stat=False, verbose=False,
            )
            logger._w(f"\nModel complexity — MACs: {macs}  |  Params: {params}\n")
        except ImportError:
            logger._w("ptflops not installed — run: pip install ptflops")

    # Load teacher for knowledge distillation
    teacher = None
    if training_params.distill:
        teacher_model_params = ModelParams(
            model         = "resnet",
            hidden_sizes  = [512, 256, 128],
            dropout       = 0.3,
            activation    = "relu",
            vgg_depth     = "16",
            resnet_layers = [2, 2, 2, 2],
            transfer_mode = "none",
        )
        teacher = build_model(data_params, teacher_model_params).to(device)
        teacher.load_state_dict(torch.load(training_params.teacher_path, map_location=device))
        teacher.eval()
        logger._w(f"Teacher loaded from: {training_params.teacher_path}")

    if training_params.mode in ("train", "both"):
        run_training(model, data_params, model_params, training_params, device, config_title, logger, teacher)

    if training_params.mode in ("test", "both"):
        run_test(model, data_params, model_params, training_params, device, config_title)


if __name__ == "__main__":
    main()
