import random
import ssl

import numpy as np
import torch
import torch.nn as nn

from parameters import get_params, DataParams, ModelParams, TrainingParams
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from train import run_training
from test  import run_test


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

    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    """Entry point: parse parameters, build model, run training and/or testing."""
    data_params, model_params, training_params = get_params()

    set_seed(training_params.seed)
    print(f"Seed set to: {training_params.seed}")
    print(f"Dataset: {data_params.dataset}  |  Model: {model_params.model}")

    device = torch.device(
        training_params.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(data_params, model_params).to(device)
    print(model)

    if training_params.mode in ("train", "both"):
        run_training(model, data_params, training_params, device)

    if training_params.mode in ("test", "both"):
        run_test(model, data_params, training_params, device)


if __name__ == "__main__":
    main()
