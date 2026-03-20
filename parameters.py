import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataParams:
    """Parameters for dataset loading and preprocessing.

    Attributes:
        dataset:     Dataset name (``'mnist'`` or ``'cifar10'``).
        data_dir:    Root directory for downloaded data.
        num_workers: Number of DataLoader worker processes.
        mean:        Per-channel normalisation mean.
        std:         Per-channel normalisation std.
        input_size:  Flattened feature dimension (784 or 3072).
        num_classes: Number of target classes.
    """

    dataset:     str
    data_dir:    str
    num_workers: int
    mean:        Tuple[float, ...]
    std:         Tuple[float, ...]
    input_size:  int
    num_classes: int


@dataclass
class ModelParams:
    """Parameters defining the neural-network architecture.

    Attributes:
        model:         Architecture name (``'mlp'``, ``'cnn'``, ``'vgg'``, ``'resnet'``).
        hidden_sizes:  List of hidden layer widths (MLP only).
        dropout:       Dropout probability (MLP only).
        activation:    Activation function (``'relu'`` or ``'gelu'``, MLP only).
        vgg_depth:     VGG variant depth string (``'11'``/``'13'``/``'16'``/``'19'``).
        resnet_layers: Blocks per ResNet stage (4-element list).
    """

    model:         str
    hidden_sizes:  List[int]
    dropout:       float
    activation:    str
    vgg_depth:     str
    resnet_layers: List[int]
    transfer_mode: str   # 'none' | 'resizeFreeze' | 'modifyFinetune'


@dataclass
class TrainingParams:
    """Parameters controlling the training process.

    Attributes:
        mode:          Run mode (``'train'``, ``'test'``, or ``'both'``).
        epochs:        Maximum number of training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Initial learning rate.
        weight_decay:  L2 regularisation coefficient (Adam ``weight_decay``).
        l1_lambda:     L1 regularisation coefficient (added to loss manually).
        scheduler:     LR scheduler type (``'step'``, ``'cosine'``, or ``'none'``).
        patience:      Early-stopping patience in epochs (0 = disabled).
        save_path:     File path for saving the best model weights.
        log_interval:  Batches between training-progress prints.
        seed:          Global random seed.
        device:        Requested compute device string (e.g. ``'cuda'``, ``'cpu'``).
        plot:          If ``True``, save training curves and confusion matrix to ``plots/``.
    """

    mode:            str
    epochs:          int
    batch_size:      int
    learning_rate:   float
    weight_decay:    float
    l1_lambda:       float
    label_smoothing: float
    scheduler:       str
    patience:      int
    save_path:     str
    log_interval:  int
    seed:          int
    device:        str
    plot:          bool
    log:           bool


def get_params() -> Tuple[DataParams, ModelParams, TrainingParams]:
    """Parse CLI arguments and return three grouped parameter dataclasses.

    Returns:
        Tuple of ``(DataParams, ModelParams, TrainingParams)``.
    """
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    # General
    parser.add_argument("--mode",    choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model",   choices=["mlp", "cnn", "vgg", "resnet"], default="mlp")
    parser.add_argument("--device",  type=str,  default="auto", help="Device: auto detects cuda/mps/cpu, or specify explicitly")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--plot",    action="store_true",
                        help="Save training curves and confusion matrix to plots/")
    parser.add_argument("--log",     action=argparse.BooleanOptionalAction, default=True,
                        help="Save training log to logs/ (default: True, disable with --no-log)")

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularisation coefficient (optimizer weight_decay)")
    parser.add_argument("--l1_lambda",       type=float, default=0.0,
                        help="L1 regularisation coefficient (added to loss)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon for CrossEntropyLoss (0 = disabled)")
    parser.add_argument("--scheduler",    choices=["step", "cosine", "none"], default="step",
                        help="LR scheduler type")
    parser.add_argument("--patience",     type=int,   default=0,
                        help="Early-stopping patience in epochs (0 = disabled)")

    # MLP-specific
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--hidden_sizes", type=int,   nargs="+", default=[512, 256, 128],
                        metavar="H", help="Hidden layer widths for MLP")
    parser.add_argument("--activation",   choices=["relu", "gelu"], default="relu")

    # Transfer learning
    parser.add_argument("--transfer_mode", choices=["none", "resizeFreeze", "modifyFinetune"],
                        default="none",
                        help="Transfer learning mode: resizeFreeze (resize to 224, freeze backbone) "
                             "or modifyFinetune (adapt first conv for 32x32, fine-tune all)")

    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")

    # ResNet-specific
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Blocks per ResNet stage (default: 2 2 2 2 = ResNet-18)")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean: Tuple[float, ...] = (0.1307,)
        std:  Tuple[float, ...] = (0.3081,)
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    data_params = DataParams(
        dataset     = args.dataset,
        data_dir    = "./data",
        num_workers = 2,
        mean        = mean,
        std         = std,
        input_size  = input_size,
        num_classes = 10,
    )

    model_params = ModelParams(
        model         = args.model,
        hidden_sizes  = args.hidden_sizes,
        dropout       = args.dropout,
        activation    = args.activation,
        vgg_depth     = args.vgg_depth,
        resnet_layers = args.resnet_layers,
        transfer_mode = args.transfer_mode,
    )

    training_params = TrainingParams(
        mode          = args.mode,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = args.lr,
        weight_decay     = args.weight_decay,
        l1_lambda        = args.l1_lambda,
        label_smoothing  = args.label_smoothing,
        scheduler        = args.scheduler,
        patience      = args.patience,
        save_path     = "best_model.pth",
        log_interval  = 100,
        seed          = args.seed,
        device        = args.device,
        plot          = args.plot,
        log           = args.log,
    )

    return data_params, model_params, training_params
