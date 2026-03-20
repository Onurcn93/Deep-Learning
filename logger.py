"""Training logger: prints formatted experiment info and saves to logs/."""

import os
import time

LOGS_DIR = "logs"
SEP  = "=" * 72
LINE = "─" * 72
HEADER_FMT = (
    " {:>5} | {:>7} | {:>6} | {:>7} | {:>6}"
)
ROW_FMT = (
    " {:>5} | {:>7.4f} | {:>6.3f} | {:>7.4f} | {:>6.3f}"
)


class TrainLogger:
    def __init__(self, experiment: str, enabled: bool = True, out_dir: str = LOGS_DIR):
        self.experiment = experiment
        self.enabled    = enabled
        self._file      = None
        self._start     = None

        if enabled:
            os.makedirs(out_dir, exist_ok=True)
            safe = experiment.replace(" | ", "_").replace("=", "").replace(" ", "_")
            safe = "".join(c for c in safe if c not in r'\/:*?"<>|')
            path = os.path.join(out_dir, f"{safe}.log")
            self._file = open(path, "w", encoding="utf-8")

    # ------------------------------------------------------------------ #
    def _w(self, line: str = "") -> None:
        """Always print to stdout; also write to file when logging is enabled."""
        print(line)
        if self.enabled and self._file:
            self._file.write(line + "\n")
            self._file.flush()

    # ------------------------------------------------------------------ #
    def log_start(self, model, data_params, model_params, training_params, device=None) -> None:
        """Print experiment header before training begins."""
        self._start = time.time()

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct       = 100.0 * trainable / total if total else 0.0

        if model_params.transfer_mode != "none":
            arch = f"ResNet-18 (pretrained) → Linear(512, {data_params.num_classes})  [{model_params.transfer_mode}]"
        else:
            arch = f"{model_params.model.upper()}"
            if model_params.model == "mlp":
                arch += f"  [{' → '.join(str(h) for h in model_params.hidden_sizes)}]"
            elif model_params.model == "vgg":
                arch += f"-{model_params.vgg_depth}"
            elif model_params.model == "resnet":
                arch += f"  layers={model_params.resnet_layers}"
            elif model_params.model == "mobilenet":
                arch = "MobileNetV2  [stride-1 stem for 32×32]"

        self._w()
        self._w(f"▶▶▶  Starting  {self.experiment}")
        self._w(f"Architecture : {arch}")
        self._w(f"Parameters   : {trainable:,} trainable / {total:,} total ({pct:.1f}%)")
        self._w()
        self._w(SEP)
        self._w(f"  Experiment : {self.experiment}")
        self._w(f"  Dataset    : {data_params.dataset.upper()}")
        self._w(f"  Epochs     : {training_params.epochs}"
                f"  |  Batch: {training_params.batch_size}"
                f"  |  Device: {device if device else training_params.device}")
        sched_str = training_params.scheduler
        if training_params.warmup_epochs > 0:
            sched_str += f" (warmup={training_params.warmup_epochs})"
        self._w(f"  LR         : {training_params.learning_rate}"
                f"  |  Scheduler: {sched_str}"
                f"  |  WD: {training_params.weight_decay}")
        if training_params.label_smoothing > 0:
            self._w(f"  Label smooth: eps={training_params.label_smoothing}")
        if training_params.distill:
            if training_params.distill_mode == "teacher_prob":
                self._w(f"  Distillation: teacher_prob"
                        f"  |  teacher={training_params.teacher_path}")
            else:
                self._w(f"  Distillation: hinton | T={training_params.temperature}"
                        f"  |  alpha={training_params.alpha}"
                        f"  |  teacher={training_params.teacher_path}")
        if training_params.patience > 0:
            self._w(f"  Early stop : patience={training_params.patience}")
        self._w(f"  Save by    : val acc")
        self._w(SEP)
        self._w()
        self._w(HEADER_FMT.format("Epoch", "TrLoss", "TrAcc", "VaLoss", "VaAcc"))
        self._w(LINE)

    def log_epoch(
        self,
        epoch: int,
        tr_loss: float,
        tr_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        self._w(ROW_FMT.format(epoch, tr_loss, tr_acc, val_loss, val_acc))

    def log_best(self, val_acc: float, save_path: str) -> None:
        self._w(f"      ↑ New best val acc: {val_acc:.4f} — checkpoint saved → {save_path}")

    def log_complete(self, best_val: float, save_path: str) -> None:
        elapsed = time.time() - self._start if self._start else 0.0
        mins, secs = divmod(int(elapsed), 60)
        self._w()
        self._w(f"✓  Training complete  |  Best val acc: {best_val:.4f}")
        self._w(f"   Time: {elapsed:.0f}s ({mins}m {secs:02d}s)"
                f"  |  Checkpoint: {save_path}")
        self._w()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None