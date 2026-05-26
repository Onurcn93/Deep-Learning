"""
forecaster/test.py — Evaluation for HW4 financial forecasting.

sub-task b/c : per-horizon MSE + MAE table
sub-task d   : accuracy, precision, recall, F1
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def run_test(
    model      : nn.Module,
    loader     : DataLoader,
    device     : torch.device,
    task       : str = 'b',    # 'b'|'c' for regression, 'd' for classification
    D          : int = 5,
    logger     = None,
) -> dict:
    """
    Evaluate on the test loader and print a results table.

    Returns a dict with all computed metrics.
    """
    model.eval()

    all_preds  = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        all_preds.append(pred.cpu())
        all_labels.append(y)

    preds  = torch.cat(all_preds,  dim=0)   # (N, D) or (N,)
    labels = torch.cat(all_labels, dim=0)   # (N, D) or (N,)

    results = {}

    def _log(msg):
        print(msg)
        if logger:
            logger.log(msg)

    sep = '-' * 52

    if task in ('b', 'c'):
        # ── Regression: per-horizon MSE + MAE ────────────────────────────────
        _log(f'\n{"Test Results (regression)":^52}')
        _log(sep)
        _log(f'  {"Horizon":>8}  {"MSE":>12}  {"MAE":>12}')
        _log(sep)

        per_d_mse = []
        per_d_mae = []
        for d in range(D):
            mse = float(((preds[:, d] - labels[:, d]) ** 2).mean())
            mae = float((preds[:, d] - labels[:, d]).abs().mean())
            per_d_mse.append(mse)
            per_d_mae.append(mae)
            _log(f'  {"d=" + str(d+1):>8}  {mse:>12.6f}  {mae:>12.6f}')

        overall_mse = float(((preds - labels) ** 2).mean())
        overall_mae = float((preds - labels).abs().mean())

        _log(sep)
        _log(f'  {"Overall":>8}  {overall_mse:>12.6f}  {overall_mae:>12.6f}')
        _log(sep)

        results['per_d_mse']   = per_d_mse
        results['per_d_mae']   = per_d_mae
        results['overall_mse'] = overall_mse
        results['overall_mae'] = overall_mae

        # Store raw arrays for plotting
        results['preds']  = preds.numpy()
        results['labels'] = labels.numpy()

    else:
        # ── Classification: accuracy, precision, recall, F1 ──────────────────
        probs     = torch.sigmoid(preds)
        predicted = (probs >= 0.5).long()
        labels_i  = labels.long()

        tp = ((predicted == 1) & (labels_i == 1)).sum().item()
        fp = ((predicted == 1) & (labels_i == 0)).sum().item()
        tn = ((predicted == 0) & (labels_i == 0)).sum().item()
        fn = ((predicted == 0) & (labels_i == 1)).sum().item()

        accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        buy_rate  = labels_i.float().mean().item()

        _log(f'\n{"Test Results (turning-point detection)":^52}')
        _log(sep)
        _log(f'  Buy rate in test set : {buy_rate*100:.1f}%')
        _log(sep)
        _log(f'  {"Metric":>14}  {"Value":>10}')
        _log(sep)
        for name, val in [
            ('Accuracy',  accuracy),
            ('Precision', precision),
            ('Recall',    recall),
            ('F1',        f1),
        ]:
            _log(f'  {name:>14}  {val:>10.4f}')
        _log(sep)
        _log(f'  Confusion matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}')
        _log(sep)

        results.update(dict(
            accuracy=accuracy, precision=precision,
            recall=recall, f1=f1,
            tp=tp, fp=fp, tn=tn, fn=fn,
            # raw arrays for ROC / PR curve plotting
            probs     = probs.numpy(),
            labels_np = labels_i.numpy(),
        ))

    return results
