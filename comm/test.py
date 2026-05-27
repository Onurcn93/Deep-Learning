"""
comm/test.py — Final evaluation of a trained CommSystem.

Reports Symbol Error Rate (SER) and Message Error Rate (MER) over a large
number of random test batches.

SER  = fraction of individual symbols decoded incorrectly  (fine-grained)
MER  = fraction of complete 4-symbol messages decoded incorrectly (practical)

Random-chance baseline:
    SER = 7/8 = 0.875  (8 equally likely symbols, guess one → 1/8 correct)
    MER = 1 - (1/8)^4 = 1 - 1/4096 ≈ 0.9998
"""

import torch


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    label: str = 'Test',
    logger=None,
) -> tuple:
    """
    Evaluate CommSystem on n_batches * batch_size random messages.

    Returns
    -------
    ser : float   symbol error rate
    mer : float   message error rate
    """
    model.eval()
    total_correct_sym = 0
    total_correct_msg = 0
    total_symbols  = 0
    total_messages = 0

    for _ in range(n_batches):
        m = torch.randint(0, 8, (batch_size, 4), device=device)
        logits = model(m)                           # (B, 4, 8)
        preds  = logits.argmax(dim=-1)              # (B, 4)

        correct_sym = preds == m                    # (B, 4) bool
        total_correct_sym += correct_sym.sum().item()
        total_correct_msg += correct_sym.all(dim=-1).sum().item()
        total_symbols  += batch_size * 4
        total_messages += batch_size

    ser = 1.0 - total_correct_sym / total_symbols
    mer = 1.0 - total_correct_msg / total_messages

    lines = [
        '',
        '=' * 48,
        f'  {label} Evaluation',
        '=' * 48,
        f'  Messages evaluated  : {total_messages:,}',
        f'  Symbol Error Rate   : {ser:.4f}  ({ser * 100:.2f}%)',
        f'  Message Error Rate  : {mer:.4f}  ({mer * 100:.2f}%)',
        f'  -----------------------------------------',
        f'  Random-chance SER   : {7/8:.4f}  (baseline)',
        f'  Random-chance MER   : {1 - 1/4096:.4f}  (baseline)',
        '=' * 48,
        '',
    ]
    for line in lines:
        print(line)
        if logger is not None:
            logger.log(line)

    return ser, mer
