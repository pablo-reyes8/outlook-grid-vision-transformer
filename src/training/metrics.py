import gc
from typing import Dict

import torch


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5)) -> Dict[int, float]:
    """
    targets can be:
      - int64 class indices [B]
      - soft targets [B, num_classes] (we'll argmax for accuracy reporting)
    """
    if targets.ndim == 2:
        targets = targets.argmax(dim=1)

    max_k = max(ks)
    B = targets.size(0)
    _, pred = torch.topk(logits, k=max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    out = {}
    for k in ks:
        out[k] = 100.0 * correct[:, :k].any(dim=1).float().sum().item() / B
    return out


def free_all_cuda(*names, verbose=True, globals_dict=None, locals_dict=None):
    """
    Borra variables por nombre (strings) de globals/locals para evitar referencias colgadas en notebooks.
    """
    if globals_dict is None: globals_dict = globals()
    if locals_dict is None:  locals_dict  = locals()

    for n in names:
        if n in locals_dict:
            del locals_dict[n]
        if n in globals_dict:
            del globals_dict[n]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if verbose and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        res   = torch.cuda.memory_reserved() / 1024**2
        print(f"[CUDA] allocated={alloc:.1f} MB | reserved(cache)={res:.1f} MB")
