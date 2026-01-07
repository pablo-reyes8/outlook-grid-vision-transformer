import math 
import torch 
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from collections import Counter
import math


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def describe_loader(loader, name="loader", max_batches_for_stats=50):
    ds = loader.dataset
    n = len(ds)

    print("\n" + "="*90)
    print(f"{name.upper()} SUMMARY")
    print("="*90)

    print(f"Dataset type        : {type(ds).__name__}")
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        print(f"  ↳ Wrapped dataset  : {type(ds.dataset).__name__} (Subset-like)")
        print(f"  ↳ Subset size      : {len(ds.indices)}")

    print(f"Num samples         : {n}")
    print(f"Batch size          : {getattr(loader, 'batch_size', None)}")
    print(f"Num workers         : {getattr(loader, 'num_workers', None)}")
    print(f"Pin memory          : {getattr(loader, 'pin_memory', None)}")
    print(f"Drop last           : {getattr(loader, 'drop_last', None)}")

    sampler = getattr(loader, "sampler", None)
    sampler_name = type(sampler).__name__ if sampler is not None else None
    print(f"Sampler             : {sampler_name}")

    num_batches = len(loader)
    bs = loader.batch_size if loader.batch_size is not None else "?"
    approx_batches = math.ceil(n / loader.batch_size) if loader.batch_size else "?"
    print(f"len(loader) (#batches): {num_batches} (≈ ceil({n}/{bs}) = {approx_batches})")

    x, y = next(iter(loader))
    print("\nFirst batch:")
    print(f"  x.shape           : {tuple(x.shape)}")
    print(f"  y.shape           : {tuple(y.shape)}")
    print(f"  x.dtype           : {x.dtype}")
    print(f"  y.dtype           : {y.dtype}")
    print(f"  x.min/max         : {float(x.min()):.4f} / {float(x.max()):.4f}")
    print(f"  y.min/max         : {int(y.min())} / {int(y.max())}")
    print(f"  unique labels (batch): {len(torch.unique(y))}")
    print(f"\nQuick stats over up to {max_batches_for_stats} batches:")

    n_seen = 0
    sum_ = 0.0
    sumsq_ = 0.0
    class_counts = Counter()

    for bi, (xb, yb) in enumerate(loader):
        if bi >= max_batches_for_stats:
            break
        xb = xb.float()
        n_pix = xb.numel()
        sum_ += xb.sum().item()
        sumsq_ += (xb * xb).sum().item()
        n_seen += n_pix

        class_counts.update(yb.tolist())

    mean = sum_ / max(1, n_seen)
    var = (sumsq_ / max(1, n_seen)) - mean**2
    std = math.sqrt(max(0.0, var))

    print(f"  Approx mean        : {mean:.6f}")
    print(f"  Approx std         : {std:.6f}")
    top5 = class_counts.most_common(5)
    print(f"  Seen label counts  : {len(class_counts)} classes (in sampled batches)")
    print(f"  Top-5 labels       : {top5}")

    targets = None
    if hasattr(ds, "targets"):
        targets = ds.targets
    elif hasattr(ds, "labels"):
        targets = ds.labels
    elif hasattr(ds, "dataset") and hasattr(ds.dataset, "targets") and hasattr(ds, "indices"):
        base_targets = ds.dataset.targets
        targets = [base_targets[i] for i in ds.indices]

    if targets is not None:
        full_counts = Counter(list(map(int, targets)))
        k = len(full_counts)
        print(f"\nFull dataset label distribution:")
        print(f"  #classes detected  : {k}")
        if k > 0:
            mn = min(full_counts.values())
            mx = max(full_counts.values())
            print(f"  min/max per class  : {mn} / {mx}")
            first10 = sorted(full_counts.items(), key=lambda t: t[0])[:10]
            print(f"  first 10 classes   : {first10}")
            if mn == mx:
                print("  balance check      : perfectly balanced")
            else:
                print("  balance check      : not perfectly balanced")
    else:
        print("\nFull dataset label distribution: (couldn't find targets/labels attribute)")

    print("="*90)


def unnormalize(images: torch.Tensor,
                mean=CIFAR100_MEAN,
                std=CIFAR100_STD):
    """
    Des-normaliza un batch de imágenes.
    images: tensor [B, C, H, W] normalizado.
    """
    mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


def show_batch(images: torch.Tensor,
               labels: torch.Tensor,
               class_names=None,
               n: int = 8):
    """
    Muestra las primeras n imágenes de un batch con sus labels.

    Args:
        images: tensor [B, C, H, W] (normalizado).
        labels: tensor [B].
        class_names: lista de nombres de clases (len = 100).
        n: cuántas imágenes mostrar (en una fila).
    """
    images = images[:n].cpu()
    labels = labels[:n].cpu()
    images_unnorm = unnormalize(images)

    grid = make_grid(images_unnorm, nrow=n, padding=2)
    npimg = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(2 * n, 2.5))
    plt.imshow(npimg)
    plt.axis("off")

    if class_names is not None:
        title = " | ".join(class_names[int(lbl)] for lbl in labels)
        plt.title(title, fontsize=10)
    plt.show()