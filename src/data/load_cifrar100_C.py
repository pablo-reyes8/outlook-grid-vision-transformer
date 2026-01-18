import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import datasets

def _seed_worker_factory(seed: int):
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


class CIFAR100C_HF(Dataset):
    """
    Wrapper puro sobre un Dataset HF YA cargado.
    Filtra a una sola (corruption, severity) => 10k imgs.
    """
    def __init__(self, base_ds, corruption: str, severity: int, transform=None):
        assert 1 <= int(severity) <= 5
        self.transform = transform
        self.corruption = corruption
        self.severity = int(severity)

        ds = base_ds.filter(
            lambda batch: [
                (c == self.corruption) and (lvl == self.severity)
                for c, lvl in zip(batch["corruption_name"], batch["corruption_level"])
            ],
            batched=True,
            batch_size=10_000,)

        if len(ds) != 10000:
            raise RuntimeError(
                f"Esperaba 10000 ejemplos, obtuve {len(ds)}. "
                f"Revisa corruption='{corruption}' y severity={severity}.")

        # sanity rápido: labels 0..99 en muestra
        sample = ds.select(range(min(512, len(ds))))
        mx = max(int(x["label"]) for x in sample)
        if mx > 99:
            raise RuntimeError(f"Labels fuera de rango: max label en muestra = {mx}")

        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex["image"]         # PIL
        label = int(ex["label"])  # 0..99
        if self.transform is not None:
            img = self.transform(img)
        return img, label



def get_cifar100c_testloader(
    base_ds, 
    corruption: str = "gaussian_noise",
    severity: int = 1,
    batch_size: int = 128,
    img_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 7):
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    ops = []
    if img_size != 32:
        ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),]
    test_transform = transforms.Compose(ops)

    ds = CIFAR100C_HF(
        base_ds=base_ds,
        corruption=corruption,
        severity=int(severity),
        transform=test_transform,)


    dl_gen = torch.Generator().manual_seed(seed)
    worker_init_fn = _seed_worker_factory(seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=dl_gen,
        worker_init_fn=worker_init_fn)

    return loader

def evaluate_tinyc_suite(
    model,
    evaluate_one_epoch_fn,
    reference_clean_loader,
    base_ds,
    batch_size=128,
    img_size=32,
    num_workers=2,
    corruptions=None,
    severities=(1, 2, 3, 4, 5),
    verbose=True,
    seed=7):

    if corruptions is None:
        corruptions = ["gaussian_noise"]

    model.to("cuda")
    results = {}

    if reference_clean_loader is not None:
        clean_loss, clean_m = evaluate_one_epoch_fn(model=model, dataloader=reference_clean_loader, device="cuda")
        results["clean"] = {"loss": float(clean_loss), "metrics": {k: float(v) for k, v in clean_m.items()}}
        if verbose:
            print(f"[Clean] loss {clean_loss:.4f} | " +
                  " | ".join([f"{k} {clean_m[k]:.2f}%" for k in sorted(clean_m.keys())]))

    for corr in corruptions:
        results[corr] = {}
        for s in severities:
            test_loader_c = get_cifar100c_testloader(
                base_ds=base_ds,
                corruption=corr,
                severity=int(s),
                batch_size=batch_size,
                img_size=img_size,
                num_workers=num_workers,
                seed=seed,)
            
            loss_c, m_c = evaluate_one_epoch_fn(model=model, dataloader=test_loader_c, device="cuda")
            results[corr][int(s)] = {"loss": float(loss_c), "metrics": {k: float(v) for k, v in m_c.items()}}

            if verbose:
                msg = f"[{corr:>16} | s={s}] loss {loss_c:.4f} | " + " | ".join(
                    [f"{k} {m_c[k]:.2f}%" for k in sorted(m_c.keys())])
                print(msg)

    return results


def summarize_tinyc_results(results: dict, metric_key: str = "top1") -> str:
    """
    Devuelve un string con resumen: por corrupción promedia severidades, y global.
    metric_key: 'top1', 'top3', 'top5' (según tu evaluate_one_epoch_fn)
    """
    lines = []
    if "clean" in results:
        clean_val = results["clean"]["metrics"].get(metric_key, float("nan"))
        lines.append(f"clean: {clean_val:.2f}% ({metric_key})")

    corr_vals = []
    for corr, by_s in results.items():
        if corr == "clean":
            continue
        vals = []
        for s, pack in by_s.items():
            vals.append(pack["metrics"].get(metric_key, float("nan")))
        avg = float(np.mean(vals)) if len(vals) else float("nan")
        corr_vals.append(avg)
        lines.append(f"{corr}: {avg:.2f}% avg over severities ({metric_key})")

    if len(corr_vals):
        lines.append(f"MEAN over corruptions: {float(np.mean(corr_vals)):.2f}% ({metric_key})")

    return "\n".join(lines)


def verify_cifar100c_matches_torchvision(base_ds, data_dir="./data", corruption="gaussian_noise"):
    tv = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
    y_tv = np.array(tv.targets, dtype=np.int64)

    ds_hf_s1 = base_ds.filter(
        lambda batch: [
            (c == corruption) and (lvl == 1)
            for c, lvl in zip(batch["corruption_name"], batch["corruption_level"])],
        batched=True,
        batch_size=10_000,)

    y_hf = np.array([int(x["label"]) for x in ds_hf_s1], dtype=np.int64)

    print("HF len:", len(y_hf), "TV len:", len(y_tv))
    print("HF label min/max:", y_hf.min(), y_hf.max(), "unique:", len(np.unique(y_hf)))
    print("TV label min/max:", y_tv.min(), y_tv.max(), "unique:", len(np.unique(y_tv)))

    match = (y_hf == y_tv).mean()
    print(f"Label match rate (HF vs torchvision): {match*100:.2f}%")

    if match < 1.0:
        idx = np.where(y_hf != y_tv)[0][:20]
        print("Ejemplos mismatch idx:", idx.tolist())
        print("HF:", y_hf[idx].tolist())
        print("TV:", y_tv[idx].tolist())