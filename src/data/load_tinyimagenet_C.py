import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torchvision.datasets.folder import default_loader
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
TINYC_URL = "https://zenodo.org/records/8206060/files/Tiny-ImageNet-C.tar?download=1"


def _unwrap_dataset(ds: Dataset) -> Dataset:
    """Quita capas de Subset para llegar al dataset base."""
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def extract_class_names_from_loader(loader: DataLoader) -> Optional[List[str]]:
    """
    Intenta extraer class_names (wnids) desde tu dataset HF wrapper
    a partir de un DataLoader (aunque sea Subset).
    """
    base = _unwrap_dataset(loader.dataset)
    return getattr(base, "class_names", None)

def build_class_to_idx_from_clean_loader(clean_loader: DataLoader) -> dict:
    """
    Devuelve {wnid: idx} compatible con TinyImageNet-C (carpetas n0xxxxxxx).
    Intenta varias estrategias basadas en lo que trae el loader HF.
    """
    base = _unwrap_dataset(clean_loader.dataset)

    names = getattr(base, "class_names", None)
    if names is not None and len(names) == 200 and all(isinstance(x, str) and x.startswith("n") for x in names):
        return {wnid: i for i, wnid in enumerate(names)}

    # Si el HF split trae una columna "wnid" o "class_id", construimos mapping label->wnid
    hf_ds = getattr(base, "ds", None)
    if hf_ds is None:
        raise RuntimeError("No pude acceder al HF dataset interno desde el loader (base.ds no existe).")

    cols = set(getattr(hf_ds, "column_names", []))

    # prueba columnas comunes
    wnid_col = None
    for cand in ["wnid", "class_id", "id", "synset", "nid"]:
        if cand in cols:
            wnid_col = cand
            break

    if wnid_col is None:
        raise RuntimeError(
            f"No encontré columna wnid/class_id en HF dataset. Columnas disponibles: {sorted(list(cols))}")

    # construye label->wnid leyendo ejemplos hasta completar 200 labels
    label_to_wnid = {}
    for ex in hf_ds:
        y = int(ex["label"])
        if y not in label_to_wnid:
            label_to_wnid[y] = str(ex[wnid_col])
            if len(label_to_wnid) == 200:
                break

    if len(label_to_wnid) != 200:
        raise RuntimeError(f"Solo pude mapear {len(label_to_wnid)} labels a wnids. Algo raro con el HF dataset.")

    wnids_in_label_order = [label_to_wnid[i] for i in range(200)]
    if not all(w.startswith("n") for w in wnids_in_label_order):
        raise RuntimeError(
            f"Los ids encontrados no parecen WNIDs. Ejemplo: {wnids_in_label_order[:10]}")

    return {wnid: i for i, wnid in enumerate(wnids_in_label_order)}


def _looks_like_tinyc_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    
    for corr in root.iterdir():
        if corr.is_dir():
            for lvl in ["1", "2", "3", "4", "5"]:
                p = corr / lvl
                if p.is_dir() and any(x.is_dir() for x in p.iterdir()):
                    return True
    return False


def find_tinyimagenet_c_root(search_dir: Path) -> Optional[Path]:
    # comunes
    for cand in [search_dir / "Tiny-ImageNet-C", search_dir / "Tiny-Imagenet-C", search_dir]:
        if cand.exists() and _looks_like_tinyc_root(cand):
            return cand
    # búsqueda recursiva (poco profunda)
    for cand in search_dir.glob("**/*"):
        if cand.is_dir() and _looks_like_tinyc_root(cand):
            return cand
    return None


def download_and_extract_tiny_imagenet_c(data_dir: str = "./data", force: bool = False) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "Tiny-ImageNet-C.tar"

    # si ya está extraído, úsalo
    existing = find_tinyimagenet_c_root(data_dir)
    if existing is not None and not force:
        return existing

    if force or (not tar_path.exists()):
        print(f"[Tiny-ImageNet-C] Downloading tar to: {tar_path}")
        urllib.request.urlretrieve(TINYC_URL, tar_path)

    print(f"[Tiny-ImageNet-C] Extracting: {tar_path}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(data_dir)

    root = find_tinyimagenet_c_root(data_dir)
    if root is None:
        raise RuntimeError("No pude localizar la carpeta raíz de Tiny-ImageNet-C tras extraer el tar.")
    return root


class FileListDataset(Dataset):
    """Dataset con lista explícita de (path, y)."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # list[(path, y)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, int(y)
    

def list_tinyimagenet_c_corruptions_local(data_dir: str = "./data") -> List[str]:
    root = find_tinyimagenet_c_root(Path(data_dir))
    if root is None:
        root = download_and_extract_tiny_imagenet_c(data_dir=data_dir)
    root = Path(root)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


##############################################################################

def _tinyc_test_transform(img_size: int = 64):
    ops = []
    if img_size != 64:
        ops.append(transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),]

    return transforms.Compose(ops)


def get_tinyimagenet200c_loader_intersection(
    batch_size: int,
    data_dir: str,
    corruption_name: str,
    corruption_level: int,
    img_size: int,
    num_workers: int,
    reference_clean_loader: DataLoader,
    pin_memory: bool = True,
    verbose: bool = False,) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Retorna:
      loader_c  (solo ejemplos cuya clase está en el train clean)
      info dict (overlap, only_C, only_train, n_samples, etc.)
    """
    base_clean = _unwrap_dataset(reference_clean_loader.dataset)
    train_wnids = list(base_clean.class_names)  # orden de logits
    train_set = set(train_wnids)
    train_wnid_to_idx = {w: i for i, w in enumerate(train_wnids)}

    root = find_tinyimagenet_c_root(Path(data_dir))
    if root is None:
        root = download_and_extract_tiny_imagenet_c(data_dir=data_dir)

    split_dir = Path(root) / corruption_name / str(int(corruption_level))
    if not split_dir.exists():
        avail = list_tinyimagenet_c_corruptions_local(data_dir=data_dir)
        raise FileNotFoundError(
            f"No existe: {split_dir}\nEjemplo corrupciones disponibles: {avail[:25]}")

    ds_raw = ImageFolder(root=str(split_dir), transform=None)
    c_wnids = list(ds_raw.classes)
    c_set = set(c_wnids)

    overlap = sorted(list(train_set & c_set))
    only_c = sorted(list(c_set - train_set))
    only_train = sorted(list(train_set - c_set))

    # samples_new: solo clases en overlap, remapeadas a idx de train
    samples_new = []
    for path, y_c in ds_raw.samples:
        wnid = c_wnids[y_c]
        if wnid in train_set:
            y_new = train_wnid_to_idx[wnid]
            samples_new.append((path, y_new))

    tfm = _tinyc_test_transform(img_size=img_size)
    ds = FileListDataset(samples_new, transform=tfm)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0))

    info = {
        "corruption": corruption_name,
        "severity": int(corruption_level),
        "overlap_classes": len(overlap),
        "only_c_classes": len(only_c),
        "only_train_classes": len(only_train),
        "n_samples": len(ds)}

    if verbose:
        print(f"[TinyC] {corruption_name} s={corruption_level} | overlap={len(overlap)} | n={len(ds)}")
        if only_c:
            print("  example only_C:", only_c[:10])
        if only_train:
            print("  example only_train:", only_train[:10])

    return loader, info


def _normalize_metrics_from_eval(out):
    """
    Soporta el output exacto de tu evaluate_one_epoch:
      return avg_loss, {"top1":..., "top3":..., "top5":...}
    """
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        loss = float(out[0])
        metrics = {k: float(v) for k, v in out[1].items()}
        return {"loss": loss, **metrics}

    if isinstance(out, dict):
        return {k: float(v) for k, v in out.items()}

    if isinstance(out, (int, float)):
        return {"top1": float(out)}

    raise ValueError(f"Formato de salida no soportado: type={type(out)} out={out}")


def evaluate_tinyc_suite(
    model,
    evaluate_one_epoch_fn,
    reference_clean_loader,
    data_dir: str = "./data",
    batch_size: int = 128,
    img_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    corruptions=None,
    severities=(1, 2, 3, 4, 5),
    verbose: bool = True,):
    if corruptions is None:
        corruptions = list_tinyimagenet_c_corruptions_local(data_dir=data_dir)

    results = []
    t0 = time.time()

    for corr in corruptions:
        for sev in severities:
            loader_c, info = get_tinyimagenet200c_loader_intersection(
                batch_size=batch_size,
                data_dir=data_dir,
                corruption_name=corr,
                corruption_level=sev,
                img_size=img_size,
                num_workers=num_workers,
                reference_clean_loader=reference_clean_loader,
                pin_memory=pin_memory,
                verbose=False,)

            out = evaluate_one_epoch_fn(model=model, dataloader=loader_c)
            metrics = _normalize_metrics_from_eval(out)

            row = {**info, **metrics}
            results.append(row)

            if verbose:
                print(
                    f"{corr:>18s} | s={sev} | n={info['n_samples']}"
                    f" | loss={row['loss']:.4f} | top1={row['top1']:.2f} | top5={row['top5']:.2f}")

    if verbose:
        print(f"\nDone. {len(results)} runs in {(time.time()-t0)/60:.2f} min")

    return results

def summarize_tinyc_results(results, metric_key="top1"):
    overall = []
    by_sev = {}
    by_corr = {}

    for r in results:
        v = r.get(metric_key, None)
        if v is None:
            continue
        overall.append(float(v))
        sev = int(r["severity"])
        corr = r["corruption"]
        by_sev.setdefault(sev, []).append(float(v))
        by_corr.setdefault(corr, []).append(float(v))

    return {
        "metric_key": metric_key,
        "overall_mean": sum(overall) / len(overall) if overall else None,
        "by_severity_mean": {k: sum(v)/len(v) for k, v in sorted(by_sev.items())},
        "by_corruption_mean": {k: sum(v)/len(v) for k, v in sorted(by_corr.items())},}