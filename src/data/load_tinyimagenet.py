from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import RandAugment

from datasets import load_dataset
from PIL import Image

from src.data.load_tinyimagenet_C import *
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class HFTorchImageDataset(Dataset):
    """
    Wrapper Torch para un split de HuggingFace `datasets`.
    Espera columnas: image (PIL) y label (int).
    """
    def __init__(self, hf_split, transform=None, image_key="image", label_key="label"):
        self.ds = hf_split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

        # intenta inferir nombres de clases si es ClassLabel
        self.class_names = None
        try:
            feat = self.ds.features.get(label_key, None)
            if feat is not None and hasattr(feat, "names"):
                self.class_names = list(feat.names)
        except Exception:
            pass

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[idx]
        img = ex[self.image_key]
        if not isinstance(img, Image.Image):
            # por si viene como dict tipo {"bytes":..., "path":...}
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        y = ex.get(self.label_key, -1)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)


def _pick_split(ds_dict, *candidates):
    for k in candidates:
        if k in ds_dict:
            return ds_dict[k]
    return None


def get_tinyimagenet200_hf_datasets(
    data_dir: str = "./data",
    hf_name: str = "zh-plus/tiny-imagenet",
    val_split: float = 0.0,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 64):

    """
    Tiny ImageNet-200 desde HuggingFace.
    Default img_size=64 (nativo). Si subes, hace Resize.

    Retorna:
      train_dataset, val_dataset, test_dataset
    """
    if img_size < 64:
        raise ValueError(f"img_size must be >= 64 for Tiny ImageNet. Got {img_size}.")

    cache_dir = str(Path(data_dir) / "hf_cache")

    # descarga y cachea automáticamente
    ds = load_dataset(hf_name, cache_dir=cache_dir)

    train_split = _pick_split(ds, "train")
    if train_split is None:
        raise RuntimeError(f"No train split found for dataset: {hf_name}. Available: {list(ds.keys())}")

    # muchos repos usan "validation" o "valid"
    official_val_split = _pick_split(ds, "validation", "valid", "val")
    test_split = _pick_split(ds, "test")

    crop_padding = max(8, img_size // 8)

    train_ops = []
    if img_size != 64:
        train_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    train_ops += [
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=random_erasing_p,
            scale=erasing_scale,
            ratio=erasing_ratio,
            value="random",),]

    train_transform = transforms.Compose(train_ops)

    test_ops = []
    if img_size != 64:
        test_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),]

    test_transform = transforms.Compose(test_ops)

    train_full = HFTorchImageDataset(train_split, transform=train_transform)

    # Caso A: haces split interno del train
    if val_split > 0.0:
        n_total = len(train_full)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            train_full,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(7))

        if official_val_split is not None:
            test_ds = HFTorchImageDataset(official_val_split, transform=test_transform)
        else:
            test_ds = HFTorchImageDataset(test_split, transform=test_transform) if test_split is not None else None

        return train_ds, val_ds, test_ds

    val_ds = HFTorchImageDataset(official_val_split, transform=test_transform) if official_val_split is not None else None
    test_ds = HFTorchImageDataset(test_split, transform=test_transform) if test_split is not None else None
    return train_full, val_ds, test_ds


def get_tinyimagenet200_hf_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    hf_name: str = "zh-plus/tiny-imagenet",
    num_workers: int = 2,
    val_split: float = 0.0,
    pin_memory: bool = True,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    img_size: int = 64,
    drop_last: bool = True):

    train_ds, val_ds, test_ds = get_tinyimagenet200_hf_datasets(
        data_dir=data_dir,
        hf_name=hf_name,
        val_split=val_split,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        img_size=img_size,)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0))

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0))

    return train_loader, val_loader, test_loader


def _unwrap_dataset(ds: Dataset) -> Dataset:
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_clean_test_loader_intersection_182(
    test_loader_clean: DataLoader,
    reference_train_loader: DataLoader,   # para obtener class_names (wnids)
    data_dir: str = "./data",
    corruption_name: str = "motion_blur",
    corruption_level: int = 3,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
):
    """
    Devuelve un loader del test clean filtrado a las clases que están en TinyImageNet-C (182 en tu caso).
    - test_loader_clean: tu loader clean (val/test)
    - reference_train_loader: tu train_loader clean (para obtener wnids ordenados)
    """

    base_train = _unwrap_dataset(reference_train_loader.dataset)
    train_wnids = list(base_train.class_names)  

    root = find_tinyimagenet_c_root(Path(data_dir))
    if root is None:
        root = download_and_extract_tiny_imagenet_c(data_dir=data_dir)

    split_dir = Path(root) / corruption_name / str(int(corruption_level))
    if not split_dir.exists():
        raise FileNotFoundError(f"No existe: {split_dir}")

    c_wnids = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    c_set = set(c_wnids)

    # Intersección de wnids
    keep_wnids = [w for w in train_wnids if w in c_set]
    keep_set = set(keep_wnids)

    print(f"[clean∩C] keep_classes={len(keep_wnids)} | drop_classes={len(train_wnids)-len(keep_wnids)}")

    labels_keep = {i for i, w in enumerate(train_wnids) if w in keep_set}

    base_test = _unwrap_dataset(test_loader_clean.dataset)

    hf_ds = getattr(base_test, "ds", None)
    if hf_ds is None:
        raise RuntimeError("No encuentro base_test.ds (HF dataset).")

    keep_indices = []
    for i in range(len(hf_ds)):
        y = int(hf_ds[i]["label"])   
        if y in labels_keep:
            keep_indices.append(i)

    print(f"[clean∩C] keep_samples={len(keep_indices)} / total={len(hf_ds)}")

    filtered_ds = Subset(base_test, keep_indices)

    filtered_loader = DataLoader(
        filtered_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=drop_last,
    )
    return filtered_loader
