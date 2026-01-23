from typing import Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import RandAugment

from datasets import load_dataset
from PIL import Image

import numpy as np
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def seed_worker(worker_id: int):
    """
    Hace reproducible el RNG dentro de cada worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    img_size: int = 64,
    seed: int = 7):
    """
    Tiny ImageNet-200 desde HuggingFace.
    Retorna: train_dataset, val_dataset, test_dataset
    """
    if img_size < 64:
        raise ValueError(f"img_size must be >= 64 for Tiny ImageNet. Got {img_size}.")

    cache_dir = str(Path(data_dir) / "hf_cache")
    ds = load_dataset(hf_name, cache_dir=cache_dir)

    train_split = _pick_split(ds, "train")
    if train_split is None:
        raise RuntimeError(f"No train split found for dataset: {hf_name}. Available: {list(ds.keys())}")

    official_val_split = _pick_split(ds, "validation", "valid", "val")
    test_split = _pick_split(ds, "test")

    crop_padding = max(8, img_size // 8)

    train_ops = []
    if img_size != 64:
        train_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))

    train_ops += [transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    train_transform = transforms.Compose(train_ops)

    test_ops = []
    if img_size != 64:
        test_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    test_transform = transforms.Compose(test_ops)

    train_full = HFTorchImageDataset(train_split, transform=train_transform)

    if val_split > 0.0:
        n_total = len(train_full)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        g_split = torch.Generator().manual_seed(seed)

        train_ds, val_ds = random_split(
            train_full,
            [n_train, n_val],
            generator=g_split,
        )

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
    drop_last: bool = True,
    seed: int = 7,):

    train_ds, val_ds, test_ds = get_tinyimagenet200_hf_datasets(
        data_dir=data_dir,
        hf_name=hf_name,
        val_split=val_split,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        img_size=img_size,
        seed=seed)

    g_loader = torch.Generator().manual_seed(seed + 1)

    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g_loader,)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **common_loader_kwargs,)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **common_loader_kwargs)

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            **common_loader_kwargs,)

    return train_loader, val_loader, test_loader


