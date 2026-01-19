from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import load_dataset
from PIL import Image
import numpy as np
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class HFTorchImageDataset(torch.utils.data.Dataset):
    """
    Wrapper Torch para un split de HuggingFace `datasets`.
    Espera columnas: image (PIL) y label (int).
    """
    def __init__(self, hf_split, transform=None, image_key="image", label_key="label"):
        self.ds = hf_split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

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
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        y = ex.get(self.label_key, -1)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)


def get_food101_hf_datasets(
    data_dir: str = "./data",
    hf_name: str = "food101",
    img_size: int = 96,
    seed: int = 7,
):
    """
    Food-101 desde HuggingFace, sin augmentations:
      Resize(96) + ToTensor + Normalize(ImageNet)
    Retorna: train_ds, val_ds, test_ds
    """
    if img_size <= 0:
        raise ValueError(f"img_size must be > 0. Got {img_size}.")

    cache_dir = str(Path(data_dir) / "hf_cache")
    ds = load_dataset(hf_name, cache_dir=cache_dir)

    # Food-101 en HF típicamente: train / validation
    train_split = ds.get("train", None)
    val_split   = ds.get("validation", None)
    test_split  = ds.get("test", None)  # por si existe

    if train_split is None:
        raise RuntimeError(f"No train split found for dataset: {hf_name}. Available: {list(ds.keys())}")
    if val_split is None and test_split is None:
        raise RuntimeError(f"No validation/test split found for dataset: {hf_name}. Available: {list(ds.keys())}")

    ops = [
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),]
    
    transform = transforms.Compose(ops)

    train_ds = HFTorchImageDataset(train_split, transform=transform)
    val_ds   = HFTorchImageDataset(val_split,   transform=transform) if val_split is not None else None
    test_ds  = HFTorchImageDataset(test_split,  transform=transform) if test_split is not None else None

    return train_ds, val_ds, test_ds

def get_food101_hf_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    hf_name: str = "food101",
    num_workers: int = 2,
    pin_memory: bool = True,
    img_size: int = 96,
    drop_last: bool = True,
    seed: int = 7) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    DataLoaders reproducibles (orden de batches + seeds de workers).
    """
    train_ds, val_ds, test_ds = get_food101_hf_datasets(
        data_dir=data_dir,
        hf_name=hf_name,
        img_size=img_size,
        seed=seed,)

    g_loader = torch.Generator().manual_seed(seed)

    common = dict(
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
        **common,)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common)
        

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **common,)

    return train_loader, val_loader, test_loader