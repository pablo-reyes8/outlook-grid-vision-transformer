import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
import random
import numpy as np

def _seed_worker_factory(seed: int):
    """
    Crea un worker_init_fn que fija seeds de python/numpy/torch en cada worker
    de forma determinística a partir de `seed`.
    """
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return seed_worker


def get_cifar100_datasets(
    data_dir: str = "./data",
    val_split: float = 0.0,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    erasing_scale=(0.02, 0.20),
    erasing_ratio=(0.3, 3.3),
    img_size: int = 32,
    seed: int = 7):
    
    """
    CIFAR-100 datasets con augmentations "mix-friendly".
    Reproducibilidad: el split train/val queda fijo con `seed`.
    """
    if img_size < 32:
        raise ValueError(f"img_size must be >= 32 for CIFAR-100. Got {img_size}.")

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std  = (0.2675, 0.2565, 0.2761)

    crop_padding = max(4, img_size // 8)

    train_ops = []
    if img_size != 32:
        train_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))

    train_ops += [
        transforms.RandomCrop(img_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
        transforms.RandomErasing(
            p=random_erasing_p,
            scale=erasing_scale,
            ratio=erasing_ratio,
            value="random",
        ),]
    train_transform = transforms.Compose(train_ops)

    test_ops = []
    if img_size != 32:
        test_ops.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC))
    test_ops += [
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),]
    
    test_transform = transforms.Compose(test_ops)

    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform)
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform)

    if val_split > 0.0:
        n_total = len(full_train_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val

        split_gen = torch.Generator().manual_seed(seed)  # <-- fijo
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [n_train, n_val],
            generator=split_gen,)
        
    else:
        train_dataset = full_train_dataset
        val_dataset = None

    return train_dataset, val_dataset, test_dataset






def get_cifar100_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 2,
    val_split: float = 0.0,
    pin_memory: bool = True,
    ra_num_ops: int = 2,
    ra_magnitude: int = 7,
    random_erasing_p: float = 0.25,
    img_size: int = 32,
    seed: int = 7):

    """
    DataLoaders CIFAR-100 reproducibles:
    - mismo split train/val si val_split > 0
    - mismo orden de batches con shuffle=True
    - misma secuencia de RNG en cada worker (augmentations) entre corridas
    """
    train_ds, val_ds, test_ds = get_cifar100_datasets(
        data_dir=data_dir,
        val_split=val_split,
        ra_num_ops=ra_num_ops,
        ra_magnitude=ra_magnitude,
        random_erasing_p=random_erasing_p,
        img_size=img_size,
        seed=seed)

    dl_gen = torch.Generator().manual_seed(seed)
    worker_init_fn = _seed_worker_factory(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=dl_gen,               
        worker_init_fn=worker_init_fn)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            worker_init_fn=worker_init_fn,
            generator=dl_gen,)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn,
        generator=dl_gen)

    return train_loader, val_loader, test_loader

