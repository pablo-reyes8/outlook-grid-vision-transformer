# Outlook-Grid Vision Transformer

A research-grade hybrid vision architecture that fuses **VOLO-style Outlooker** local attention with **MaxViT-style grid attention** and modern convolutional inductive bias (MBConv). The core idea is to inject *Outlooker bias in every stage* (Model B), not only at the front, while retaining grid-based global mixing and efficient convolutional processing.

This repo provides two model variants, a reproducible training CLI, and tests that validate both block wiring and the end-to-end training loop.

## Highlights

- **Outlooker + MBConv + Grid Attention** inside each stage (Model B)
- **Two architectures**: front-only Outlooker (Model A) and per-block Outlooker (Model B)
- **CLI training** with YAML configs and overrides
- **PyTorch** implementation with mixup/cutmix, warmup+cosine LR, early stopping
- **Tests** for block correctness, model forward, and training smoke

## Architecture Overview

### Core Block: OutGridBlock (Model B)

The main block fuses three inductive biases:
1. **Outlooker**: dynamic local aggregation (VOLO-style)
2. **MBConv**: local convolutional mixing with SE
3. **Grid Attention**: global-ish mixing via grid partition (MaxViT-style)

In pseudo-form:

- Input/Output: `x in R^{B x C x H x W}`
- Uses NCHW for outlooker/mbconv, then BHWC for grid attention and MLP

Forward equation (block-level):

$$
\begin{aligned}
 x_1 &= \mathrm{Outlooker}(x) \\ 
 x_2 &= \mathrm{MBConv}(x_1) \\ 
 \hat{x}_2 &= \mathrm{permute}(x_2) \\ 
 y &= \hat{x}_2 + \mathrm{DP}(\mathrm{GridAttn}(\mathrm{LN}(\hat{x}_2))) \\ 
 z &= y + \mathrm{DP}(\mathrm{MLP}(\mathrm{LN}(y))) \\ 
 \mathrm{OutGridBlock}(x) &= \mathrm{permute}^{-1}(z)
\end{aligned}
$$

Where `DP` is stochastic depth and `LN` is layer norm in BHWC.

### Model A: OutlookerFrontGridNet

**Flow**:

`Stem -> OutlookerFront (L blocks) -> GridOnlyBlock stacks (per stage) -> Head`

- Outlooker is only at the front (VOLO-like). 
- Later stages are GridOnlyBlock: MBConv + GridAttn + MLP.
- Good as a baseline to isolate the effect of front-only outlook bias.

### Model B: MaxOutNet (Main Model)

**Flow**:

`Stem -> [OutGridBlock x depth] -> Downsample -> ... -> Head`

- Each stage uses **OutGridBlock**, injecting the Outlooker inductive bias throughout the network.
- This is the best-performing variant in this repo.

## Research Motivation

- **VOLO** uses Outlooker as a front-end local aggregator.
- **MaxViT** uses block-wise attention with structured window/grid partitions.

This repo explores a hybrid where *Outlooker bias is present at every stage*, combined with MBConv and grid attention. The intuition is that local dynamic aggregation can keep feature extraction rich and stable while grid attention provides global mixing.

## Configuration

Training is configured via YAML in `configs/train.yaml`.

Key sections:
- `model`: model type, stages, downsampling
- `training`: optimizer, LR schedule, mixed precision, early stopping
- `data`: dataset and augmentation options
- `runtime`: device, output dir, seed

Minimal example:

```yaml
model:
  type: model_b
  num_classes: 100
  stages:
    - dim: 64
      depth: 2
      num_heads: 4
      grid_size: 4
      window_size: 8
      outlook_heads: 4
      outlook_kernel: 3
      outlook_mlp_ratio: 2.0
      mbconv_expand_ratio: 4.0
      mbconv_se_ratio: 0.25
      mbconv_act: silu
      use_bn: true
      attn_drop: 0.0
      proj_drop: 0.0
      ffn_drop: 0.0
      drop_path: 0.0
      mlp_ratio: 4.0
      mlp_act: gelu

training:
  epochs: 100
  lr: 0.0005
  weight_decay: 0.05
  use_amp: true

runtime:
  device: cuda
```

## Training CLI

Install requirements:

```bash
pip install -r requirements.txt
```

Run with the default config:

```bash
python scripts/train.py --config configs/train.yaml
```

Switch models or override settings:

```bash
python scripts/train.py --config configs/train.yaml --model b
python scripts/train.py --config configs/train.yaml --model a --epochs 50
python scripts/train.py --config configs/train.yaml --batch-size 256 --img-size 64
```

Quick CPU smoke run with synthetic data:

```bash
python scripts/train.py --config configs/train.yaml --device cpu --epochs 1 --batch-size 8 --val-split 0 --data-dir ./data --model b
```

## Docker

```bash
docker build -t outlook-grid-vit .
docker run --rm -it outlook-grid-vit
```

## Tests

```bash
pytest -q
```

Tests cover:
- Grid partition/unpartition correctness
- Outlooker and OutGridBlock wiring
- Model forward passes for Model A and B
- Training loop smoke test

## Project Structure

```
configs/          # YAML configs
scripts/          # CLI entrypoints
notebooks/        # Jupyter Showcase 
src/              # models, blocks, training
tests/            # pytest suite
```

## Notes on Block Compatibility

- Grid attention requires `H` and `W` divisible by `grid_size`.
- Outlooker kernel size must be odd and > 0.
- The implementation uses NCHW for conv/outlook and BHWC for grid attention.

## Future Research Ideas

- Ablations: remove Outlooker or MBConv to quantify contributions.
- Try larger grid sizes or add window attention for local-global mix.
- Benchmark on ImageNet or larger CIFAR variants.
- Explore token mixing alternatives (e.g., axial or windowed attention).

## Citation

If you use this code or ideas, cite VOLO and MaxViT.

---

If you want, I can add ablation configs, training curves logging, or a more formal experiment table next.
