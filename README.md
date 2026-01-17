# Outlook-Grid Vision Transformer (OutGridViT)

A research-grade hybrid vision architecture that fuses **VOLO-style Outlooker** local attention with **MaxViT-style grid attention** and MBConv inductive bias. The main idea is to inject dynamic local aggregation inside *every stage* (Model A) while retaining efficient global mixing through grid attention.

This repo includes:
- Two model variants (A and B)
- A training CLI driven by YAML configs
- Multiple dataset loaders (CIFAR-100, SVHN, Tiny-ImageNet-200)
- Tests for block wiring, model forward, and training smoke
- Model Comparisons

## Model Variants

**Model A (MaxOutNet, main)**
- Outlooker in every block (strong local bias throughout the network)
- Block: `Outlooker -> MBConv -> GridAttn -> MLP`
- File: `src/Model_A_OutGridNet.py`

**Model B (OutlookerFrontGridNet, baseline)**
- Outlooker only at the front, then Grid-only blocks
- Stage: `OutlookerFront -> GridOnlyBlock x depth`
- File: `src/Model_B_OutGridNet.py`

Naming note: **Model A** is the per-block Outlooker variant (MaxOutNet) and **Model B** is the front-only Outlooker baseline, matching the notebooks.

## Block Forward (OutGridBlock)

Input/Output: `x in R^{B x C x H x W}`

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

`DP` is stochastic depth. `LN` is layer norm in BHWC. The implementation uses NCHW for Outlooker/MBConv and BHWC for GridAttn/MLP.

### Model Forward (High Level)

Model A (MaxOutNet):

$$
\\begin{aligned}
 x_0 &= \\mathrm{Stem}(x) \\\\
 x_{s,i+1} &= \\mathrm{OutGridBlock}(x_{s,i}), \\quad i=0..(d_s-1) \\\\
 x_{s+1,0} &= \\mathrm{Downsample}_s(x_{s,d_s}) \\\\
 \\mathrm{logits} &= \\mathrm{Linear}(\\mathrm{GAP}(\\mathrm{BN}(x_{S,d_S})))
\\end{aligned}
$$

Model B (OutlookerFrontGridNet):

$$
\\begin{aligned}
 x_0 &= \\mathrm{Stem}(x) \\\\
 x'_0 &= \\mathrm{OutlookerFront}^L(x_0) \\\\
 x_{s,i+1} &= \\mathrm{GridOnlyBlock}(x_{s,i}) \\\\
 \\mathrm{logits} &= \\mathrm{Linear}(\\mathrm{GAP}(\\mathrm{BN}(x_{S,d_S})))
\\end{aligned}
$$

## Reported Results (from notebooks)

All numbers below are pulled from notebook logs (single runs). See `notebooks/` for exact settings, seeds, and loaders.

| Dataset | Model | Img size | Top-1 (val/test) | Params | Notes |
| --- | --- | --- | --- | --- | --- |
| CIFAR-100 | Model A (MaxOutNet) | 32 | 74.7 / 78.4 | 14.7M | best test from notebook
| CIFAR-100 | Model B (OutlookerFront) | 32 | 73.7 / 77.5 | 12.1M | front-only baseline
| CIFAR-100 | Model A (MaxOutNet) | 64 | 78.7 / 81.2 | 16.5M | upsampled CIFAR-100
| Tiny-ImageNet-200 | Model A (MaxOutNet) | 64 | 66.5 / 69.8 | 22.5M | HF Tiny-ImageNet
| SVHN | Model A (MaxOutNet) | 32 | 96.1 / - | 9.5M | val only in log

Highlights you can claim:
- **Tiny-ImageNet-200**: 66.5 val top-1 with **~22.5M params**
- **CIFAR-100**: ~81 top-1 (64x64 setup)

## Configs

Prebuilt configs (mirrored from notebooks):
- `configs/cifar100_model_a.yaml`
- `configs/cifar100_model_b.yaml`
- `configs/cifar100_64_model_a.yaml`
- `configs/svhn_model_a.yaml`
- `configs/tinyimagenet200_model_a.yaml`

`configs/train.yaml` is a default alias of `cifar100_model_a.yaml`.

## Training CLI

Install requirements:

```bash
pip install -r requirements.txt
```

Run a config:

```bash
python scripts/train.py --config configs/cifar100_model_a.yaml
python scripts/train.py --config configs/cifar100_model_b.yaml
python scripts/train.py --config configs/tinyimagenet200_model_a.yaml
```

Override common settings:

```bash
python scripts/train.py --config configs/cifar100_model_a.yaml --epochs 100
python scripts/train.py --config configs/cifar100_64_model_a.yaml --batch-size 128
```

## Datasets

Supported loaders:
- **CIFAR-100**: `data.dataset: cifar100`
- **SVHN**: `data.dataset: svhn`
- **Tiny-ImageNet-200 (HF)**: `data.dataset: tinyimagenet200`, `data.hf_name: zh-plus/tiny-imagenet`

Tiny-ImageNet uses the HuggingFace `datasets` package, cached under `data/hf_cache`.
Tiny-ImageNet-C helpers are in `src/data/load_tinyimagenet_C.py` and used in the notebooks.

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
configs/          # YAML configs for datasets and model variants
scripts/          # CLI entrypoints
notebooks/        # experiments + logs
src/              # models, blocks, training, data
tests/            # pytest suite
```

## Notes

- Grid attention requires `H` and `W` divisible by `grid_size`.
- Outlooker kernel size must be odd and > 0.
- NCHW for conv/outlook, BHWC for grid attention.

## Citation

If you use this code or ideas, cite VOLO and MaxViT.
