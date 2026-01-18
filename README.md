# OutGridViT: Outlook-Grid Vision Transformer

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/outlook-grid-vision-transformer)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/outlook-grid-vision-transformer)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/outlook-grid-vision-transformer)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/outlook-grid-vision-transformer)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/outlook-grid-vision-transformer?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/outlook-grid-vision-transformer?style=social)


OutGridViT is a research-focused hybrid vision architecture that fuses **Outlooker** local attention (VOLO-style), **MBConv** inductive bias, and **Grid Attention** (MaxViT-style). The core design is Model A, where *every block* injects dynamic local aggregation before global mixing.

This repo contains the full training stack, ablations, baseline comparisons, and analysis tools (MAD metrics + attention visualizations).

## Table of Contents

- [Model A (Main Architecture)](#model-a-main-architecture)
- [Block Forward (tensor form)](#block-forward-tensor-form)
- [Model Forward (high level)](#model-forward-high-level)
- [Visual Experiments](#visual-experiments)
- [Reported Results](#reported-results)
- [MAD Metrics (Grid vs Outlooker)](#mad-metrics-grid-vs-outlooker)
- [Model Comparisons (CIFAR-100, 32x32)](#model-comparisons-cifar-100-32x32)
- [Baseline Training Recipe](#baseline-training-recipe)
- [Configs](#configs)
- [Training (Model A)](#training-model-a)
- [Baseline Comparisons (CIFAR-32)](#baseline-comparisons-cifar-32)
- [Attention Analysis and MAD Metrics](#attention-analysis-and-mad-metrics)
- [Project Structure](#project-structure)
- [Notes](#notes)
- [Citation](#citation)

## Model A (Main Architecture)

**Block composition (OutGridBlock):**
`Outlooker -> MBConv -> GridAttn -> MLP`

Input/output stays in 2D feature maps (NCHW). Only the GridAttn path temporarily uses BHWC and tokenized grids.

## Block Forward (tensor form)

Input: `x in R^{B x C x H x W}`

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

`DP` is stochastic depth. `LN` is LayerNorm in BHWC.

## Model Forward (high level)

$$
\begin{aligned}
 x_0 &= \mathrm{Stem}(x) \\ 
 x_{s,i+1} &= \mathrm{OutGridBlock}(x_{s,i}), \quad i=0..(d_s-1) \\ 
 x_{s+1,0} &= \mathrm{Downsample}_s(x_{s,d_s}) \\ 
 \mathrm{logits} &= \mathrm{Linear}(\mathrm{GAP}(\mathrm{BN}(x_{S,d_S})))
\end{aligned}
$$

## Visual Experiments

Outlooker behaves like a **dynamic 3x3 kernel** per position, while Grid Attention provides more global mixing per block. Below are sample overlays produced by the analysis pipeline.

Outlooker locality (kernel weights + center/spread overlays):

![Outlooker locality example](experiments_results/Visual%20Experiments/normal_example_outlooker/stage1_block_0.png)

More Outlooker examples (different stages/blocks):

| Example A | Example B |
| --- | --- |
| ![Outlooker locality stage0](experiments_results/Visual%20Experiments/easy_example_outlooker/stage0_block_0.png) | ![Outlooker locality stage1](experiments_results/Visual%20Experiments/easy_example_outlooker/stage1_block_2.png) |
| ![Outlooker locality stage2](experiments_results/Visual%20Experiments/normal_example_outlooker/stage2_block_1.png) | ![Outlooker locality stage3](experiments_results/Visual%20Experiments/normal_example_outlooker/stage3_block_1.png) |

Grid attention heatmap (query overlays over the input):

![Grid attention example](experiments_results/Visual%20Experiments/normal_example_grid/stage_1_block_0.png)

More Grid Attention examples (different stages/blocks):

| Example A | Example B |
| --- | --- |
| ![Grid attention stage0](experiments_results/Visual%20Experiments/easy_example_grid/stage_0_block_0.png) | ![Grid attention stage1](experiments_results/Visual%20Experiments/easy_example_grid/stage_1_block_2.png) |
| ![Grid attention stage2](experiments_results/Visual%20Experiments/normal_example_grid/stage_2_block_1.png) | ![Grid attention stage3](experiments_results/Visual%20Experiments/normal_example_grid/stage_3_block_1.png) |


## Reported Results

<div align="center">


| Dataset | Img size | Top-1 (val/test) | Params | Notes |
| :---: | :---: | :---: | :---: | :---: |
| CIFAR-100 | 32 | 74.7 / 78.4 | - | Model A, CIFAR-32 |
| CIFAR-100 | 64 | 78.7 / 81.2 | - | Upsampled CIFAR-100 |
| Tiny-ImageNet-200 | 64 | 66.5 / 69.8 | 22.5M | Competitive for 22M params |
| SVHN | 32 | 96.1 / - | - | Val reported in logs |

</div>

<br>


## MAD Metrics (Grid vs Outlooker)

<div align="center">


Quantitative summary (CIFAR-100, Model A). `GRID_abs` is L1 distance in feature-map pixels, with max = `(Hf-1)+(Wf-1)` per stage. `OUT_abs` is L1 distance inside a 3x3 kernel, max = 2.

| Stage | Hf x Wf | GRID_abs | OUT_abs | GRID max | OUT max |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 64 x 64 | 28.49 ± 1.17 | 1.19 ± 0.18 | 126 | 2 |
| 1 | 32 x 32 | 18.30 ± 0.24 | 1.61 ± 0.13 | 62 | 2 |
| 2 | 16 x 16 | 9.06 ± 0.21 | 1.62 ± 0.25 | 30 | 2 |
| 3 | 8 x 8 | 5.41 ± 0.55 | 1.69 ± 0.14 | 14 | 2 |

*Interpretation: Outlooker stays strictly local, while Grid Attention provides larger effective range per stage through global grid mixing.*

</div>

<br>


## Model Comparisons (CIFAR-100, 32x32)

<div align="center">


All baselines below were trained with the same recipe. Values are from `logs/Logs Models Comparisons.txt`.

| Model | Top-1 | Params |
| :--- | :---: | :---: |
| **OutGridViT (Model A)** | **79.8** | **14.1M** |
| ConvNeXt-Tiny | 72.60 | 27.89M |
| DeiT-Tiny (patch4) | 63.77 | 5.38M |
| DeiT-Small (patch4) | 59.00 | 21.38M |
| EfficientNetV2-S | 64.62 | 20.31M |
| MaxViT-Nano (surgery) | 75.41 | 17.38M |
| MaxViT-Tiny | 75.90 | 30.43M |
| ResNet18 (CIFAR stem) | 73.25 | 11.22M |
| ResNet50 (CIFAR stem) | 77.42 | 23.71M |
| Swin-Tiny (patch2) | 59.89 | 27.57M |

</div>

## Baseline Training Recipe

All baseline models use the same training config:

```python
history, model = train_model(
    model=model,
    train_loader=train_loader,
    epochs=100,
    val_loader=val_loader,
    device=device,

    lr=5e-4,
    weight_decay=0.05,

    autocast_dtype="fp16" if device == "cuda" else "fp32",
    use_amp=(device == "cuda"),
    grad_clip_norm=1.0,

    warmup_ratio=0.05,
    min_lr=1e-6,

    label_smoothing=0.1,

    print_every=400,
    mix_prob=0.5,
    mixup_alpha=0.8,
    cutmix_alpha=1.0,

    num_classes=100,
    channels_last=True)
```

## Configs

Prebuilt configs:
- `configs/cifar100_model_a.yaml`
- `configs/cifar100_64_model_a.yaml`
- `configs/svhn_model_a.yaml`
- `configs/tinyimagenet200_model_a.yaml`

`configs/train.yaml` is an alias of the CIFAR-100 Model A config.

## Training (Model A)

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/cifar100_model_a.yaml
```

## Baseline Comparisons (CIFAR-32)

Use the baseline runner to train timm models with a shared training recipe:

```bash
python scripts/train_cifar32_baselines.py --device cuda
```

Baselines included:
- DeiT Tiny (patch4)
- DeiT Small (patch4)
- Swin Tiny (patch2)
- MaxViT Tiny
- MaxViT Nano (surgery)
- ResNet18 (CIFAR stem)

## Attention Analysis and MAD Metrics

The analysis CLI generates:
- Outlooker locality visualizations
- Grid attention heatmaps
- MAD metrics for both attention paths

```bash
python scripts/run_attention_analysis.py \
  --config configs/cifar100_model_a.yaml \
  --checkpoint outputs/best_cifar100_model_a.pt \
  --split val \
  --output-dir analysis_outputs
```

Outputs:
- `analysis_outputs/outlooker/` (locality overlays + kernels)
- `analysis_outputs/grid/` (grid attention overlays)
- `analysis_outputs/mad_metrics.json` and `.csv`

Additional figures can be generated with the CLI and added to `experiments_results/` or a dedicated `figures/` folder.

## Project Structure

```
configs/          # YAML configs
scripts/          # CLI entrypoints
src/              # models, blocks, training, experiments
training_notebooks/  # ablations + baselines
tests/            # pytest suite
```

## Notes

- Grid attention requires `H` and `W` divisible by `grid_size`.
- Outlooker kernel size must be odd and > 0.
- NCHW for conv/outlook, BHWC for grid attention.

## Citation

If you use this code or ideas, cite VOLO and MaxViT.
