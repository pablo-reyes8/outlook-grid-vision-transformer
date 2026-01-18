# OutGridViT: Outlook-Grid Vision Transformer

OutGridViT is a research-focused hybrid vision architecture that fuses **Outlooker** local attention (VOLO-style), **MBConv** inductive bias, and **Grid Attention** (MaxViT-style). The core design is Model A, where *every block* injects dynamic local aggregation before global mixing.

This repo contains the full training stack, ablations, baseline comparisons, and analysis tools (MAD metrics + attention visualizations).

## Model A (Main Architecture)

**Block composition (OutGridBlock):**
`Outlooker -> MBConv -> GridAttn -> MLP`

Input/output stays in 2D feature maps (NCHW). Only the GridAttn path temporarily uses BHWC and tokenized grids.

### Block Forward (tensor form)

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

### Model Forward (high level)

$$
\begin{aligned}
 x_0 &= \mathrm{Stem}(x) \\ 
 x_{s,i+1} &= \mathrm{OutGridBlock}(x_{s,i}), \quad i=0..(d_s-1) \\ 
 x_{s+1,0} &= \mathrm{Downsample}_s(x_{s,d_s}) \\ 
 \mathrm{logits} &= \mathrm{Linear}(\mathrm{GAP}(\mathrm{BN}(x_{S,d_S})))
\end{aligned}
$$

## Reported Results (from notebooks)

| Dataset | Img size | Top-1 (val/test) | Params | Notes |
| --- | --- | --- | --- | --- |
| CIFAR-100 | 32 | 74.7 / 78.4 | - | Model A, CIFAR-32
| CIFAR-100 | 64 | 78.7 / 81.2 | - | Upsampled CIFAR-100
| Tiny-ImageNet-200 | 64 | 66.5 / 69.8 | 22.5M | Competitive for 22M params
| SVHN | 32 | 96.1 / - | - | Val reported in logs

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

If you want to include figures in this README, generate them with the CLI and move them into a `figures/` folder.

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
