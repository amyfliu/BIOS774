---
name: model-selection
description: Choose the generative model family (VAE, WGAN-GP, normalizing flow, DDPM), whether to condition on labels, and the backbone — by comparing the families against this dataset and objective.
---

# When to apply
After the objective and preprocessing decisions. Then read the chosen
family's own skill (`vae`, `gan`, `flow`, `diffusion`) to configure it.

# The four families side by side (lectures L14–L17)

| | VAE (L14) | WGAN-GP (L15) | Normalizing flow (L16) | DDPM (L17) |
|---|---|---|---|---|
| Per-sample likelihood | lower bound (ELBO) | none | **exact** log p(x) | none here |
| Sample quality (images) | blurry (decoder mean) | sharp when it works | usually weak | **best** |
| Data needed | **small n OK** (hundreds) | large (≥ ~10k) | moderate; low D for tabular | large (≥ ~5–10k) |
| Training stability | **stable** | fragile (mode collapse, balance) | stable, but D×layers memory | stable, slow to converge |
| Compute (train / sample) | cheap / cheap | moderate / cheap | moderate / cheap | **expensive / expensive** (many denoising steps) |
| Latent representation | **yes** (encoder q(z\|x)) | no encoder | invertible z, same dim as x | no compact latent |
| Class conditioning | easy | easy | easy | easy (+ classifier-free guidance) |

# Decision rules (apply in order, log which one decided)

1. **Objective needs a likelihood / typicality score** (anomaly, QC, OOD,
   uncertainty) → **flow** (exact) as primary; VAE ELBO is a reasonable
   alternative/baseline. GAN and DDPM are out (no per-sample likelihood in
   these tools).
2. **Objective needs a latent representation** → **VAE**.
3. **Objective needs samples** (release, balancing, augmentation):
   - **Small n** (hundreds to a few thousand rows, or a target class with
     only tens of training samples) → **conditional VAE**. DDPMs and GANs
     overfit or fail to converge at this size.
   - **Images with enough data** (≥ ~5k training images) and quality
     matters → **conditional DDPM**. Consider a conditional VAE as the cheap
     baseline only if the comparison is informative for the report.
   - **Large tabular data** → cVAE first; WGAN-GP or tabular DDPM are
     alternatives when cVAE samples are too smooth (low recall) and there is
     budget.
   - Choose WGAN-GP over DDPM only for a specific reason (e.g. sampling
     speed is a hard requirement). Its failure modes need extra checks.
4. **Budget check.** Estimate cost before committing: run a 1-epoch probe
   (`--epochs 1 --max-minutes 1`) and read `seconds` in `train_log.json`.
   If the chosen family cannot reach a useful number of updates inside the
   budget (DDPM: ≥ ~10k optimizer steps on 28×28 images), shrink it
   (subsample, smaller `ch`) or fall back to the cheaper family — and log it.

# Conditioning
- Use `--conditional` whenever the objective needs control over which
  class is generated (balancing, labeled release). Unconditional is fine
  for anomaly scoring of "normal" data (train only on the normal classes
  with `--include-classes` / `--exclude-classes`).
- With a very rare class (≈10 training samples), a conditional model may
  not learn the label well. Plan to check label consistency after
  sampling (evaluation skill), and see the VAE skill for posterior-based
  sampling.

# Backbone
- Images → `conv` (spatial structure; neighbour-pixel correlation in
  `profile.json` is typically ≥ 0.8).
- Tabular → `mlp`.

# How many models?
One primary model, plus at most one alternative when the comparison
answers a question the report needs (e.g. "is the expensive DDPM worth it
over a cVAE?"). Running every family is NOT the goal — the instructions
reward justified choices, not exhaustive benchmarks.
