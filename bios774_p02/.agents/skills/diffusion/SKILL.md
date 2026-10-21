---
name: diffusion
description: Configure, train, check and sample a (conditional) DDPM (tools/train.py --model ddpm), including compute budgeting and DDIM / classifier-free-guidance sampling.
---

# Model (tools/models/ddpm.py)
Forward process adds Gaussian noise over T steps; a network predicts the
noise ε (L17, Ho et al. 2020). Conv backbone = small UNet (28→14→7);
mlp backbone = residual MLP for tabular data. Optional class conditioning
with classifier-free guidance (labels dropped with prob `p_uncond` during
training). Sampling uses DDIM with fewer steps than T and the EMA
(exponential moving average) weights.

# When it is (not) a good choice
Best image quality of the four families, but data- and compute-hungry.
Use for images with ≥ ~5k training samples. Avoid when n is small (it
memorizes or underfits) or when a likelihood is required.

# Budgeting (most important decision for DDPM)
Quality depends mostly on the number of optimizer steps.
1. Probe: `--epochs 1 --max-minutes 1` → read `epochs[0].seconds` in
   `train_log.json` → steps per second = (n_train / batch) / seconds.
2. Aim for ≥ ~10k steps (more is better); choose `--max-train`,
   `--batch-size`, `ch`, and `--max-minutes` to get there within budget.
3. Set `--epochs` high and let `--max-minutes` stop training.
4. Reserve time for sampling: cost ≈ n_samples × steps × (one forward
   pass), doubled when guidance ≠ 1. Probe with a small `--n-per-class` first
   and read `sample_config.json → seconds`.

# Hyperparameters — starting points (28×28 images)
`--prep scale_pm1 --backbone conv --conditional --lr 2e-4 --batch-size 64-128`
`--params '{"ch": 32-64, "T": 1000, "schedule": "cosine", "p_uncond": 0.1, "ema_decay": 0.999}'`
`ch: 32` is ~4× cheaper than 64; prefer 32 on a laptop unless the probe
shows plenty of headroom.

# Health checks
- Loss falls fast then plateaus slowly; a plateau is normal.
- `final_diagnostics.eps_mse_t{5..95}pct`: expected highest at small t and
  falling toward ~0 at large t. Values near 1.0 everywhere = not trained.
- LOOK at the sample grid: noise-like or colour-saturated samples mean too
  few steps of training (or guidance too high).

# Sampling knobs (tools/sample.py)
- `--steps` (DDIM): 50–100 is a good default; more steps = slower,
  slightly better.
- `--guidance` w: 1.0 = plain conditional; 1.5–3 makes samples more
  class-typical (higher precision / label consistency) but less diverse
  (lower recall). If label consistency is low, raise w; if recall is low
  and labels are fine, lower it. Log the choice.
- `--eta 0` (deterministic DDIM) is fine.
