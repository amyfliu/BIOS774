---
name: gan
description: Configure, train, check and sample a (conditional) Wasserstein GAN with gradient penalty (tools/train.py --model wgan).
---

# Model (tools/models/gan.py)
WGAN-GP (L15): the critic estimates a Wasserstein distance, avoiding the
uninformative Jensen–Shannon gradient of the original GAN when real and
generated distributions barely overlap. Conditional version: one-hot label
appended to z; projection term in the critic. No per-sample likelihood.

# When it is (not) a good choice
Use only with plenty of data (≥ ~10k samples) and a reason to prefer it
over DDPM (e.g. fast sampling). Avoid on small n: the critic memorizes the
training set and the generator follows.

# Hyperparameters — starting points
`--lr 1e-4` (Adam betas (0, 0.9) are built in), `--batch-size 64–128`,
`--params '{"latent_dim": 64, "n_critic": 5, "gp_lambda": 10, "ch": 32}'`
(conv) or `"hidden": [256, 256]` (mlp). Each training step is one critic
update; the generator updates every `n_critic` steps, so GANs need many
epochs — budget accordingly (measure with a 1-epoch probe).

# Health checks (train_log.json)
- `final_diagnostics.mode_collapse_suspected` / `diversity_ratio`
  (pairwise distance of fakes ÷ reals): < 0.5 means collapse — samples look
  alike. Look at the sample grid too.
- `wasserstein_est` per epoch should trend down and stabilize; erratic
  large swings → lower lr or raise `n_critic`.
- `gen_loss` absolute values are not meaningful on their own.
- Non-finite losses → train.py stops with "diverged": lower lr.

# Sampling
`--n-per-class` (conditional) or `--n`. Sampling is a single forward pass
(cheap).
