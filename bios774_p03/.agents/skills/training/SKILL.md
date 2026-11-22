---
name: training
description: Size hyperparameters to the data and budget, smoke-test, train, and run the health checks (overfitting, collapse, chance-level matching, temperature) before trusting any embedding.
---

# When to apply
After the fusion decision (log `hyperparameters`), and after every training
run (log `training_check`).

# Defaults and how to adjust them

| Knob | Starting value | Adjust when |
|---|---|---|
| `--batch-size` | 512 for `clip`/`supcon_clip` with n_train ≥ 5,000; 256 below; ≤ n_train/4 | contrastive losses get more negatives from bigger batches; VICReg is fine at 256 |
| `--lr` | 1e-3 (`mlp`, `linear`); 3e-4 (`adapter`) | loss oscillates → halve; flat from epoch 1 → double once |
| `--hidden` / `--layers` | 256 / 2 (`mlp`); 512 / – (`adapter`) | strong overfitting (below) → 128 or `--layers 1`; underfitting on train → 512 |
| `--dropout` | 0.1 | overfitting → 0.2–0.3 |
| `--weight-decay` | 1e-4 | overfitting → 1e-3 |
| `--dim` | fusion skill | |
| `--temperature` | 0.07, learned | stuck → `--fixed-temperature` with 0.1 |
| `--epochs` / `--patience` | 300 / 20 (early stopping on val loss keeps the best epoch) | |
| `--max-minutes` | split the budget: primary ≈ 50%, alternative ≈ 25%, ≥ 25% kept for iterations | |

Precomputed inputs (PCA/LSI/pretrained features) make an epoch take
seconds at n ~ 10⁴, so budget is rarely the binding constraint — overfitting
is. Always run `--smoke` first for each new configuration.

# Health checks — read `train_log.json`, `loss.png`, `run_config.json → summary`

| Check | Where | Healthy | Problem → action |
|---|---|---|---|
| Overfitting | train vs val loss curves; `best_epoch` | val loss flattens near train | val loss rises while train falls and `best_epoch` ≲ 10% of epochs run → more dropout/weight decay, smaller `--hidden`, fewer layers, or `vicreg` |
| Matching above chance | `val_recall_at_1` vs `val_recall_at_1_chance`; `summary.val_recall_at_1_kept_model` | many times chance | ≈ chance → inputs not paired correctly, lr wrong, or no shared signal; check the inputs' own baselines before tuning |
| Collapse | `val_embedding_sd_mean`; later `collapse.effective_rank`, `dead_dims` | sd stays well above 0; effective rank ≳ a few | sd → 0 or effective rank ≈ 1 → collapse: for `vicreg` raise the variance weight (not exposed: switch to `clip`), for `clip` check the temperature |
| Temperature | `temperature` column | settles between ~0.01 and ~0.2 | runs to the clamp (0.01) early with rising val loss → `--fixed-temperature` |
| Stop reason | `summary.stop_reason` | early stop or epochs | "time budget" → the model may be undertrained: say so or give it more minutes in an iteration |
| Adapter | `adapter` starts at the input | val loss and R@1 should improve on epoch 1's | never improves → the pretrained space is already as good as this data can make it: report the zero-shot model |

Log a `training_check` decision per model with the numbers you looked at
(best epoch, val loss, val R@1 vs chance, embedding sd) and what you did.

# When to iterate (training-side)
Only for a concrete symptom from the table above, or when val evaluation
shows a specific weakness (evaluation skill). Change one or two knobs at a
time, name the new folder after the change (`clip_mlp_d32_drop03`), and log
an `iteration` decision with the before/after val numbers.
