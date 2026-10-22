1. **Goal** *(objective skill)* — Primary task: class balancing. Success: recommend synthetic only if macro-F1 of real+synthetic beats best of real_only/class-weighted/oversample with paired bootstrap CI excluding 0; else recommend the simple baseline. Run one cheap conditional DDPM to give generation a fair test.
   - *Evidence:* profile.json: imbalance_ratio_max_over_min=1.64 (<3), normalized_entropy=0.9949, smallest class normal colon mucosa n_train=7886
   - *Alternatives rejected:* decline outright without any evidence: rejected, cheap test can show baselines
2. **Data prep** *(preprocessing skill)* — --prep scale_pm1, no subsampling of rare classes
   - *Evidence:* profile.json: uint8 0-255, neighbour_pixel_correlation=0.9032
   - *Alternatives rejected:* scale01: used for flows only
3. **Model** *(model-selection skill)* — conditional DDPM, conv backbone, ch=32
   - *Evidence:* n_train=89996, smallest class 7886 (>5k)
   - *Alternatives rejected:* cVAE: blurry; WGAN: fragile
4. **Settings** *(diffusion skill)* — ddpm conv ch=32, T=1000 cosine, p_uncond 0.1, ema 0.999, bs128, lr2e-4, max-minutes 12, full train split (smoke test passed)
   - *Evidence:* smoke run ok (ddpm_probe/train_log.json)
   - *Alternatives rejected:* ch=64: ~4x slower, too few steps
5. **Training check** *(diffusion skill)* — ddpm_conv_ch32: stopped by 12 min limit at epoch 12 (~8.4k steps, below the 10k target); eps_mse falls from 0.24 (t5%) to 0.001 (t95%) as expected
   - *Evidence:* train_log.json final_diagnostics; loss.png
   - *Alternatives rejected:* retrain longer: budget exhausted (12 of 15 min used)
6. **Generate** *(diffusion skill)* — class-counts: normal colon mucosa 4000, mucus 4000, adipose 4000; DDIM 50 steps, guidance 2.0, tag balance
   - *Evidence:* probe: 100 samples in 3.3s (sample_config_probe.json); train counts 7886/8006/9366
   - *Alternatives rejected:* full balance to 12885 (~26k images): too slow
7. **Evaluate** *(evaluation skill)* — fidelity, memorization, utility augment on samples_balance. Synthetic images are visibly unrealistic (grid), recognised as intended class only 55% (real 60%), C2ST 0.957 vs 0.900 ref, precision 0.33 vs 0.73. Adding them collapses macro-F1 to 0.211 vs real_only 0.619, class-weighted 0.472, oversample 0.584; all CIs of difference below 0. No copying (frac_nearest_train 0.468, 0 exact copies).
   - *Evidence:* metrics_balance.json
   - *Alternatives rejected:* skip fidelity: rejected, skill says always run
8. **Recommend** *(recommendation skill)* — Do not use (use real data as is; real_only was best baseline). No iteration run: 12 of 15 budget minutes used, and imbalance 1.64 means little to fix; a longer-trained model would have to overcome a -0.40 macro-F1 gap.
   - *Evidence:* metrics_balance.json macro_f1_diff_95ci synthetic_minus_real_only [-0.4189,-0.3969]; profile.json imbalance 1.64
   - *Alternatives rejected:* retrain/iterate: unaffordable and unlikely to change verdict
