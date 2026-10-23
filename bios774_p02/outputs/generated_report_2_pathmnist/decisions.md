1. **Goal** *(objective skill)* — Task: synthetic data release. Success: TSTR accuracy >= 0.8x train-on-real, C2ST clearly below 1, frac_nearest_real_is_train near 0.5, no exact copies, all 9 classes label-consistent. Reference: val (same source); test (other collection) for classifier tests. Conditional on y.
   - *Evidence:* profile.json: 9 classes, imbalance ratio 1.64, n_train=89996, 28x28x3
   - *Alternatives rejected:* Decline: not applicable, no non-generative way to share data
2. **Data prep** *(preprocessing skill)* — --prep scale_pm1, conv backbone, no flattening; stratified subsample via --max-train decided after probe
   - *Evidence:* profile.json: values 0-255, neighbour correlation 0.904
   - *Alternatives rejected:* mlp/flatten: loses spatial structure
3. **Model** *(model-selection skill)* — conditional DDPM, conv UNet, ch 32 pending probe
   - *Evidence:* n_train=89996 >> 5k; smallest class 7886
   - *Alternatives rejected:* cVAE: blurry; WGAN-GP: fragile; flow: weak samples
4. **Settings** *(diffusion skill)* — ddpm conv ch48, T1000 cosine, p_uncond 0.1, ema 0.999, bs128, lr2e-4, full train set (no subsample), --max-minutes 34 (probe 1 min counted; total 35 <= 40)
   - *Evidence:* probe/train_log.json: 27.4s per 157 steps (~5.7 steps/s) => ~11.6k steps in 34 min
   - *Alternatives rejected:* ch32: lower quality; ch64: too few steps
5. **Training check** *(diffusion skill)* — Stopped by time limit at 34 min, 18 epochs (~11.2k steps). Loss 0.118->0.054 plateau; eps_mse falls from 0.22 (t5) to 0.0006 (t95); no noise/saturation in probe grid; weaker on normal mucosa glands.
   - *Evidence:* ddpm_ch48/train_log.json, loss.png, sample_grid_probe.png
   - *Alternatives rejected:* longer training: budget spent
6. **Generate** *(diffusion skill)* — 9000 samples (1000/class), DDIM 50 steps, guidance 1.5, eta 0, tag rel; probe of 20/class took 8.8s
   - *Evidence:* ddpm_ch48/sample_config_probe.json
   - *Alternatives rejected:* guidance 1.0: less class-typical; higher guidance: lower diversity
7. **Evaluate** *(evaluation skill)* — fidelity,memorization,utility(tstr) on samples_rel. Memorization clean (0.499 vs 0.5, 0 exact copies); TSTR/TRTR acc 0.977 but macro-F1 gap -0.042..-0.024 (significant); C2ST 0.84 vs 0.49 ref: clearly distinguishable; label consistency 0.75 vs 0.96 real, weakest normal mucosa 0.59 and stroma 0.55; precision 0.65, recall 0.85.
   - *Evidence:* ddpm_ch48/metrics_rel.json
   - *Alternatives rejected:* -
8. **Adjust** *(evaluation skill)* — Resample 1000/class with guidance 2.5 (tag g25), same 50 DDIM steps, re-evaluate; no retraining
   - *Evidence:* metrics_rel.json label_consistency 0.752, weak classes mucosa 0.588, stroma 0.547
   - *Alternatives rejected:* retrain: budget spent (35 of 40 min)
9. **Evaluate** *(evaluation skill)* — g25: label consistency 0.861 (vs 0.752), but recall 0.587 (vs 0.846), C2ST 0.911 (vs 0.845), Frechet 19.4 (vs 14.8), TSTR macro-F1 0.706 (vs 0.746; real 0.779), TSTR/TRTR acc 0.936 (vs 0.977). Memorization still clean (0.521, 0 copies). Keep guidance 1.5 as release set.
   - *Evidence:* ddpm_ch48/metrics_g25.json vs metrics_rel.json
   - *Alternatives rejected:* no further iteration: budget spent, remaining options unlikely to change verdict
10. **Recommend** *(recommendation skill)* — Use with restrictions: share ddpm_ch48/samples_rel.npz (9000 images, 1000/class, 28x28 RGB uint8, labeled synthetic) for classifier prototyping/method development; not for final evaluation, clinical claims, or replacing real test sets; no formal privacy guarantee. Expect ~25% label noise (esp. mucosa, stroma); smooth muscle and debris weak when transferred to the other-source test set.
   - *Evidence:* metrics_rel.json: TSTR/TRTR acc 0.977, macro-F1 0.746 vs 0.779; memorization 0.499, 0 copies; C2ST 0.845; label consistency 0.752
   - *Alternatives rejected:* g25 set: better labels but worse utility; do not release: no memorization or utility failure
