1. **Goal** *(objective skill)* — Task=class balancing. Generate rare-class cells with a conditional model; recommend synthetic only if real+synthetic macro-F1 (and rare-class F1) beats best of real_only/class-weighted/oversample with paired bootstrap CI excluding 0; otherwise recommend the baseline. Reference: real test split.
   - *Evidence:* imbalance ratio 76.27; Megakaryocytes 11 train, Dendritic 28 train (profile.json)
   - *Alternatives rejected:* decline outright: rejected, imbalance is extreme so worth testing
2. **Data prep** *(preprocessing skill)* — --prep libnorm,log1p,hvg:2000,standardize
   - *Evidence:* row_sum max/min 15.9; skewness 13.9; samples_per_feature 0.159; log1p top2000 variance share 0.689 (profile.json)
   - *Alternatives rejected:* pca: loses gene interpretability; raw counts: model learns depth
3. **Model** *(model-selection skill)* — conditional VAE, mlp backbone, no second family
   - *Evidence:* n_train=1978; smallest class 11 train cells
   - *Alternatives rejected:* ddpm/wgan: need >=5-10k samples; flow: no likelihood needed
4. **Settings** *(vae skill)* — latent_dim 16, hidden [512,256], beta 1, warmup 10, lr 1e-3, bs 64, 150 epochs, max 4 min
   - *Evidence:* smoke test ran OK; budget 10 min
   - *Alternatives rejected:* bigger latent: unnecessary for 8 classes
5. **Training check** *(vae skill)* — No posterior collapse; accept model
   - *Evidence:* active_units 16/16, kl_total 46.8, 52s (train_log.json)
6. **Generate** *(vae skill)* — sample.py --class-counts Mega 200, DC 200, prior latent, tag prior
   - *Evidence:* 400 samples; label consistency 0.0725 vs real 0.953 (metrics_prior.json); macro-F1 diff CIs all include 0
7. **Adjust** *(vae skill)* — Iteration 1: resample with --latent-source posterior --temperature 1.5, tag post
   - *Evidence:* per-class label consistency DC 0.14, Mega 0.005 (metrics_prior.json); recall 0.31
   - *Alternatives rejected:* retrain with lower beta: costs budget, posterior sampling is the documented fix
8. **Evaluate** *(evaluation skill)* — Evaluated fidelity, memorization, augment utility on post samples
   - *Evidence:* label consistency 0.905 (ref 0.953); recall 0.0; nearest-real-is-train 0.9975 (exp 0.5); macro-F1 real+syn 0.9033 vs real_only 0.9079, CI [-0.03,0.021]; vs class-weighted [-0.006,0.013]; test n: DC 9, Mega 4 (metrics_post.json)
9. **Recommend** *(recommendation skill)* — Do not use (use class weighting or real_only). Synthetic cells (samples_post.npz) are smoothed variants of 11 and 28 real cells, no gain; stop iterating.
   - *Evidence:* all macro-F1 diff CIs include 0 (metrics_post.json); Mega F1 0.667 in every arm; model space log-normalized standardized 2000 HVG
   - *Alternatives rejected:* more iterations: unlikely to change verdict given 4 test Mega cells
