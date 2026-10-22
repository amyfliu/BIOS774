1. **Goal** *(objective skill)* — Task = class balancing. Success: recommend synthetic only if real+synthetic macro-F1 beats best of real_only/class-weighted/oversample with paired bootstrap CI excluding 0 and diff > 2 seed SDs; else recommend baseline. Real reference: val for fidelity, test for classifier. Conditioning on y needed. Run one small conditional DDPM to test this empirically since the augment evaluation supplies the baselines.
   - *Evidence:* profile.json: imbalance_ratio_max_over_min=1.64 (<3), smallest class normal colon mucosa n_train=7886, normalized_entropy=0.9949
   - *Alternatives rejected:* Decline outright without evidence: rejected, baselines come with augment evaluation at modest cost
2. **Data prep** *(preprocessing skill)* — --prep scale_pm1, conv backbone, no flattening
   - *Evidence:* profile.json: value_range 0-255, neighbour_pixel_correlation=0.904, no missing values
   - *Alternatives rejected:* mlp/flatten: discards spatial structure
3. **Model** *(model-selection skill)* — conditional DDPM, conv UNet, ch=32; no second family
   - *Evidence:* n_train=89996 >> 5k; image data; need class control (model-selection rule 3)
   - *Alternatives rejected:* cVAE: blurry samples; WGAN-GP: fragile, no specific reason
4. **Settings** *(diffusion skill)* — ch=32, T=1000 cosine, p_uncond=0.1, ema 0.999, lr 2e-4, batch 64, max-train 40000 (stratified), max-minutes 12 (probe used 1; total 13 of 15)
   - *Evidence:* probe train_log: 156 steps in 14.3s on mps; smoke test passed
   - *Alternatives rejected:* ch=64: ~4x slower, too few steps; full 90k train: no benefit within the step budget
5. **Training check** *(diffusion skill)* — Health check passed; stopped by 12-min limit (25 epochs, ~15.6k steps at batch 64)
   - *Evidence:* train_log.json: eps_mse t5=0.229, t25=0.062, t50=0.020, t75=0.0056, t95=0.0007; loss.png plateau ~0.056 without divergence
   - *Alternatives rejected:* none needed
6. **Generate** *(diffusion skill)* — class-counts (20k-subset scale) 782/750/561/552/1084/156/1111/774 for classes 0-7, none for class 8; DDIM 50 steps, guidance 1.5, tag balance; 5770 samples
   - *Evidence:* sample_config_balance.json: 164.2s, non-finite 0; targets scaled from profile.json class_counts_train to the 20,000-image real subset used by the augment test
   - *Alternatives rejected:* guidance 1.0: less class-typical; more samples: no budget
7. **Evaluate** *(evaluation skill)* — Ran fidelity, memorization and augment utility on the balance samples. Criterion met nominally: real+synthetic macro-F1 0.8271 vs best baseline (real_only) 0.8080; diff CI [0.0145,0.0232]; seed SD 0.0007-0.018. But fidelity weak: C2ST 0.913 (ref 0.5015), precision 0.603 / recall 0.802 (ref 0.964/0.962), label consistency 0.5265 (ref 0.9636), mucosa 0.2313. No copying: nearest-train fraction 0.501 (expected 0.5), 0 exact copies. Caveat: no size-matched control, so the gain cannot be attributed to balancing; the real subset was already near balanced (1.64x).
   - *Evidence:* ddpm_conv_ch32/metrics_balance.json
   - *Alternatives rejected:* No further iteration: budget nearly spent (13/15 min); lower guidance resample would not remove the missing size-matched control
8. **Recommend** *(recommendation skill)* — Use with restrictions: samples_balance.npz may supplement training of the tissue classifier (flag as synthetic; keep val/test real); validate on real held-out data from a different source; do not use as a stand-in for real images. Class weights/oversampling are not needed for this mild imbalance and were not better than plain real data.
   - *Evidence:* metrics_balance.json: macro-F1 0.8271 vs 0.8080/0.7988/0.7983 (real_only/class-weighted/oversample); profile.json imbalance 1.64
   - *Alternatives rejected:* Do not use: gain excludes 0 and no copying; Use: fidelity too weak
