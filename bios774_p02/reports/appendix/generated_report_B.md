# generated_report_B: PathMNIST — synthetic images to balance tissue classes

**Verdict:** Do not use synthetic images here: the tissue classes are already nearly balanced, and adding generated images made the classifier much worse (macro-F1 0.21 vs 0.62 with real data only).

## 1. Summary

- **Question:** Can synthetic images for under-represented tissue types improve a downstream tissue classifier?
- **Approach:** Trained a class-conditional diffusion model (12 min), generated 4,000 images each for the three smallest classes (12,000 total), and compared a classifier trained on real+synthetic against real data only, class weighting and random oversampling.
- **Result:** Real data only scored best (macro-F1 0.619); oversampling 0.584; class weighting 0.472; real+synthetic 0.211. Every difference favouring the baselines has a confidence interval excluding 0.
- **Recommendation:** Train on the real data as is. Use class weighting or oversampling only if a specific class is later shown to be neglected.

## 2. Data at a glance

- 97,176 colon-pathology patches, 28×28 RGB, 9 tissue classes; 89,996 for training, 7,180 held-out real test images.
- Imbalance is mild: the largest class (colorectal adenocarcinoma epithelium, 12,885 training images) is only 1.64× the smallest (normal colon mucosa, 7,886). Class balance score is 0.995 of 1.
- Even the "rare" classes have thousands of real examples, so there is little to fix by generation.

![Class counts and real examples per class](../../outputs/generated_report_B_pathmnist/profile.png){width=90%}

## 3. Decision path

| # | Step | Decision | Why |
|:---|:------|:------------------|:-------------------------|
| 1 | Goal | Class balancing; test whether it is needed | Classes are already nearly even, so synthetic images may add nothing |
| 2 | Data prep | Scale pixels to [-1, 1] | Standard input range for image diffusion with a convolutional network |
| 3 | Model | Conditional diffusion model, conv network | Enough images per class; diffusion gives sharpest samples |
| 4 | Settings | Small diffusion net, 12 minutes training | Fits the 15-minute budget; leaves time for sampling and testing |
| 5 | Training check | Training healthy but short | Loss fell steadily; time limit stopped it after 12 passes |
| 6 | Generate | Add 4,000 images to each of 3 smallest classes | Lifts the rarest classes toward the largest; keeps sampling time modest |
| 7 | Evaluate | Compare against simple baselines on a classifier | Synthetic data must beat class weights and oversampling to be worth using |
| 8 | Recommend | Do not use synthetic images | They hurt the classifier; classes are already nearly balanced |

The turning point was the first look at the data: with imbalance of only 1.64 the case for generation was weak from the start, so the model was trained as a fair test, and the evaluation then settled it.

## 4. What was generated

12,000 synthetic 28×28 colour images: 4,000 each of normal colon mucosa, mucus and adipose (fat), the three smallest classes. The figure shows 10 synthetic and 10 real examples per class. Synthetic adipose lacks the clear fat-cell outlines; synthetic mucus and mucosa are dark, blotchy textures lacking the gland and fibre structure of real tissue.

![Synthetic (left) vs real test images (right)](../../outputs/generated_report_B_pathmnist/ddpm_conv_ch32/sample_grid_balance.png){width=90%}

## 5. Results

Realism of the synthetic images (three generated classes only; real-vs-real values in brackets are the best achievable):

| Measure | Synthetic | Real vs real |
|:---|---:|---:|
| Realism: share of synthetic images that look like real ones (precision) | 0.33 | 0.73 |
| Coverage: share of real images resembled by synthetic ones (recall) | 0.76 | 0.85 |
| A classifier can tell synthetic from real (two-sample test; 0.5 = cannot) | 0.957 | 0.900 |
| Recognised as intended tissue by a real-trained classifier | 0.55 | 0.60 |

The images are easily told apart from real ones and often not recognised as the intended type (mucus 0.45, mucosa 0.49). There is no sign of copying: the share of synthetic images whose nearest real image is a training image is 0.47 (0.50 expected), with no exact copies.

Effect on a downstream classifier (trained on real training images, scored on the real test set; 9 classes):

| Training set | Macro-F1 |
|:---|---:|
| Real only | 0.619 |
| Real + class weights | 0.472 |
| Real + random oversampling | 0.584 |
| Real + synthetic (12,000) | 0.211 |

Real+synthetic minus real-only: 95% CI [−0.419, −0.397]; minus class weighting [−0.273, −0.249]; minus oversampling [−0.384, −0.362]. Each of the three target classes has 741–1,338 test images, so per-class results are not anecdotal. With synthetic data, recall for adipose fell from 0.60 to 0.00, and several non-target classes (lymphocytes, stroma) collapsed too, because the poorly-realistic images pulled decision boundaries.

## 6. Recommendation

**Do not use** these synthetic images. Use the real training set unchanged (real-only was the best option). Deciding numbers: macro-F1 0.211 with synthetic vs 0.619 real-only (CI of difference excludes 0), and class imbalance of just 1.64. Caveats: the diffusion model was small and stopped by the time limit after about 8,400 updates, so better synthetic images are possible; but closing a 0.40 macro-F1 gap, to merely match a baseline, is unlikely to be worth it when no imbalance problem exists. Class weighting also lowered macro-F1 in this setting, so it is not a free fix either. What would change the verdict: a much more severe imbalance (e.g. a class under ~500 images), or a much longer-trained, validated generator that beats the real-only baseline.

## 7. Limitations

- One generator, one short training run, one sampling setting (guidance 2.0), no iteration (the budget was nearly spent).
- Only three classes were topped up (to about 12,000 each, roughly the largest class), not all nine.
- The evaluation classifier is small and reaches only 0.62 macro-F1 even on real data, with weak performance on stroma (F1 0.11) and mucus; single training run, so differences between baselines are less certain than the bootstrap intervals suggest (they capture test-set noise only).
- No formal privacy guarantee; the copying check is evidence, not proof.

## Appendix A. Full decision log

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

## Appendix B. Training details

Conditional diffusion model with a small convolutional U-Net (448,163 parameters), pixels scaled to [−1, 1], 1,000 noise steps with cosine schedule, label dropout 0.1, exponential moving average 0.999, batch 128, learning rate 2e-4, full training set (89,996 images). Stopped by the 12-minute limit after 12 of 100 planned epochs (about 8,400 updates). Loss fell from 0.139 to 0.058 and flattened. Noise-prediction error fell from 0.24 (early steps) to 0.001 (late steps), as expected for a trained model. Sampling: 50 deterministic steps, guidance 2.0, 12,000 images in 354 s.

![Training loss](../../outputs/generated_report_B_pathmnist/ddpm_conv_ch32/loss.png){width=45%}

## Appendix C. Glossary

- **Macro-F1:** average over classes of a balanced precision/recall score; treats every class equally.
- **Class weighting:** counting errors on smaller classes more heavily during training.
- **Random oversampling:** repeating real images of smaller classes.
- **Diffusion model (DDPM):** generator that learns to remove noise step by step; images are made by denoising random noise.
- **Guidance:** a setting that makes generated images more typical of the requested class, at some cost to variety.
- **Precision / recall (fidelity):** realism / coverage of synthetic images relative to real ones.
- **Two-sample test (C2ST):** accuracy of a classifier separating real from synthetic; 0.5 means indistinguishable.
- **Label consistency:** how often a real-trained classifier agrees with the intended class of a synthetic image.
- **Confidence interval (bootstrap):** range of plausible values for a difference given test-set noise.

## Appendix D. Reproduce

Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pathmnist --out outputs/generated_report_B_pathmnist
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --model ddpm --backbone conv --prep scale_pm1 --conditional --epochs 100 --batch-size 128 --lr 2e-4 --params '{"ch":32,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-minutes 12
python3 tools/sample.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --class-counts '{"normal colon mucosa": 100}' --steps 50 --guidance 2.0 --tag probe
python3 tools/sample.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --class-counts '{"normal colon mucosa": 4000, "mucus": 4000, "adipose": 4000}' --steps 50 --guidance 2.0 --tag balance
python3 tools/evaluate.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --tasks fidelity,memorization,utility --utility augment --samples-tag balance
```
