1. **Goal** *(objective skill)* — Tasks: (1) integration/classification: probe balanced accuracy on label_fine (primary; 17 classes) and label (coarse, 8 classes), joint vs RNA-alone vs ATAC-alone vs pca_concat; (2) cross-modal prediction rna->atac and atac->rna via evaluate cross. Labels NOT used in training (label-free clip primary). Success: joint probe bal-acc on label_fine >= best single modality with paired-difference CI not below 0 (>= means 'at least as well'); 'helps' only if CI excludes 0 positive. Prediction: per-sample Pearson r (RNA target) / AUROC (ATAC target) CI above train-mean baseline. Labels come from CellTypist on RNA only so favour RNA; classes <10 test cells (pDC 7, Plasma 6, Mk 2 etc.) uninterpretable individually.
   - *Evidence:* profile.json: test n=1560, label 8 classes smallest train 11, label_fine 17 classes; DATA.md label source RNA
   - *Alternatives rejected:* supcon_clip rejected: labels not allowed for training in objective
2. **Data prep** *(preprocessing skill)* — rna: log1p_hvg2000_pca50 (encoder input); atac: tfidf_lsi50_drop1 (first LSI comp corr with depth 0.856 -> dropped, 49 dims). Targets: rna log1p_hvg2000 (Pearson), atac binary_top10000 (AUROC).
   - *Evidence:* profile.json: rna 94.2% zeros, overdispersion 1.23, 68% nonzeros=1; atac 91.8% zeros, 72% nonzeros<=2, 10% =1; features/*.json: rna PCA var 0.333, atac LSI var 0.067
   - *Alternatives rejected:* lsi100/pca100 deferred unless fine-label probe weak
3. **Combine modalities** *(fusion skill)* — Baselines: rna_only_pca, atac_only_pca (pca_concat single input), early_fusion_pca (pca_concat both). Primary: clip with mlp encoders dim 32, label-free. Alternative: vicreg mlp dim 32.
   - *Evidence:* n_train=7285 (>2000, so clip ok); many same-type cells -> false-negative risk motivates vicreg comparison
   - *Alternatives rejected:* supcon_clip rejected: labels not allowed; simclr: single modality only
4. **Settings** *(training skill)* — clip mlp d32 hidden256 layers2 dropout0.1 batch512 lr1e-3 wd1e-4, max 8 min; vicreg same with batch 256, max 5 min; 7 min kept for iteration. Total max <= 20. Smoked clip OK (R@1 above chance 1/1560).
   - *Evidence:* smoke run val R@1 0.006-0.009, loss decreasing
   - *Alternatives rejected:* bigger dim rejected: bigger not better at n~1e4
5. **Training check** *(training skill)* — clip_mlp_d32: best_epoch 11/31, early stop, val R@1 0.137 vs chance ~0.0006; train loss 1.8 vs val 3.7 -> overfitting. vicreg_mlp_d32: best epoch 59/79, val R@1 0.053. Both took <0.3 min. Iterate on clip with more regularisation.
   - *Evidence:* train_log.json / run_config.json summary of each run
   - *Alternatives rejected:* none
6. **Adjust** *(training skill)* — clip_mlp_d32_reg: hidden128 dropout0.3 wd1e-3, max 4 min (total budget used <= 8+5+4=17)
   - *Evidence:* clip_mlp_d32 best_epoch 11/31
   - *Alternatives rejected:* fewer layers
7. **Evaluate** *(evaluation skill)* — val label_fine probe bal-acc: clip_mlp_d32_reg 0.900, early_fusion_pca 0.880, vicreg 0.868, rna_only 0.849, atac_only 0.818, clip_mlp_d32 0.832 (overfit). Coarse label ~0.98 for all joint models vs RNA 0.976, ATAC 0.827. clip_reg FOSCTTM 0.024 (chance 0.5), R@5 ~0.27 vs 0.003. Cross: atac->rna per-cell r 0.466 vs baseline 0.30; rna->atac per-cell AUROC 0.672 vs mean baseline 0.661 (small). Chosen for test: clip_mlp_d32_reg; also test baselines rna_only, atac_only, early_fusion and vicreg (alternative).
   - *Evidence:* comparison_val.md
   - *Alternatives rejected:* early_fusion_pca is simpler; kept as strongest-baseline reference on test; k-means ARI best for pca baselines
8. **Downstream** *(recommendation skill)* — Classification: test label_fine probe bal-acc clip_reg 0.884 vs rna 0.831, atac 0.833, early fusion 0.849 (chance 0.059); coarse 0.983 vs 0.977/0.951/0.976. Weakest large class: Tcm/Naive helper T (n=573) recall 0.61 clip_reg, 0.60 rna, 0.56 atac. Prediction: atac->rna per-cell r 0.466 (clip_reg encoder) vs 0.457 (ATAC alone) vs 0.300 train-mean baseline; per-feature r 0.118, 13.5% genes r>0.3. rna->atac per-cell AUROC 0.671 vs 0.661 baseline, per-feature AUROC 0.597 (0.5 chance): weak. Retrieval FOSCTTM 0.024 (chance 0.5), R@5 0.28 vs 0.003.
   - *Evidence:* comparison_test.md, comparison_test_vs_*.md, clip_mlp_d32_reg/metrics.json, rna_only_pca/metrics.json, atac_only_pca/metrics.json
   - *Alternatives rejected:* none
9. **Recommend** *(recommendation skill)* — Verdict: Multimodal helps (vs single modalities). Use clip_mlp_d32_reg joint embedding for annotation when both modalities exist; early_fusion_pca is within CI (+0.035 [-0.000,+0.075]) and simpler. For prediction: ATAC->RNA clear gain over mean baseline (r 0.466 vs 0.300), RNA->ATAC negligible. Caveats: labels from RNA via CellTypist; one donor; small classes uninterpretable; k-means ARI not better than PCA baselines.
   - *Evidence:* label_fine test: clip_reg 0.884 vs RNA 0.831 diff +0.053 [+0.015,+0.087]; vs ATAC 0.833 +0.051 [+0.011,+0.096]; vs early fusion +0.035 [-0.000,+0.075]
   - *Alternatives rejected:* early_fusion_pca; vicreg (no gain: 0.827)
