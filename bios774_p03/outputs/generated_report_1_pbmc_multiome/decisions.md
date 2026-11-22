1. **Goal** *(objective skill)* — Primary: classification/integration on label_fine (17 classes) and label (8); joint vs RNA alone vs ATAC alone, probe balanced accuracy + macro-F1. Secondary: cross-modal prediction rna->atac (binary_top10000, AUROC) and atac->rna (log1p_hvg2000, Pearson r) vs train-mean baseline. No labels in training (label-free). Success: joint probe bal. acc. on label_fine >= best single modality (CI overlap or better = 'at least as well'); prediction clearly above train-mean baseline. Labels come from RNA (CellTypist) so joint can at best tie RNA; tiny classes (<10 test cells) not conclusive.
   - *Evidence:* profile.json: n_train=7285; label classes 8, label_fine 17; smallest train class 11
   - *Alternatives rejected:* supcon_clip: rejected, objective does not allow labels for training
2. **Data prep** *(preprocessing skill)* — rna: log1p_hvg2000_pca50 (encoder input); atac: tfidf_lsi50_drop1 (72% nonzeros<=2, 92% zeros, ATAC-like; first LSI comp corr with depth r=0.86 so dropped, 49 dims). Targets: rna log1p_hvg2000 (Pearson r), atac binary_top10000 (AUROC).
   - *Evidence:* profile.json atac frac_le_2=0.72; features/atac__tfidf_lsi50_drop1.json corr=0.856; rna PCA50 explained var 0.333, atac 0.067
   - *Alternatives rejected:* lsi100/pca100: only if fine-label probe weak
3. **Combine modalities** *(fusion skill)* — Baselines: rna_only_pca, atac_only_pca, early_fusion_pca (dim 32). Primary: clip with mlp encoders dim 32, label-free. One alternative: vicreg (many same-type look-alike cells are false negatives for contrastive loss).
   - *Evidence:* n_train=7285 paired; objective includes cross-modal prediction
   - *Alternatives rejected:* supcon_clip: labels not allowed; simclr: single modality
4. **Settings** *(training skill)* — clip: mlp hidden256 x2, dim 32, batch 512, lr 1e-3, dropout 0.1, epochs 300 patience 20, max 8 min. vicreg: batch 256, max 5 min. Total planned 13 of 20 min, 7 kept for iteration.
   - *Evidence:* smoke runs OK (train_log.json); n_train=7285
   - *Alternatives rejected:* larger dim: not needed for 8 classes
5. **Training check** *(training skill)* — clip: best_epoch 11 of 31, train loss 1.82 vs val 3.68 at epoch 30, val R@1 0.137 (chance ~0.001); effective rank 12.8/32 on val. vicreg: best_epoch 59 of 79, val R@1 0.053, eff rank 8.4/32. No collapse (sd>0). Action: rerun clip with stronger regularisation.
   - *Evidence:* clip_mlp_d32/train_log.json, run_config.json summary; metrics_val.json collapse
   - *Alternatives rejected:* accept as is: overfit symptom matches training-skill rule (best_epoch <=10% ... borderline)
6. **Evaluate** *(evaluation skill)* — val probe bal acc label: rna 0.976, atac 0.827, early fusion 0.980, clip 0.981, vicreg 0.984. label_fine: rna 0.849, atac 0.818, early fusion 0.880, clip joint 0.832, vicreg joint 0.868 (CIs ~±0.04, overlapping). Cross: atac->rna per-cell r clip 0.463, vicreg 0.475, atac-only 0.456 vs train-mean 0.30; rna->atac per-feature AUROC ~0.60 for all. Retrieval: clip R@5 0.27, FOSCTTM 0.025 (chance R@5 0.003); vicreg R@5 0.13. Joint CLIP is not above early fusion on label_fine -> iterate on CLIP regularisation.
   - *Evidence:* comparison_val.md, metrics_val.json
   - *Alternatives rejected:* -
7. **Adjust** *(training skill)* — clip_mlp_d32_reg: hidden 128, dropout 0.3, weight decay 1e-3. Trained 0.29 min (best epoch 127 of 147, no early overfit). Val label_fine joint probe 0.900 (was 0.832; early fusion 0.880; rna 0.849); label 0.979; retrieval R@5 0.276/0.265, FOSCTTM 0.024; atac->rna r 0.466. Total training minutes used ~0.7 of 20.
   - *Evidence:* comparison_val.md; clip_mlp_d32_reg/metrics_val.json, train_log.json
   - *Alternatives rejected:* vicreg regularised: not needed, vicreg retrieval 2x worse
8. **Encoder + loss** *(evaluation skill)* — Report on test: clip_mlp_d32_reg (primary, label-free), vicreg_mlp_d32 (alternative), baselines rna_only_pca, atac_only_pca, early_fusion_pca. Original clip_mlp_d32 dropped (superseded).
   - *Evidence:* comparison_val.md
   - *Alternatives rejected:* early_fusion_pca: close on label_fine (0.880) but gives no cross-modal map
9. **Downstream** *(recommendation skill)* — Test (n=1560). label_fine probe bal acc: clip joint 0.884 [0.858,0.908]; RNA enc 0.852 [0.815,0.880]; ATAC enc 0.838; rna_only 0.831; atac_only 0.833; early fusion 0.849; vicreg joint 0.827. label: clip joint 0.983, rna_only 0.977, atac_only 0.951. atac->rna per-cell r 0.466 [0.458,0.474] vs atac_only 0.457 [0.448,0.465] vs train-mean 0.300; per-gene r>0.3 for 13.5% genes. rna->atac per-sample AUROC 0.671 vs train-mean 0.661, per-peak AUROC 0.597 (0.5 chance); rna_only 0.599. Retrieval R@5 0.28/0.27 vs chance 0.003, FOSCTTM 0.024 (0.5 chance).
   - *Evidence:* comparison_test.md; */metrics.json
   - *Alternatives rejected:* -
10. **Recommend** *(recommendation skill)* — Verdict: Multimodal enables. Joint CLIP embedding is at least as good as either modality for cell types (fine 0.884 vs best single 0.852, CIs overlap, so no proven gain) and uniquely supports cross-modal matching (R@5 0.28 vs 0.003). Prediction of each modality from the other is only modestly above the train-mean baseline (ATAC->RNA r 0.466 vs 0.300; RNA->ATAC per-sample AUROC 0.671 vs 0.661). Caveats: labels predicted from RNA by CellTypist, one donor, tiny classes (2-7 test cells), early fusion PCA is within CI. Would change: RNA-independent labels, more donors.
   - *Evidence:* comparison_test.md, metrics.json files
   - *Alternatives rejected:* early_fusion_pca: simpler and within CI on label_fine
