Split scored: **val** (n per model in metrics files)

| model | method | inputs | uses_labels | probe_bal_acc[label] | kmeans_ari[label] | chance[label] | atac->rna per-cell r | atac->rna per-cell r (baseline) | eff_rank(joint) | R@5 rna->atac | R@5 atac->rna | FOSCTTM | R@5 chance | rna->atac per-feature AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac_only_pca | pca_concat | atac:tfidf_lsi50_drop1 | no | 0.827 | 0.358 | 0.125 | 0.456 | 0.3 | 17.592 | – | – | – | – | – |
| clip_mlp_d32 | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.981 | 0.284 | 0.125 | 0.463 | 0.3 | 12.815 | 0.272 | 0.255 | 0.025 | 0.003 | 0.599 |
| clip_mlp_d32_reg | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.979 | 0.295 | 0.125 | 0.466 | 0.3 | 9.472 | **0.276** | **0.265** | **0.024** | 0.003 | 0.598 |
| early_fusion_pca | pca_concat | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.98 | **0.364** | 0.125 | – | – | 26.95 | – | – | – | – | – |
| rna_only_pca | pca_concat | rna:log1p_hvg2000_pca50 | no | 0.976 | 0.356 | 0.125 | – | – | 31.604 | – | – | – | – | **0.601** |
| vicreg_mlp_d32 | vicreg | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | **0.984** | 0.308 | 0.125 | **0.475** | 0.3 | 8.372 | 0.131 | 0.143 | 0.046 | 0.003 | 0.587 |
