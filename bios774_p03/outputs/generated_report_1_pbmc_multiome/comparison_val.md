Split scored: **val**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | kmeans_ari[label] | chance[label] | probe_bal_acc[label_fine] | kmeans_ari[label_fine] | chance[label_fine] | atac->rna per-cell r | atac->rna per-cell r (baseline) | eff_rank(joint) | probe_bal_acc[label](atac) | probe_bal_acc[label](rna) | probe_bal_acc[label_fine](atac) | probe_bal_acc[label_fine](rna) | R@5 rna->atac | R@5 atac->rna | FOSCTTM | R@5 chance | rna->atac per-feature AUROC | rna->atac per-cell AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac_only_pca | pca_concat | atac:tfidf_lsi50_drop1 | no | 0.827 | 0.358 | 0.125 | 0.818 | 0.255 | 0.0588 | 0.456 | 0.3 | 17.6 | – | – | – | – | – | – | – | – | – | – |
| clip_mlp_d32 | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.981 | 0.284 | 0.125 | 0.832 | 0.257 | 0.0588 | 0.463 | 0.3 | 12.8 | 0.853 | 0.977 | **0.85** | 0.84 | 0.272 | 0.255 | 0.0254 | 0.0032 | 0.599 | **0.672** |
| clip_mlp_d32_reg | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.979 | 0.295 | 0.125 | **0.9** | 0.29 | 0.0588 | 0.466 | 0.3 | 9.47 | 0.894 | 0.973 | 0.848 | **0.889** | **0.276** | **0.265** | **0.024** | 0.0032 | 0.598 | **0.672** |
| early_fusion_pca | pca_concat | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.98 | **0.364** | 0.125 | 0.88 | **0.292** | 0.0588 | – | – | 26.9 | – | – | – | – | – | – | – | – | – | – |
| rna_only_pca | pca_concat | rna:log1p_hvg2000_pca50 | no | 0.976 | 0.356 | 0.125 | 0.849 | 0.238 | 0.0588 | – | – | 31.6 | – | – | – | – | – | – | – | – | **0.601** | 0.671 |
| vicreg_mlp_d32 | vicreg | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | **0.984** | 0.308 | 0.125 | 0.868 | 0.267 | 0.0588 | **0.475** | 0.3 | 8.37 | **0.963** | **0.98** | 0.834 | 0.843 | 0.131 | 0.143 | 0.0465 | 0.0032 | 0.587 | **0.672** |
