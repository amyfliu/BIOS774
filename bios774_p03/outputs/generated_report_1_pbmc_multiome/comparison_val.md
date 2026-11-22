Split scored: **val**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | kmeans_ari[label] | chance[label] | probe_bal_acc[label_fine] | kmeans_ari[label_fine] | chance[label_fine] | atac->rna per-cell r | atac->rna per-cell r (baseline) | eff_rank(joint) | R@5 rna->atac | R@5 atac->rna | FOSCTTM | R@5 chance | rna->atac per-feature AUROC | rna->atac per-cell AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac_only_pca | pca_concat | atac:tfidf_lsi50_drop1 | no | 0.827 | 0.358 | 0.125 | 0.818 | 0.255 | 0.059 | 0.456 | 0.3 | 17.592 | – | – | – | – | – | – |
| clip_mlp_d32 | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.981 | 0.284 | 0.125 | 0.832 | 0.257 | 0.059 | 0.463 | 0.3 | 12.815 | 0.272 | 0.255 | 0.025 | 0.003 | 0.599 | **0.672** |
| clip_mlp_d32_reg | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.979 | 0.295 | 0.125 | **0.9** | 0.29 | 0.059 | 0.466 | 0.3 | 9.472 | **0.276** | **0.265** | **0.024** | 0.003 | 0.598 | **0.672** |
| early_fusion_pca | pca_concat | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.98 | **0.364** | 0.125 | 0.88 | **0.292** | 0.059 | – | – | 26.95 | – | – | – | – | – | – |
| rna_only_pca | pca_concat | rna:log1p_hvg2000_pca50 | no | 0.976 | 0.356 | 0.125 | 0.849 | 0.238 | 0.059 | – | – | 31.604 | – | – | – | – | **0.601** | 0.671 |
| vicreg_mlp_d32 | vicreg | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | **0.984** | 0.308 | 0.125 | 0.868 | 0.267 | 0.059 | **0.475** | 0.3 | 8.372 | 0.131 | 0.143 | 0.046 | 0.003 | 0.587 | **0.672** |
