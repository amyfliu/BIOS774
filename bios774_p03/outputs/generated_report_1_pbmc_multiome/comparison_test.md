Split scored: **test** (n per model in metrics files)

| model | method | inputs | uses_labels | probe_bal_acc[label] | kmeans_ari[label] | chance[label] | atac->rna per-cell r | atac->rna per-cell r (baseline) | eff_rank(joint) | R@5 rna->atac | R@5 atac->rna | FOSCTTM | R@5 chance | rna->atac per-feature AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac_only_pca | pca_concat | atac:tfidf_lsi50_drop1 | no | 0.951 | 0.409 | 0.125 | 0.457 | 0.3 | 17.739 | – | – | – | – | – |
| clip_mlp_d32_reg | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | **0.983** | 0.281 | 0.125 | 0.466 | 0.3 | 9.404 | **0.28** | **0.274** | **0.024** | 0.003 | 0.597 |
| early_fusion_pca | pca_concat | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.976 | 0.404 | 0.125 | – | – | 27.021 | – | – | – | – | – |
| rna_only_pca | pca_concat | rna:log1p_hvg2000_pca50 | no | 0.977 | **0.425** | 0.125 | – | – | 31.543 | – | – | – | – | **0.599** |
| vicreg_mlp_d32 | vicreg | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.96 | 0.312 | 0.125 | **0.476** | 0.3 | 8.349 | 0.137 | 0.142 | 0.048 | 0.003 | 0.591 |
