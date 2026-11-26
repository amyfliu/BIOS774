Split scored: **test**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | kmeans_ari[label] | chance[label] | probe_bal_acc[label_fine] | kmeans_ari[label_fine] | chance[label_fine] | atac->rna per-cell r | atac->rna per-cell r (baseline) | eff_rank(joint) | probe_bal_acc[label](atac) | probe_bal_acc[label](rna) | probe_bal_acc[label_fine](atac) | probe_bal_acc[label_fine](rna) | R@5 rna->atac | R@5 atac->rna | FOSCTTM | R@5 chance | rna->atac per-feature AUROC | rna->atac per-cell AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atac_only_pca | pca_concat | atac:tfidf_lsi50_drop1 | no | 0.951 | 0.409 | 0.125 | 0.833 | 0.241 | 0.0588 | 0.457 | 0.3 | 17.7 | – | – | – | – | – | – | – | – | – | – |
| clip_mlp_d32_reg | clip | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | **0.983** | 0.281 | 0.125 | **0.884** | 0.284 | 0.0588 | 0.466 | 0.3 | 9.4 | 0.954 | 0.974 | **0.838** | **0.852** | **0.28** | **0.274** | **0.0242** | 0.0032 | 0.597 | 0.671 |
| early_fusion_pca | pca_concat | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.976 | 0.404 | 0.125 | 0.849 | **0.339** | 0.0588 | – | – | 27.0 | – | – | – | – | – | – | – | – | – | – |
| rna_only_pca | pca_concat | rna:log1p_hvg2000_pca50 | no | 0.977 | **0.425** | 0.125 | 0.831 | 0.263 | 0.0588 | – | – | 31.5 | – | – | – | – | – | – | – | – | **0.599** | 0.671 |
| vicreg_mlp_d32 | vicreg | rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 | no | 0.96 | 0.312 | 0.125 | 0.827 | 0.278 | 0.0588 | **0.476** | 0.3 | 8.35 | **0.956** | **0.977** | 0.834 | 0.825 | 0.137 | 0.142 | 0.0481 | 0.0032 | 0.591 | **0.672** |
