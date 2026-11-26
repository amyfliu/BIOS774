Paired differences vs **atac_only_pca** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| clip_mlp_d32_reg | cross atac_to_rna__log1p_hvg2000 per-sample mean | 0.466 | 0.457 | +0.009 [+0.008, +0.011] | yes |
| clip_mlp_d32_reg | probe balanced accuracy [label] (atac) | 0.954 | 0.951 | +0.002 [-0.003, +0.008] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label] (joint) | 0.983 | 0.951 | +0.031 [+0.005, +0.076] | yes |
| clip_mlp_d32_reg | probe balanced accuracy [label_fine] (atac) | 0.838 | 0.833 | +0.005 [-0.011, +0.022] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label_fine] (joint) | 0.884 | 0.833 | +0.051 [+0.012, +0.093] | yes |
| early_fusion_pca | probe balanced accuracy [label] (joint) | 0.976 | 0.951 | +0.024 [-0.007, +0.077] | no |
| early_fusion_pca | probe balanced accuracy [label_fine] (joint) | 0.849 | 0.833 | +0.016 [-0.010, +0.050] | no |
| rna_only_pca | probe balanced accuracy [label] (joint) | 0.977 | 0.951 | +0.026 [-0.009, +0.077] | no |
| rna_only_pca | probe balanced accuracy [label_fine] (joint) | 0.831 | 0.833 | -0.002 [-0.056, +0.055] | no |
| vicreg_mlp_d32 | cross atac_to_rna__log1p_hvg2000 per-sample mean | 0.476 | 0.457 | +0.019 [+0.018, +0.021] | yes |
| vicreg_mlp_d32 | probe balanced accuracy [label] (atac) | 0.956 | 0.951 | +0.004 [-0.001, +0.011] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label] (joint) | 0.960 | 0.951 | +0.009 [-0.010, +0.029] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label_fine] (atac) | 0.834 | 0.833 | +0.001 [-0.028, +0.037] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label_fine] (joint) | 0.827 | 0.833 | -0.006 [-0.056, +0.045] | no |
