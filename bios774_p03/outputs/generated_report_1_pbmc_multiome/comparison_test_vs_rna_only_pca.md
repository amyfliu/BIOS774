Paired differences vs **rna_only_pca** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| atac_only_pca | probe balanced accuracy [label] (joint) | 0.951 | 0.977 | -0.026 [-0.078, +0.011] | no |
| atac_only_pca | probe balanced accuracy [label_fine] (joint) | 0.833 | 0.831 | +0.002 [-0.057, +0.058] | no |
| clip_mlp_d32_reg | cross rna_to_atac__binary_top10000 per-sample mean | 0.671 | 0.671 | -0.000 [-0.000, +0.000] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label] (joint) | 0.983 | 0.977 | +0.006 [-0.012, +0.025] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label] (rna) | 0.973 | 0.977 | -0.003 [-0.013, +0.006] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label_fine] (joint) | 0.884 | 0.831 | +0.053 [+0.016, +0.086] | yes |
| clip_mlp_d32_reg | probe balanced accuracy [label_fine] (rna) | 0.852 | 0.831 | +0.022 [-0.004, +0.053] | no |
| early_fusion_pca | probe balanced accuracy [label] (joint) | 0.976 | 0.977 | -0.001 [-0.018, +0.010] | no |
| early_fusion_pca | probe balanced accuracy [label_fine] (joint) | 0.849 | 0.831 | +0.018 [-0.034, +0.067] | no |
| vicreg_mlp_d32 | cross rna_to_atac__binary_top10000 per-sample mean | 0.673 | 0.671 | +0.002 [+0.001, +0.002] | yes |
| vicreg_mlp_d32 | probe balanced accuracy [label] (joint) | 0.960 | 0.977 | -0.017 [-0.066, +0.010] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label] (rna) | 0.977 | 0.977 | +0.000 [-0.008, +0.008] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label_fine] (joint) | 0.827 | 0.831 | -0.004 [-0.036, +0.025] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label_fine] (rna) | 0.825 | 0.831 | -0.006 [-0.038, +0.031] | no |
