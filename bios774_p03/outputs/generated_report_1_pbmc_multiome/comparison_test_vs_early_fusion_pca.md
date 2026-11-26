Paired differences vs **early_fusion_pca** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| atac_only_pca | probe balanced accuracy [label] (joint) | 0.951 | 0.976 | -0.024 [-0.075, +0.008] | no |
| atac_only_pca | probe balanced accuracy [label_fine] (joint) | 0.833 | 0.849 | -0.016 [-0.047, +0.013] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label] (joint) | 0.983 | 0.976 | +0.007 [-0.002, +0.021] | no |
| clip_mlp_d32_reg | probe balanced accuracy [label_fine] (joint) | 0.884 | 0.849 | +0.035 [-0.001, +0.074] | no |
| rna_only_pca | probe balanced accuracy [label] (joint) | 0.977 | 0.976 | +0.001 [-0.011, +0.017] | no |
| rna_only_pca | probe balanced accuracy [label_fine] (joint) | 0.831 | 0.849 | -0.018 [-0.066, +0.033] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label] (joint) | 0.960 | 0.976 | -0.015 [-0.067, +0.017] | no |
| vicreg_mlp_d32 | probe balanced accuracy [label_fine] (joint) | 0.827 | 0.849 | -0.022 [-0.071, +0.035] | no |
