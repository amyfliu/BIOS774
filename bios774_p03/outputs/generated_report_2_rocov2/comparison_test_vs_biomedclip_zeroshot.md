Paired differences vs **biomedclip_zeroshot** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| clip_mlp_d256 | probe balanced accuracy [label] (image) | 0.974 | 0.972 | +0.002 [-0.003, +0.008] | no |
| clip_mlp_d256 | probe balanced accuracy [label] (joint) | 0.979 | 0.976 | +0.003 [-0.001, +0.006] | no |
| clip_mlp_d256 | probe balanced accuracy [label] (text) | 0.914 | 0.919 | -0.005 [-0.013, +0.003] | no |
| clip_mlp_d256 | recall@5 a_to_b | 0.667 | 0.808 | -0.141 [-0.158, -0.122] | yes |
| clip_mlp_d256 | recall@5 b_to_a | 0.673 | 0.808 | -0.135 [-0.154, -0.117] | yes |
| early_fusion_pca | probe balanced accuracy [label] (joint) | 0.694 | 0.976 | -0.282 [-0.304, -0.259] | yes |
| image_only_pca | probe balanced accuracy [label] (image) | 0.605 | 0.972 | -0.367 [-0.391, -0.343] | yes |
| image_only_pca | probe balanced accuracy [label] (joint) | 0.605 | 0.976 | -0.371 [-0.395, -0.349] | yes |
| text_only_pca | probe balanced accuracy [label] (joint) | 0.730 | 0.976 | -0.246 [-0.293, -0.196] | yes |
| text_only_pca | probe balanced accuracy [label] (text) | 0.730 | 0.919 | -0.189 [-0.248, -0.124] | yes |
