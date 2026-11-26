Paired differences vs **biomedclip_zeroshot** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| clip_mlp_d512 | probe balanced accuracy [label] (image) | 0.973 | 0.972 | +0.001 [-0.006, +0.008] | no |
| clip_mlp_d512 | probe balanced accuracy [label] (joint) | 0.976 | 0.976 | -0.000 [-0.006, +0.005] | no |
| clip_mlp_d512 | probe balanced accuracy [label] (text) | 0.900 | 0.919 | -0.019 [-0.048, -0.003] | yes |
| clip_mlp_d512 | recall@5 a_to_b | 0.675 | 0.808 | -0.133 [-0.152, -0.114] | yes |
| clip_mlp_d512 | recall@5 b_to_a | 0.682 | 0.808 | -0.126 [-0.146, -0.108] | yes |
| early_fusion_pca | probe balanced accuracy [label] (joint) | 0.694 | 0.976 | -0.282 [-0.304, -0.259] | yes |
| image_only_pca | probe balanced accuracy [label] (image) | 0.605 | 0.972 | -0.367 [-0.391, -0.343] | yes |
| image_only_pca | probe balanced accuracy [label] (joint) | 0.605 | 0.976 | -0.371 [-0.395, -0.349] | yes |
| text_only_pca | probe balanced accuracy [label] (joint) | 0.730 | 0.976 | -0.246 [-0.293, -0.196] | yes |
| text_only_pca | probe balanced accuracy [label] (text) | 0.730 | 0.919 | -0.189 [-0.248, -0.124] | yes |
