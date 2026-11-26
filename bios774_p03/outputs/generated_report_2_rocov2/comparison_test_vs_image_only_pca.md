Paired differences vs **image_only_pca** on the test split (model − reference; 95% paired bootstrap, 2000 resamples of the same samples)

| model | metric | model | reference | difference [95% CI] | CI excludes 0 |
|---|---|---|---|---|---|
| biomedclip_zeroshot | probe balanced accuracy [label] (image) | 0.972 | 0.605 | +0.367 [+0.343, +0.390] | yes |
| biomedclip_zeroshot | probe balanced accuracy [label] (joint) | 0.976 | 0.605 | +0.371 [+0.346, +0.394] | yes |
| clip_mlp_d256 | probe balanced accuracy [label] (image) | 0.974 | 0.605 | +0.369 [+0.346, +0.393] | yes |
| clip_mlp_d256 | probe balanced accuracy [label] (joint) | 0.979 | 0.605 | +0.373 [+0.350, +0.397] | yes |
| early_fusion_pca | probe balanced accuracy [label] (joint) | 0.694 | 0.605 | +0.089 [+0.070, +0.107] | yes |
| text_only_pca | probe balanced accuracy [label] (joint) | 0.730 | 0.605 | +0.125 [+0.070, +0.182] | yes |
