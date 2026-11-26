Split scored: **test**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | probe_bal_acc[label](image) | probe_bal_acc[label](text) | kmeans_ari[label] | chance[label] | R@5 image->text | R@5 text->image | FOSCTTM | R@5 chance | eff_rank(joint) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| biomedclip_zeroshot | identity | image:biomedclip,text:biomedclip | no | **0.976** | 0.972 | **0.919** | **0.627** | 0.167 | **0.808** | **0.808** | **0.004** | 0.003 | 52.605 |
| clip_mlp_d512 | clip | image:biomedclip,text:biomedclip | no | **0.976** | **0.973** | 0.9 | 0.611 | 0.167 | 0.675 | 0.682 | 0.008 | 0.003 | 33.466 |
| early_fusion_pca | pca_concat | image:pixels32_pca50,text:tfidf_svd128 | no | 0.694 | – | – | 0.14 | 0.167 | – | – | – | – | 31.034 |
| image_only_pca | pca_concat | image:pixels32_pca50 | no | 0.605 | – | – | 0.106 | 0.167 | – | – | – | – | 31.085 |
| text_only_pca | pca_concat | text:tfidf_svd128 | no | 0.73 | – | – | 0.025 | 0.167 | – | – | – | – | 31.098 |
