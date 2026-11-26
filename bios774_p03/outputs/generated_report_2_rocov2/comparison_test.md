Split scored: **test**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | probe_bal_acc[label](image) | probe_bal_acc[label](text) | kmeans_ari[label] | chance[label] | R@5 image->text | R@5 text->image | FOSCTTM | R@5 chance | eff_rank(joint) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| biomedclip_zeroshot | identity | image:biomedclip,text:biomedclip | no | 0.976 | 0.972 | **0.919** | **0.627** | 0.167 | **0.808** | **0.808** | **0.00367** | 0.0025 | 52.6 |
| clip_mlp_d256 | clip | image:biomedclip,text:biomedclip | no | **0.979** | **0.974** | 0.914 | 0.618 | 0.167 | 0.667 | 0.673 | 0.00623 | 0.0025 | 27.5 |
| early_fusion_pca | pca_concat | image:pixels32_pca50,text:tfidf_svd128 | no | 0.694 | – | – | 0.14 | 0.167 | – | – | – | – | 31.0 |
| image_only_pca | pca_concat | image:pixels32_pca50 | no | 0.605 | – | – | 0.106 | 0.167 | – | – | – | – | 31.1 |
| text_only_pca | pca_concat | text:tfidf_svd128 | no | 0.73 | – | – | 0.0248 | 0.167 | – | – | – | – | 31.1 |
