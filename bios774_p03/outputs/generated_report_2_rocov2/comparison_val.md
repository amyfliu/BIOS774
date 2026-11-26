Split scored: **val**. Per-model numbers; for 'is A better than B' use the paired table (--reference).

| model | method | inputs | uses_labels | probe_bal_acc[label] | probe_bal_acc[label](image) | probe_bal_acc[label](text) | kmeans_ari[label] | chance[label] | R@5 image->text | R@5 text->image | FOSCTTM | R@5 chance | eff_rank(joint) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| biomedclip_zeroshot | identity | image:biomedclip,text:biomedclip | no | **0.972** | 0.968 | **0.954** | 0.465 | 0.167 | **0.81** | **0.807** | **0.004** | 0.003 | 52.881 |
| clip_adapter_d512 | clip | image:biomedclip,text:biomedclip | no | **0.972** | 0.968 | **0.954** | 0.465 | 0.167 | **0.81** | **0.807** | **0.004** | 0.003 | 52.881 |
| clip_adapter_d512_lr1e5 | clip | image:biomedclip,text:biomedclip | no | **0.972** | 0.968 | **0.954** | 0.465 | 0.167 | **0.81** | **0.807** | **0.004** | 0.003 | 52.881 |
| clip_mlp_d512 | clip | image:biomedclip,text:biomedclip | no | 0.968 | **0.969** | 0.93 | **0.564** | 0.167 | 0.693 | 0.689 | 0.008 | 0.003 | 33.422 |
| early_fusion_pca | pca_concat | image:pixels32_pca50,text:tfidf_svd128 | no | 0.703 | – | – | 0.138 | 0.167 | – | – | – | – | 31.006 |
| image_only_pca | pca_concat | image:pixels32_pca50 | no | 0.585 | – | – | 0.08 | 0.167 | – | – | – | – | 31.108 |
| text_only_pca | pca_concat | text:tfidf_svd128 | no | 0.809 | – | – | 0.054 | 0.167 | – | – | – | – | 31.311 |
