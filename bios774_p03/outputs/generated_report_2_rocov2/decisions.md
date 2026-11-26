1. **Goal** *(objective skill)* — Tasks: image<->text retrieval (primary; recall@1/5/10 both ways, FOSCTTM) and imaging-modality classification from the IMAGE embedding (balanced accuracy, label 'label'). Labels NOT used in training (label-free clip/adapter). Baselines: image alone, text alone, pca_concat early fusion, BiomedCLIP zero-shot. Success: chosen model's paired image->text R@5 gain over zero-shot has CI excluding 0; otherwise recommend zero-shot. Classification judged on image side only.
   - *Evidence:* profile.json: text alone predicts label with bal.acc 0.91 (chance 0.17); 42% of captions name their label; smallest class PET n_train=113 (test 13, uninterpretable alone); test n=2000
   - *Alternatives rejected:* supcon_clip: labels not offered for training; not used
2. **Data prep** *(preprocessing skill)* — image and text: biomedclip (512-d each, shared space) for encoders and zero-shot; image pixels32_pca50 and text tfidf_svd128 as weak baselines
   - *Evidence:* features/*.json: 24000 x 512 / 24000 x 512 / 24000 x 50 / 24000 x 128; fitted on train only
   - *Alternatives rejected:* no caption cleaning (objective doesn't ask)
3. **Combine modalities** *(fusion skill)* — Primary: clip + adapter encoders on biomedclip features, dim 512, select-by val_recall. Alternative: clip with mlp projection heads. Baselines: image-only/text-only pca, pca_concat, identity zero-shot.
   - *Evidence:* fusion skill rule: pretrained shared-space encoder -> adapter
   - *Alternatives rejected:* from-scratch clip on weak features: no pretrained advantage; vicreg: no false-negative concern at n=20000
4. **Settings** *(training skill)* — adapter: clip, dim 512, bs 512, lr 1e-4, select-by val_recall, 8 min. Alternative mlp heads dim 256, lr 1e-3, hidden 256, dropout 0.2, 5 min. Budget total 13 of 20 min, 7 reserved.
   - *Evidence:* smoke: zero-shot val R@1 0.623 at epoch 0; smoke mlp from scratch 0.001->0.019
   - *Alternatives rejected:* larger lr: risks losing pretrained alignment
5. **Training check** *(training skill)* — clip_adapter_d512 kept epoch 0 (kept_initial_model true, 0.26 min); val R@1 fell to 0.576 at epoch 1 while train loss fell: overfitting. One iteration with lr 1e-5.
   - *Evidence:* clip_adapter_d512/run_config.json: val_recall_at_1_initial 0.623, kept 0.623, best_epoch 0
   - *Alternatives rejected:* stop now: allowed, but skill permits one lower-lr retry
6. **Adjust** *(training skill)* — clip_adapter_d512_lr1e5 (lr 1e-5) also kept epoch 0; stop adapter tuning. Adapters therefore equal zero-shot (identical val metrics).
   - *Evidence:* clip_adapter_d512_lr1e5/run_config.json: kept_initial_model true, val R@1 0.623 before and after
   - *Alternatives rejected:* more iterations: rule says stop
7. **Training check** *(training skill)* — clip_mlp_d256: best epoch 28 of 48, early stop; val loss rises after epoch ~10 while train falls (overfitting) but R@1 0.484 vs chance far lower; no collapse.
   - *Evidence:* clip_mlp_d256/run_config.json: initial R@1 0.001, kept 0.484; metrics_val.json: R@5 0.668 vs zero-shot 0.810
   - *Alternatives rejected:* stronger regularisation: would not close a 0.14 gap
8. **Evaluate** *(evaluation skill)* — Choose frozen BiomedCLIP (adapter models are identical to it). Report on test: zero-shot, clip_mlp_d256 (alternative), and the three simple baselines.
   - *Evidence:* comparison_val.md: R@5 img->txt 0.81 vs 0.668 (mlp); image-side probe 0.968 vs 0.969 (mlp); pixels PCA image 0.585; chance R@5 0.0025, probe chance 0.167
   - *Alternatives rejected:* clip_mlp_d256: worse retrieval, equal classification
9. **Downstream** *(recommendation skill)* — Retrieval: zero-shot image->text R@1 0.4965, R@5 0.808 [0.7905,0.824], R@10 0.882 of 2000 candidates (chance R@5 0.0025); text->image R@5 0.8085. Classification from image embedding: probe bal.acc 0.972 vs pixel-PCA image alone 0.605 (paired +0.367 [+0.343,+0.390]); weakest large class X-ray recall 0.904; PET n=13 uninterpretable.
   - *Evidence:* biomedclip_zeroshot/metrics.json; comparison_test_vs_image_only_pca.md; comparison_test_vs_biomedclip_zeroshot.md
   - *Alternatives rejected:* none
10. **Recommend** *(recommendation skill)* — Verdict: multimodal enables (pretrained shared space gives retrieval; classification already ~0.97 from image). Recommend frozen BiomedCLIP image/text embeddings (exported). Trained MLP heads: retrieval R@5 -0.141 [-0.158,-0.122] vs zero-shot; classification +0.002 [-0.003,+0.008] (no difference). Caveat: BiomedCLIP was pretrained on PubMed Central figure-caption pairs (ROCOv2 is also PMC), so zero-shot scores may be inflated by overlap. Would change: an in-domain pretrained-free test set, or a larger lr-tuned adapter gaining on val.
   - *Evidence:* comparison_test_vs_biomedclip_zeroshot.md
   - *Alternatives rejected:* clip_mlp_d256, adapters (identical to zero-shot), pixel/TF-IDF baselines
