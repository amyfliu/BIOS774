1. **Goal** *(objective skill)* — Tasks: (1) image<->text retrieval, headline recall@5 both directions + FOSCTTM; (2) modality classification from IMAGE embedding, headline probe balanced accuracy (label 'label', 6 classes). No labels in training (label-free). Baselines: image alone, text alone, pca_concat early fusion, BiomedCLIP zero-shot identity. Success: tuned model's i2t and t2i recall@5 exceed zero-shot's by paired CI excluding 0, and image-probe bal. acc not below zero-shot image probe. Text-side classification not used (text shortcut). PET has 13 test samples: uninterpretable alone.
   - *Evidence:* profile.json: text predicts label from words bal.acc 0.91 (chance 0.17); 42% captions name the label; test PET n=13; n_train 20000
   - *Alternatives rejected:* supcon_clip: labels not granted for training, rejected
2. **Data prep** *(preprocessing skill)* — image:biomedclip and text:biomedclip (512-d, shared space, cached) for encoders and zero-shot; image:pixels32_pca50 and text:tfidf_svd128 as simple baselines. No caption cleaning.
   - *Evidence:* features/*.json: 24000 rows each; profile.json 70% grayscale images, median 16 words
   - *Alternatives rejected:* No cross-modal prediction targets: objective does not ask for them
3. **Combine modalities** *(fusion skill)* — Primary: clip with adapter encoders on biomedclip features, dim 512, select-by val_recall (light fine-tuning). Alternative: clip with mlp projection heads on same features. Baselines: image-only pca, text-only pca, pca_concat early fusion, biomedclip zero-shot identity. Label-free.
   - *Evidence:* val zero-shot R@5 i2t 0.810 t2i 0.807, probe bal acc image 0.968 (metrics_val.json); retrieval already far above chance 0.0025
   - *Alternatives rejected:* vicreg: n_train 20000 is large enough for clip; supcon_clip: labels not allowed
4. **Settings** *(training skill)* — adapter: dim 512, batch 512, lr 1e-4, hidden 256, dropout 0.1, select-by val_recall, max 8 min. mlp alternative: dim 512, batch 512, lr 1e-3, hidden 512, layers 2, dropout 0.2, select-by val_recall, max 5 min. 7 min reserved for iteration (total <= 20).
   - *Evidence:* training skill defaults; n_train 20000
   - *Alternatives rejected:* larger lr: risks leaving the pretrained optimum
5. **Training check** *(training skill)* — clip_adapter_d512: kept_initial_model True, best_epoch 0, early stop at epoch 20, val R@1 initial 0.623 vs 0.576-0.592 after training; train loss fell 2.40->1.16, val loss fell 2.77->1.35. Per skill: one iteration with 3-10x smaller lr.
   - *Evidence:* clip_adapter_d512/run_config.json summary, train_log.json; 0.23 min used
   - *Alternatives rejected:* more epochs: val recall not improving
6. **Adjust** *(training skill)* — clip_adapter_d512_lr1e5 (lr 1e-5): again kept epoch 0; val R@1 fell to 0.565-0.577 during training. Stop tuning adapters.
   - *Evidence:* clip_adapter_d512_lr1e5/run_config.json: kept_initial_model True, minutes 0.21
   - *Alternatives rejected:* further lr cuts: skill says stop
7. **Evaluate** *(evaluation skill)* — Chosen for report: BiomedCLIP zero-shot (adapter runs are numerically identical to it since epoch 0 kept). Alternative clip_mlp_d512 (trained from scratch projections on same features) is worse at retrieval and equal on classification. Baselines on val are far below on classification.
   - *Evidence:* comparison_val.md: zero-shot R@5 i2t 0.810 / t2i 0.807, image probe 0.968; mlp R@5 0.693/0.689, image probe 0.969; image-only pca 0.585; text-only pca 0.809; early fusion 0.703
   - *Alternatives rejected:* clip_mlp: lower retrieval
8. **Downstream** *(recommendation skill)* — Test (n=2000): zero-shot R@5 i2t 0.808 [0.7905,0.824], t2i 0.808 (chance 0.0025), R@1 0.497/0.513; image probe bal acc 0.972 [0.9635,0.9791] (chance 0.167). clip_mlp_d512 retrieval worse by -0.133 [-0.152,-0.114] i2t and -0.126 [-0.146,-0.108] t2i; image probe +0.001 [-0.006,+0.008] (no difference). Pixel/TF-IDF baselines 0.605 image, 0.694 early fusion. Weakest large class: X-ray (recall 0.904, n=565). Modality gap 0.80 yet retrieval works.
   - *Evidence:* metrics.json, comparison_test.md, comparison_test_vs_biomedclip_zeroshot.md
   - *Alternatives rejected:* text-side classification not reported as evidence (text shortcut 0.91)
9. **Recommend** *(recommendation skill)* — Verdict: No measurable gain from training; use frozen BiomedCLIP zero-shot for caption<->image retrieval and image encoder + linear probe for modality classification. Caveat: BiomedCLIP was pretrained on PubMed Central figure-caption pairs, same source as ROCOv2, so scores may be inflated by overlap (cannot be measured here). Would change if: test on an out-of-PMC dataset, or labelled/larger-scale fine-tuning.
   - *Evidence:* comparison_test_vs_biomedclip_zeroshot.md
   - *Alternatives rejected:* clip_mlp_d512: worse retrieval, same classification
