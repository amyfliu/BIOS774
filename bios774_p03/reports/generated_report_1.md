# generated_report_1: PBMC Multiome — a joint RNA+ATAC cell representation

**Verdict:** Multimodal helps for fine-grained cell types: the joint aligned (CLIP-style) embedding beats RNA alone and ATAC alone, but not clearly early-fusion PCA; cross-modal prediction is only modestly above baseline.

## 1. Summary

- **Question:** Can one RNA+ATAC representation identify cell types at least as well as either modality alone, and how well can each modality be predicted from the other?
- **Approach:** Compressed each modality (50 components), trained a label-free contrastive model (CLIP-style, 32 dimensions), and compared it with RNA alone, ATAC alone and simple concatenation on held-out cells.
- **Result:** On 17 fine cell types the joint embedding reaches balanced accuracy 0.884 vs 0.831 (RNA alone) and 0.833 (ATAC alone); gains of +0.053 and +0.051 with CIs excluding 0. Versus early fusion (0.849) the gain, +0.035, has a CI that includes 0. Coarse types are saturated (about 0.98 for all except ATAC alone, 0.951).
- **Recommendation:** Use the joint embedding for cell-type annotation; for predicting RNA from ATAC, ATAC alone is almost as good; RNA→ATAC is barely better than predicting the average profile.

## 2. Data at a glance

- 10,405 nuclei from one healthy donor, measured in both RNA (25,598 genes) and ATAC (111,698 peaks); split 7,285 / 1,560 / 1,560 (train / validation / test).
- RNA is sparse counts (94% zeros); ATAC is near-binary (72% of non-zero values ≤ 2, 92% zeros).
- Labels: 8 coarse and 17 fine cell types, predicted by a reference classifier (CellTypist) **from RNA only**, so they favour RNA. Several fine classes have < 10 test cells (pDC 7, Plasma cells 6, CD16- NK cells 5, Megakaryocyte precursor 2) and cannot be judged individually.
- No labels were used to train any model.

![Profile of the data](../outputs/generated_report_1_pbmc_multiome/profile.png){width=90%}

## 3. Decision path

| # | Step | Decision | Why |
|:---|:------|:------------------|:-------------------------|
| 1 | Goal | Joint cell-type ID, then cross-modal prediction | Objective names classification first, prediction second; labels stay a yardstick |
| 2 | Data prep | Standard RNA and ATAC reductions, plus targets | RNA is overdispersed counts; ATAC is near-binary, so each gets its usual treatment |
| 3 | Combine modalities | Compare aligned CLIP against simple PCA baselines | Prediction needs a shared space; baselines show whether learning helps |
| 4 | Settings | Default settings sized to 7k training cells | Inputs are small, so overfitting rather than time is the risk |
| 5 | Training check | CLIP overfits early; VICReg weaker matching | Validation loss rose after epoch 11 while training loss kept falling |
| 6 | Adjust | Retry CLIP with stronger regularisation | First CLIP run overfit quickly; more dropout may generalise better |
| 7 | Evaluate | Choose regularised CLIP on validation | It scored highest on fine cell types and matched cells best |
| 8 | Downstream | Cell-type probe and cross-modal prediction on test | These are the two tasks the objective named |
| 9 | Recommend | Use regularised CLIP joint embedding; multimodal helps | It beats each single modality on fine cell types, with CI above zero |


The turning point was the first CLIP run: validation loss rose after epoch 11 while training loss kept falling (overfitting). Stronger regularisation (more dropout, weight decay, smaller network) trained to epoch 127 and raised the validation fine-type score from 0.832 to 0.900.

## 4. Methods compared

- **RNA alone / ATAC alone:** PCA (32 dims) of the processed single modality.
- **Early fusion:** the two processed modalities standardised, concatenated, PCA to 32 dims; no learning.
- **Joint CLIP (chosen):** one small neural network per modality, trained so that the two views of the same cell land close together and other cells land apart (contrastive learning); the joint embedding is the average of the two. Label-free.
- **Joint VICReg (alternative):** same networks, trained to match paired cells without negative examples (non-contrastive).
- Chosen on validation data only (fine-type score 0.900 vs 0.880 early fusion, 0.868 VICReg, 0.849 RNA alone, 0.818 ATAC alone); a first, less regularised CLIP (0.832) was discarded.

## 5. Results

Test set, 1,560 cells. Balanced accuracy = average per-class recall of a linear classifier trained on the embedding (chance 0.059 for 17 fine types, 0.125 for 8 coarse types).

| Model | Fine types (17) | Coarse types (8) | ATAC→RNA per-cell r | RNA→ATAC per-cell AUROC |
|:---|:---|:---|:---|:---|
| RNA alone | 0.831 | 0.977 | – | 0.671 |
| ATAC alone | 0.833 | 0.951 | 0.457 | – |
| Early fusion | 0.849 | 0.976 | – | – |
| Joint CLIP | **0.884** [0.858, 0.908] | **0.983** | 0.466 | 0.671 |
| Joint VICReg | 0.827 | 0.960 | 0.476 | 0.673 |
| Train-mean baseline | – | – | 0.300 | 0.661 |

Paired differences on the same test cells (Joint CLIP minus reference, fine types): vs RNA alone +0.053 [+0.016, +0.086]; vs ATAC alone +0.051 [+0.012, +0.093]; vs early fusion +0.035 [−0.001, +0.074]. Coarse types: vs RNA alone +0.006 [−0.012, +0.025], i.e. no difference.

![Embedding of test cells: left coloured by coarse type, right both modalities in the shared space](../outputs/generated_report_1_pbmc_multiome/clip_mlp_d32_reg/embedding.png){width=90%}

Left: coarse cell types form distinct groups (monocytes, B cells, T cells split into several sub-clouds). Right: RNA and ATAC points of the same cell (grey lines) sit in the same regions, with the two modalities partly offset from each other.

## 6. Downstream analyses

### 6.1 Cell-type identification
A linear classifier trained on the embeddings of training cells predicts the type of test cells. The joint CLIP embedding gets 0.884 on fine types (see Section 5 for baselines and paired differences). Weakest large class: Tcm/Naive helper T cells (573 test cells) with recall 0.61 (RNA alone 0.60, ATAC alone 0.56), most likely confused with other T-cell subtypes. Classes helped by the joint embedding include Tem/Effector helper T (0.87 vs 0.77 RNA alone) and CD16- NK (1.00 vs 0.40, but only 5 cells). Because the labels were derived from RNA, a joint model can at best tie on them; the gain seen here is therefore not an artefact in its favour. The joint embedding's k-means agreement with labels (ARI 0.284 fine) is below RNA alone on coarse types (0.281 vs 0.425), so the classes are linearly separable but not compact clusters.

### 6.2 Cross-modal prediction
Ridge regression from the source embedding to the target modality, scored on test cells.

- **ATAC→RNA** (2,000 variable genes): per-cell correlation 0.466 for the joint CLIP model's ATAC embedding and 0.476 for VICReg, vs 0.457 for ATAC alone and 0.300 for predicting the train average. Per gene the correlation is low (0.118 on average; 13.5% of genes above 0.3). Paired, CLIP is +0.009 [+0.008, +0.011] over ATAC alone, a gain that excludes 0 but is practically tiny.
- **RNA→ATAC** (10,000 most detected peaks, on/off): per-cell AUROC 0.671 vs 0.661 for the average profile (per-peak 0.597 vs 0.5). RNA does not usefully predict individual peaks with a 32-dimensional linear map.
- Extra: matching a cell's RNA view to its ATAC view among 1,560 test cells puts the true partner in the top 5 for 28% of cells (chance 0.3%); average rank fraction 0.024 (0.5 = random).

## 7. Recommendation

**Multimodal helps** (fine types only). Use the joint CLIP embedding for cell-type annotation: balanced accuracy 0.884 vs 0.831 for RNA alone (+0.053 [+0.016, +0.086]) and 0.833 for ATAC alone. Early-fusion PCA (0.849) is a simpler near-equal: the paired gap of +0.035 [−0.001, +0.074] just includes 0, so choose early fusion if simplicity matters. For coarse types any representation works. For ATAC→RNA prediction, ATAC alone is almost as good as the joint model; RNA→ATAC prediction is not recommended. Caveats: labels come from RNA, one donor, rare classes too small to judge. This would change if a larger test set resolved the early-fusion gap, or if labels not derived from RNA were available. The exported file contains the per-modality (RNA and ATAC) encoder outputs of the joint model.

## 8. Limitations

- Single donor and one dataset; test set 1,560 cells, so CIs are wide (about ±0.04).
- Reference labels are predicted from RNA, partly circular for RNA, and several classes have < 10 test cells.
- One run per configuration (single seed); hyper-parameter search was limited to one regularisation change.
- Cross-modal prediction used only a linear (ridge) map from a 32-dimensional embedding; richer models could do better.


## Appendix A. Full decision log

1. **Goal** *(objective skill)* — Primary: integration/classification on label_fine (17 classes; label as secondary), headline = linear-probe balanced accuracy of joint vs RNA alone vs ATAC alone vs early fusion. Secondary: cross-modal prediction rna->atac (binary_top10000, AUROC) and atac->rna (log1p_hvg2000, per-sample/per-feature Pearson r) vs train-mean baseline and single-modality baseline. No labels in training (label-free clip/vicreg). Success: joint probe bal-acc on label_fine >= best single modality, paired-difference 95% CI excluding or including 0 (>= means CI lower bound > -0.02 and no significant loss); if CI includes 0 the verdict is no measurable gain. Prediction: per-sample r / AUROC clearly above train-mean baseline. Change course if joint < best single modality by a paired CI excluding 0.
   - *Evidence:* profile.json: 8 coarse / 17 fine classes; test classes <10 cells: pDC 7, Plasma 6, CD16- NK 5, Megakaryocyte 2 (uninterpretable individually); n_train 7285; labels from CellTypist on RNA only (DATA.md) so favour RNA
   - *Alternatives rejected:* supcon_clip: labels not allowed for training by objective; clustering metrics: not asked
2. **Data prep** *(preprocessing skill)* — rna: log1p_hvg2000_pca50 (encoder input); atac: tfidf_lsi50_drop1 (encoder input, 49 dims; depth corr of first component r=0.86 so drop1 kept). Targets: rna log1p_hvg2000, atac binary_top10000.
   - *Evidence:* profile.json: rna 68% nonzeros =1, overdispersion 1.23; atac 72% nonzeros <=2, 91.8% zeros; atac__tfidf_lsi50_drop1.json first_component_corr_with_depth 0.856, explained var 0.067
   - *Alternatives rejected:* pca100/lsi100: only if fine-label probe is weak
3. **Combine modalities** *(fusion skill)* — Baselines: rna_only_pca, atac_only_pca (pca_concat single input), early_fusion_pca (pca_concat both). Primary: clip with mlp encoders, dim 32, label-free. Alternative: vicreg mlp dim 32 (n_train 7285, many look-alike cells of same type -> false negatives under InfoNCE).
   - *Evidence:* profile.json n_train 7285; 17 fine classes -> dim 32; objective includes cross-modal prediction
   - *Alternatives rejected:* supcon_clip: labels not allowed; simclr: single modality only
4. **Settings** *(training skill)* — clip_mlp_d32: batch 512, lr 1e-3, hidden 256, 2 layers, dropout 0.1, wd 1e-4, select-by val_loss, max 8 min. vicreg_mlp_d32: batch 256, same else, max 5 min. Smoke tests ran OK. Total reserved 13 of 20 min; 7 kept for iterations.
   - *Evidence:* training skill defaults; n_train 7285; smoke runs finished in seconds
   - *Alternatives rejected:* select-by val_recall: objective primary is classification not retrieval
5. **Training check** *(training skill)* — clip_mlp_d32: best epoch 11 of 31 (early stop), val loss 3.27, val R@1 0.137 vs initial 0.004; overfitting pattern. vicreg_mlp_d32: best epoch 59 of 79, val R@1 0.053. Both well above chance (0.0006). Next: regularised CLIP iteration.
   - *Evidence:* clip_mlp_d32/run_config.json summary; vicreg_mlp_d32/run_config.json summary
   - *Alternatives rejected:* accept as is: overfitting rule says add dropout/wd
6. **Adjust** *(training skill)* — clip_mlp_d32_reg: dropout 0.3, weight decay 1e-3, hidden 128, max 3 min (budget total 8+5+3+3 baselines = 19 of 20)
   - *Evidence:* clip_mlp_d32 best epoch 11/31
   - *Alternatives rejected:* lower lr: not indicated
7. **Evaluate** *(evaluation skill)* — On val, clip_mlp_d32_reg is chosen as the primary joint model: label_fine probe 0.900 vs RNA alone 0.849, ATAC alone 0.818, early fusion 0.880, clip_mlp_d32 0.832, vicreg 0.868; R@5 0.276/0.265, FOSCTTM 0.024. vicreg kept as alternative on test (best atac->rna r 0.475). rna->atac per-cell AUROC 0.672 vs train-mean 0.661 for all models: weak. No more iterations; budget used 19 of 20 min. Test run on rna_only, atac_only, early_fusion, clip_mlp_d32_reg, vicreg.
   - *Evidence:* comparison_val.md
   - *Alternatives rejected:* clip_mlp_d32 (overfit, lower probe), vicreg (weaker matching, lower fine probe)
8. **Downstream** *(recommendation skill)* — Test: label_fine probe bal-acc joint CLIP-reg 0.884 [0.858,0.908] vs RNA alone 0.831, ATAC alone 0.833, early fusion 0.849 (chance 0.059); paired diff vs RNA +0.053 [+0.016,+0.086], vs ATAC +0.051 [+0.012,+0.093], vs early fusion +0.035 [-0.001,+0.074]. Coarse label 0.983 vs 0.977/0.951/0.976 (vs RNA +0.006 [-0.012,+0.025]). Weakest large class: Tcm/Naive helper T (573 test cells) recall 0.613. Prediction atac->rna per-cell r: CLIP-reg 0.466, vicreg 0.476, ATAC-alone 0.457, train-mean 0.300; per-feature r 0.118, 13.5% of genes r>0.3. rna->atac per-cell AUROC 0.671 vs train-mean 0.661 (per-feature 0.597 vs 0.5): barely above baseline. Retrieval (extra): FOSCTTM 0.024 (chance 0.5), R@5 0.28 (chance 0.003).
   - *Evidence:* comparison_test.md; comparison_test_vs_rna_only_pca.md; comparison_test_vs_atac_only_pca.md; comparison_test_vs_early_fusion_pca.md; clip_mlp_d32_reg/metrics.json
   - *Alternatives rejected:* none
9. **Recommend** *(recommendation skill)* — Verdict: Multimodal helps (fine cell types only). Recommend clip_mlp_d32_reg joint embedding for cell-type annotation (+0.053 vs RNA, +0.051 vs ATAC, CIs exclude 0; vs early fusion +0.035, CI includes 0, so simple PCA concat is a near-equal, simpler option). For atac->rna prediction, ATAC alone ridge is nearly as good (0.457 vs 0.466; tiny gain); rna->atac not usefully predictable. Caveats: labels predicted from RNA (CellTypist), one donor, rare classes <10 test cells, kmeans ARI lower than baselines. Would change: a test gain vs early fusion with CI excluding 0, or labels not derived from RNA.
   - *Evidence:* comparison_test_vs_*.md
   - *Alternatives rejected:* early_fusion_pca: CI vs clip includes 0; vicreg: no probe gain

## Appendix B. Training details

- Joint CLIP (chosen): MLP encoders (hidden 128, 2 layers), dropout 0.3, weight decay 1e-3, batch 512, lr 1e-3, dim 32; 147 epochs run, best epoch 127, early stop, about 0.3 min; val R@1 0.141 vs 0.001 at start (chance 0.0006); no dead dimensions, effective rank 9.4, modality gap 0.12.
- First CLIP (dropout 0.1, hidden 256): best epoch 11 of 31 (overfit), val R@1 0.137. VICReg: best epoch 59 of 79, val R@1 0.053.
- Budget: 3 baselines × 1 + 8 + 5 + 3 = 19 of 20 minutes reserved.

![Loss curves of the chosen model](../outputs/generated_report_1_pbmc_multiome/clip_mlp_d32_reg/loss.png){width=90%}

## Appendix C. Glossary

- **Balanced accuracy:** average of per-class recall; not dominated by large classes.
- **Linear probe:** a linear classifier trained on the embedding.
- **Contrastive learning (CLIP):** training so matching pairs are close and non-matching pairs far apart.
- **VICReg:** matching pairs without negatives, with variance and decorrelation penalties.
- **Early fusion:** concatenating processed modalities then PCA.
- **LSI / TF-IDF:** standard reduction for ATAC peak counts.
- **AUROC:** probability a random open peak is scored above a random closed one (0.5 = chance).
- **Pearson r (per cell):** correlation between predicted and observed profile of one cell.
- **FOSCTTM:** average fraction of cells closer than the true partner (0 perfect, 0.5 random).
- **Paired bootstrap CI:** interval for a difference, resampling the same cells for both models.
- **ARI:** agreement of k-means clusters with labels (0 random, 1 identical).

## Appendix D. Reproduce

Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome
python3 tools/preprocess.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome --modality rna --spec log1p_hvg2000_pca50
python3 tools/preprocess.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome --modality atac --spec tfidf_lsi50_drop1
python3 tools/preprocess.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome --modality rna --spec log1p_hvg2000
python3 tools/preprocess.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome --modality atac --spec binary_top10000
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/rna_only_pca --inputs rna:log1p_hvg2000_pca50 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/atac_only_pca --inputs atac:tfidf_lsi50_drop1 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/early_fusion_pca --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/clip_mlp_d32 --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 --method clip --dim 32 --encoder mlp --batch-size 512 --lr 1e-3 --epochs 300 --patience 20 --max-minutes 8
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/vicreg_mlp_d32 --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 --method vicreg --dim 32 --encoder mlp --batch-size 256 --lr 1e-3 --epochs 300 --patience 20 --max-minutes 5
python3 tools/train.py --data data/pbmc_multiome --out outputs/generated_report_1_pbmc_multiome/clip_mlp_d32_reg --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 --method clip --dim 32 --encoder mlp --batch-size 512 --lr 1e-3 --dropout 0.3 --weight-decay 1e-3 --hidden 128 --epochs 300 --patience 20 --max-minutes 3
# (not reproducible verbatim — used a shell variable) /usr/bin/python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/rna_only_pca --split val --tasks structure,collapse,cross --labels label,label_fine $X
# (not reproducible verbatim — used a shell variable) /usr/bin/python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/atac_only_pca --split val --tasks structure,collapse,cross --labels label,label_fine $X
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/early_fusion_pca --split val --tasks structure,collapse --labels label,label_fine
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/rna_only_pca --split val --tasks structure,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/atac_only_pca --split val --tasks structure,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/clip_mlp_d32 --split val --tasks structure,retrieval,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/clip_mlp_d32_reg --split val --tasks structure,retrieval,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/vicreg_mlp_d32 --split val --tasks structure,retrieval,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/compare_runs.py --out outputs/generated_report_1_pbmc_multiome --split val
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/rna_only_pca --split test --tasks structure,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/atac_only_pca --split test --tasks structure,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/early_fusion_pca --split test --tasks structure,collapse --labels label,label_fine
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/clip_mlp_d32_reg --split test --tasks structure,retrieval,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/evaluate.py --data data/pbmc_multiome --run outputs/generated_report_1_pbmc_multiome/vicreg_mlp_d32 --split test --tasks structure,retrieval,collapse,cross --labels label,label_fine --cross "rna->atac:binary_top10000,atac->rna:log1p_hvg2000"
python3 tools/compare_runs.py --out outputs/generated_report_1_pbmc_multiome --split test --reference rna_only_pca
python3 tools/compare_runs.py --out outputs/generated_report_1_pbmc_multiome --split test --reference early_fusion_pca
python3 tools/compare_runs.py --out outputs/generated_report_1_pbmc_multiome --split test --reference atac_only_pca
```

