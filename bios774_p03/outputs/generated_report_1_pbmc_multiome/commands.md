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
