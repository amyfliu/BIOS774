Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/rocov2 --out outputs/generated_report_2_rocov2
python3 tools/preprocess.py --data data/rocov2 --out outputs/generated_report_2_rocov2 --modality image --spec biomedclip
python3 tools/preprocess.py --data data/rocov2 --out outputs/generated_report_2_rocov2 --modality text --spec biomedclip
python3 tools/preprocess.py --data data/rocov2 --out outputs/generated_report_2_rocov2 --modality image --spec pixels32_pca50
python3 tools/preprocess.py --data data/rocov2 --out outputs/generated_report_2_rocov2 --modality text --spec tfidf_svd128
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/image_only_pca --inputs image:pixels32_pca50 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/text_only_pca --inputs text:tfidf_svd128 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/early_fusion_pca --inputs image:pixels32_pca50,text:tfidf_svd128 --method pca_concat --dim 32 --max-minutes 1
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/biomedclip_zeroshot --inputs image:biomedclip,text:biomedclip --method identity --max-minutes 1
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/clip_adapter_d512 --inputs image:biomedclip,text:biomedclip --method clip --encoder adapter --dim 512 --batch-size 512 --lr 1e-4 --select-by val_recall --max-minutes 8
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/clip_adapter_d512_lr1e5 --inputs image:biomedclip,text:biomedclip --method clip --encoder adapter --dim 512 --batch-size 512 --lr 1e-5 --select-by val_recall --max-minutes 3
python3 tools/train.py --data data/rocov2 --out outputs/generated_report_2_rocov2/clip_mlp_d256 --inputs image:biomedclip,text:biomedclip --method clip --encoder mlp --dim 256 --hidden 256 --dropout 0.2 --batch-size 512 --lr 1e-3 --select-by val_recall --max-minutes 5
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/biomedclip_zeroshot --split val --tasks structure,retrieval,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/clip_mlp_d256 --split val --tasks structure,retrieval,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/clip_adapter_d512 --split val --tasks structure,retrieval,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/image_only_pca --split val --tasks structure,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/text_only_pca --split val --tasks structure,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/early_fusion_pca --split val --tasks structure,collapse
python3 tools/compare_runs.py --out outputs/generated_report_2_rocov2 --split val
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/biomedclip_zeroshot --split test --tasks structure,retrieval,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/clip_mlp_d256 --split test --tasks structure,retrieval,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/image_only_pca --split test --tasks structure,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/text_only_pca --split test --tasks structure,collapse
python3 tools/evaluate.py --data data/rocov2 --run outputs/generated_report_2_rocov2/early_fusion_pca --split test --tasks structure,collapse
python3 tools/compare_runs.py --out outputs/generated_report_2_rocov2 --split test --reference biomedclip_zeroshot
python3 tools/compare_runs.py --out outputs/generated_report_2_rocov2 --split test --reference image_only_pca
```
