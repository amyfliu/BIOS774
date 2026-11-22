| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Classify cell types; predict each modality from other | Objective names joint cell-type identification and cross-modal prediction; no label training mentioned |
| 2 | Data prep | Standard reductions per modality, plus prediction targets | RNA is overdispersed counts; ATAC is near-binary and sparse, so each gets its usual treatment |
| 3 | Combine modalities | Contrastive alignment vs early-fusion baselines | Paired cells and cross-prediction goal need aligned spaces; baselines show if learning helps |
| 4 | Settings | Mid-size networks, 32 dims, 8 min each | Small data, fast epochs; budget split between main and alternative models |
| 5 | Training check | CLIP overfits early; VICReg weaker matching | Validation loss rose after epoch 11 while training loss kept falling |
| 6 | Adjust | Retrain CLIP with stronger regularisation | First CLIP run overfit quickly |
| 7 | Evaluate | Regularised CLIP best learned model on val | Beat both single modalities and early fusion on fine cell types; retrieval far above chance |
| 8 | Downstream | Cell-type probe and cross-modality prediction | These are the two analyses the objective asked for |
| 9 | Recommend | Use aligned CLIP embedding for cell types | It beats each modality alone on fine cell types; early fusion is a close, simpler rival |
