| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Joint cell-type ID, then cross-modal prediction | Objective names classification first, prediction second; labels stay a yardstick |
| 2 | Data prep | Standard RNA and ATAC reductions, plus targets | RNA is overdispersed counts; ATAC is near-binary, so each gets its usual treatment |
| 3 | Combine modalities | Compare aligned CLIP against simple PCA baselines | Prediction needs a shared space; baselines show whether learning helps |
| 4 | Settings | Default settings sized to 7k training cells | Inputs are small, so overfitting rather than time is the risk |
| 5 | Training check | CLIP overfits early; VICReg weaker matching | Validation loss rose after epoch 11 while training loss kept falling |
| 6 | Adjust | Retry CLIP with stronger regularisation | First CLIP run overfit quickly; more dropout may generalise better |
| 7 | Evaluate | Choose regularised CLIP on validation | It scored highest on fine cell types and matched cells best |
| 8 | Downstream | Cell-type probe and cross-modal prediction on test | These are the two tasks the objective named |
| 9 | Recommend | Use regularised CLIP joint embedding; multimodal helps | It beats each single modality on fine cell types, with CI above zero |
