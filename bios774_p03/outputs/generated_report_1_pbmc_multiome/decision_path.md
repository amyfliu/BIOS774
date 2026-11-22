| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Joint cell typing plus cross-modal prediction | Objective names cell-type identification first, then predicting each modality from other |
| 2 | Data prep | Standard RNA and ATAC reductions | Counts are sparse; depth-normalised, compact inputs train faster and denoise |
| 3 | Combine modalities | CLIP-style alignment, VICReg as alternative | Cells are paired and prediction across modalities needs a shared space |
| 4 | Settings | Default settings sized to 7k cells | Small inputs train in minutes; defaults are standard for this size |
| 5 | Training check | CLIP overfits early; VICReg healthier | CLIP validation loss rose after epoch 11 while training loss kept falling |
| 6 | Evaluate | All learned models match RNA on coarse labels | Coarse cell types are easy; fine types separate the methods slightly |
| 7 | Adjust | Regularise CLIP (dropout, smaller network) | Original CLIP overfit and joint trailed early fusion on fine types |
| 8 | Encoder + loss | Choose regularised CLIP as main model | Best fine-type score and best cross-modal matching on validation |
| 9 | Downstream | Cell typing and cross-modal prediction run | These are the two analyses the objective asked for |
| 10 | Recommend | Use regularised joint CLIP; RNA PCA if cheap | Joint matches or beats each modality and adds cross-modal matching |
