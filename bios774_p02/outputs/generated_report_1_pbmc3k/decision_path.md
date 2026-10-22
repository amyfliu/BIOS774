| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Class balancing; generate only if it beats baselines | Rare types are very scarce; simple fixes must be beaten first |
| 2 | Data prep | Normalize, log-transform, keep 2000 top genes | Cells differ in depth, counts are skewed, genes outnumber cells |
| 3 | Model | Conditional VAE | Small data and tiny rare classes suit a VAE |
| 4 | Settings | Latent size 16, 150 epochs, 4 minutes | Small data trains fast; leaves budget for one retry |
| 5 | Training check | Training healthy | All latent dimensions used, loss flattened |
| 6 | Generate | Generate 200 cells each for two rarest types | Balancing target types: megakaryocytes and dendritic cells |
| 7 | Adjust | Resample around real rare cells | Prior samples were rarely recognised as the intended type |
| 8 | Evaluate | Compared with three simple baselines | Synthetic cells must beat simple fixes to be worth using |
| 9 | Recommend | Do not use synthetic cells; use class weights | No measurable gain over simple methods; test counts tiny |
