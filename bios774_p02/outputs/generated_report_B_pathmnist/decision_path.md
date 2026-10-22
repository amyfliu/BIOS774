| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Balancing task; imbalance is mild, likely decline | Largest class is only 1.6 times the smallest; simple fixes probably suffice |
| 2 | Data prep | Scale pixels to [-1, 1], keep image shape | Standard input range for image diffusion models; keeps spatial structure |
| 3 | Model | Conditional diffusion model with conv network | Plenty of image data and we must choose which tissue to generate |
| 4 | Settings | Small diffusion model, 12 minutes training | Probe speed is ~11 steps/s; this fits roughly 8-10k updates in budget |
| 5 | Training check | Training looked healthy | Loss fell smoothly and noise prediction improved at every noise level |
| 6 | Generate | Top up 8 classes to the largest class | Balancing means raising smaller classes to the size of the biggest |
| 7 | Evaluate | Check realism, copying and classifier benefit | Balancing is judged by classifier gain over simple baselines, plus realism |
| 8 | Recommend | Use with restrictions; baselines first | Small classifier gain, but images are visibly imperfect and imbalance is mild |
