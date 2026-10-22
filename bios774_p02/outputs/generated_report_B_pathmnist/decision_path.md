| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Class balancing; test whether it is needed | Classes are already nearly even, so synthetic images may add nothing |
| 2 | Data prep | Scale pixels to [-1, 1] | Standard input range for image diffusion with a convolutional network |
| 3 | Model | Conditional diffusion model, conv network | Enough images per class; diffusion gives sharpest samples |
| 4 | Settings | Small diffusion net, 12 minutes training | Fits the 15-minute budget; leaves time for sampling and testing |
| 5 | Training check | Training healthy but short | Loss fell steadily; time limit stopped it after 12 passes |
| 6 | Generate | Add 4,000 images to each of 3 smallest classes | Lifts the rarest classes toward the largest; keeps sampling time modest |
| 7 | Evaluate | Compare against simple baselines on a classifier | Synthetic data must beat class weights and oversampling to be worth using |
| 8 | Recommend | Do not use synthetic images | They hurt the classifier; classes are already nearly balanced |
