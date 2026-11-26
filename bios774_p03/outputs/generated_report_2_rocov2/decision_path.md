| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Retrieval first, modality classification second | Objective names caption retrieval first; text words nearly give away the label |
| 2 | Data prep | Use BiomedCLIP features plus simple baselines | In-domain pretrained encoder matches radiology figures and captions |
| 3 | Combine modalities | Fine-tune BiomedCLIP with light adapters | Pretrained encoder already aligns both; adapters start from it and can't lose on val |
| 4 | Settings | Gentle adapter training; small projection-head alternative | Zero-shot already matches well; small steps avoid damaging it |
| 5 | Training check | Adapter never beat the starting model | Fine-tuning lowered matching on validation; try gentler step once |
| 6 | Adjust | Retry adapter once at smaller step | Adapter fell below start; skill allows one gentler retry |
| 7 | Training check | MLP heads match worse than zero-shot | Heads trained from scratch lose pretrained alignment |
| 8 | Evaluate | Zero-shot model wins on validation | Training did not improve matching; frozen model is strongest |
| 9 | Downstream | Retrieval and modality classification on test | Objective asks for both; measured on held-out test papers |
| 10 | Recommend | Use frozen BiomedCLIP for both tasks | Nothing we trained beat it; simplest option is best |
