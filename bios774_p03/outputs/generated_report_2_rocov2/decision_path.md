| # | Step | Decision | Why |
|--|------|------------------|-------------------------|
| 1 | Goal | Retrieval first, image modality classification second | Objective names caption retrieval then modality classification; labels only for scoring |
| 2 | Data prep | Use BiomedCLIP features plus simple baselines | Pretrained biomedical encoder matches the domain; simple features show what it buys |
| 3 | Combine modalities | Fine-tune BiomedCLIP lightly; keep baselines | Pretrained model already aligns image and text; small fix may help |
| 4 | Settings | Gentle settings, 8 + 5 minute budget | Start near a strong model; move slowly; keep minutes in reserve |
| 5 | Training check | Adapter never beat the pretrained start | Matching got worse though loss fell; original model was kept |
| 6 | Adjust | Retried adapter with 10x smaller step | First attempt never beat the pretrained start; skill allows one retry |
| 7 | Evaluate | Pretrained model wins on val | Fine-tuning did not help; zero-shot is already very strong |
| 8 | Downstream | Retrieval and image classification scored on test | Objective asks for both; zero-shot is the strongest model |
| 9 | Recommend | Use frozen BiomedCLIP as is | Training added nothing; the pretrained model is already strongest |
