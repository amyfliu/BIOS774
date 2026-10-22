Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pbmc3k --out outputs/generated_report_1_pbmc3k
python3 tools/train.py --data data/pbmc3k --out outputs/generated_report_1_pbmc3k/cvae_hvg2000_z16 --model vae --backbone mlp --prep libnorm,log1p,hvg:2000,standardize --conditional --epochs 150 --batch-size 64 --lr 1e-3 --params '{"latent_dim":16,"hidden":[512,256],"beta":1.0,"kl_warmup_epochs":10}' --max-minutes 4 --device mps
python3 tools/sample.py --run outputs/generated_report_1_pbmc3k/cvae_hvg2000_z16 --class-counts '{"Megakaryocytes":200,"Dendritic cells":200}' --tag prior
python3 tools/evaluate.py --run outputs/generated_report_1_pbmc3k/cvae_hvg2000_z16 --tasks fidelity,memorization,utility --utility augment --samples-tag prior
python3 tools/log_decision.py --out outputs/generated_report_1_pbmc3k --step sampling --short "Generate 200 cells each for two rarest types" --why "Balancing target types: megakaryocytes and dendritic cells" --decision "sample.py --class-counts Mega 200, DC 200, prior latent, tag prior" --evidence "400 samples
python3 tools/sample.py --run outputs/generated_report_1_pbmc3k/cvae_hvg2000_z16 --class-counts '{"Megakaryocytes":200,"Dendritic cells":200}' --latent-source posterior --temperature 1.5 --tag post
python3 tools/evaluate.py --run outputs/generated_report_1_pbmc3k/cvae_hvg2000_z16 --tasks fidelity,memorization,utility --utility augment --samples-tag post
```
