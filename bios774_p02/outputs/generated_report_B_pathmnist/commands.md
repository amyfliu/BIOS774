Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pathmnist --out outputs/generated_report_B_pathmnist
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_B_pathmnist/ddpm_probe --model ddpm --backbone conv --prep scale_pm1 --conditional --max-train 20000 --epochs 1 --batch-size 128 --lr 2e-4 --params '{"ch":32,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-minutes 1
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --model ddpm --backbone conv --prep scale_pm1 --conditional --max-train 40000 --epochs 200 --batch-size 64 --lr 2e-4 --params '{"ch":32,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-minutes 12
python3 tools/sample.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --class-counts '{"0":782,"1":750,"2":561,"3":552,"4":1084,"5":156,"6":1111,"7":774,"8":0}' --steps 50 --guidance 1.5 --tag balance
python3 tools/evaluate.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --tasks fidelity,memorization,utility --utility augment --samples-tag balance
```
