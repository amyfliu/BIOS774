Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pathmnist --out outputs/generated_report_B_pathmnist
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --model ddpm --backbone conv --prep scale_pm1 --conditional --epochs 100 --batch-size 128 --lr 2e-4 --params '{"ch":32,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-minutes 12
ps aux
while pgrep -f "tools/train.py"
python3 tools/sample.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --class-counts '{"normal colon mucosa": 100}' --steps 50 --guidance 2.0 --tag probe
python3 tools/sample.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --class-counts '{"normal colon mucosa": 4000, "mucus": 4000, "adipose": 4000}' --steps 50 --guidance 2.0 --tag balance
python3 tools/evaluate.py --run outputs/generated_report_B_pathmnist/ddpm_conv_ch32 --tasks fidelity,memorization,utility --utility augment --samples-tag balance
```
