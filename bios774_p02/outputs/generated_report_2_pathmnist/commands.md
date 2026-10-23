Tool calls in the order the agent ran them (smoke tests and --help omitted):

```bash
python3 tools/inspect_data.py --data data/pathmnist --out outputs/generated_report_2_pathmnist
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_2_pathmnist/probe --model ddpm --backbone conv --prep scale_pm1 --conditional --epochs 1 --batch-size 128 --lr 2e-4 --params '{"ch":48,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-train 20000 --max-minutes 1
python3 tools/train.py --data data/pathmnist --out outputs/generated_report_2_pathmnist/ddpm_ch48 --model ddpm --backbone conv --prep scale_pm1 --conditional --epochs 100 --batch-size 128 --lr 2e-4 --params '{"ch":48,"T":1000,"schedule":"cosine","p_uncond":0.1,"ema_decay":0.999}' --max-minutes 34 > outputs/generated_report_2_pathmnist/train_stdout.txt
pgrep -f tools/train.py
until ! pgrep -f tools/train.py
until ! pgrep -f tools/train.py
until ! pgrep -f tools/train.py
until ! pgrep -f tools/train.py
until ! pgrep -f tools/train.py
until ! pgrep -f tools/train.py
pgrep -f tools/train.py
python3 tools/sample.py --run outputs/generated_report_2_pathmnist/ddpm_ch48 --n-per-class 20 --steps 50 --guidance 1.5 --tag probe
python3 tools/sample.py --run outputs/generated_report_2_pathmnist/ddpm_ch48 --n-per-class 1000 --steps 50 --guidance 1.5 --tag rel
python3 tools/evaluate.py --run outputs/generated_report_2_pathmnist/ddpm_ch48 --tasks fidelity,memorization,utility --utility tstr --samples-tag rel
python3 tools/sample.py --run outputs/generated_report_2_pathmnist/ddpm_ch48 --n-per-class 1000 --steps 50 --guidance 2.5 --tag g25
python3 tools/evaluate.py --run outputs/generated_report_2_pathmnist/ddpm_ch48 --tasks fidelity,memorization,utility --utility tstr --samples-tag g25
```
