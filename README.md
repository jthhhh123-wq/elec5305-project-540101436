# 🎙️ Noise-Robust Keyword Spotting with ConvMixer

This project implements a **Keyword Spotting (KWS)** system based on **ConvMixer** for recognizing short speech commands under noisy environments.  
It includes a clean **baseline**, a **noise-augmented (AWGN)** experiment, and an improved **AWGN_v2** version.

## 📂 Folder Structure
noise_robust_kws_convmixer/
│
├── baseline/
│   ├── configs/
│   │   └── baseline.yaml
│   └── src/
│       ├── dataset_loader.py
│       ├── eval_noise_sweep.py
│       ├── model.py
│       ├── train.py
│       ├── utils.py
│       └── __init__.py
│
├── experiments/
│   └── awgn/
│       ├── configs/
│       │   ├── awgn_train.yaml
│       │   └── awgn_train_v2.yaml
│       └── src/
│           ├── dataset_loader.py
│           ├── eval_noise_sweep.py
│           ├── model.py
│           ├── train.py
│           ├── utils.py
│           └── __init__.py
│
├── runs/
│   ├── baseline_gpu_25ep/
│   │   └── baseline_best.pt
│   ├── awgn/
│   ├── awgn_v2/
│   ├── acc_snr.csv
│   └── acc_snr.png
│
└── data/
    └── SpeechCommands/

## ⚙️ Environment Setup
### 1. Create and activate conda environment
```bash
conda create -n kws python=3.10
conda activate kws
```
### 2. Install dependencies
```bash
pip install torch torchaudio matplotlib pyyaml
```
### 3. Prepare dataset
Download Google Speech Commands v0.02 and place it under:
```bash
project_root/data/SpeechCommands/
```
## 🚀 Run Baseline
Train the baseline model
```bash
cd baseline
python -m src.train --data_dir ../data --config ./configs/baseline.yaml --ckpt_dir ../runs/baseline_gpu_25ep
```
Evaluate robustness under noise
```bash
python -m src.eval_noise_sweep --data_dir ../data --config ./configs/baseline.yaml --ckpt ../runs/baseline_gpu_25ep/baseline_best.pt
```
Results will be automatically saved to:
```bash
runs/acc_snr.csv
```

## 🔊 Run AWGN Experiment
This version adds Additive White Gaussian Noise (AWGN) during training for noise robustness.
Train the AWGN model
```bash
cd experiments/awgn
python -m src.train --data_dir ../../data --config ./configs/awgn_train.yaml --ckpt_dir ../../runs/awgn
```
Train the AWGN model
```bash
cd baseline
python -m src.train --data_dir ../data --config ./configs/baseline.yaml --ckpt_dir ../runs/baseline_gpu_25ep
```
Evaluate the model
```bash
python -m src.eval_noise_sweep --data_dir ../../data --config ./configs/awgn_train.yaml --ckpt ../../runs/awgn/awgn_best.pt
```
Results will append to:
```bash
runs/acc_snr.csv
```

## 🧩 Run AWGN_v2 (Improved Version)
This version deepens the model and extends noise range for better low-SNR performance.
Train the improved model
```bash
cd experiments/awgn
python -m src.train --data_dir ../../data --config ./configs/awgn_train_v2.yaml --ckpt_dir ../../runs/awgn_v2
```
Evaluate the model
```bash
python -m src.eval_noise_sweep --data_dir ../../data --config ./configs/awgn_train_v2.yaml --ckpt ../../runs/awgn_v2/awgn_best.pt
```
Results will append to:
```bash
runs/acc_snr.csv
```

## 📈 Plot Accuracy vs SNR
After all experiments finish (baseline + awgn + awgn_v2),
you can visualize the comparison:
```bash
cd ../../
python experiments/plot_acc_snr.py --csv runs/acc_snr.csv --out runs/acc_snr.png --title "Baseline vs AWGN vs AWGN_v2"
```
The plot and CSV are saved to:
```bash
runs/acc_snr.csv
runs/acc_snr.png
```
