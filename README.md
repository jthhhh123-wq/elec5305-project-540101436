# 🎙️ Noise-Robust Keyword Spotting with ConvMixer

This project implements a **Keyword Spotting (KWS)** system based on **ConvMixer** for recognizing short speech commands under noisy environments.  
It includes a clean **baseline**, a **noise-augmented (AWGN)** experiment, and an improved **AWGN_v2** version.

## 📂 Folder Structure
```bash
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
```

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

### 3. Hardware environment
The experiments were run on:
```bash
GPU: NVIDIA RTX 4060 (8GB)
Framework: PyTorch (CUDA enabled)
```
(If CUDA is unavailable, the scripts will automatically fall back to CPU, but training will be significantly slower.)

### 4. Prepare dataset
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

### 📈 Plot Accuracy vs SNR
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

## 📘 Curriculum Learning Training
This experiment trains the model with multi-stage SNR curriculum learning, where the model is trained from high SNR (easy) to low SNR (hard).
All settings are defined in:

experiments/curriculum/configs/curriculum_train.yaml

###🚀 1. Training (Curriculum Learning)

Enter the curriculum folder：

```bash
cd experiments/curriculum
```

Run the curriculum-learning training:

```bash
python -m src.train \
    --data_dir ../../data \
    --config ./configs/curriculum_train.yaml \
    --ckpt_dir ../../runs/curriculum
```

This will:

Train sequentially on SNR stages (e.g., 20 → 10 → 0 → –5 dB)

Save checkpoints to runs/curriculum/

Save the best model as

```bash
runs/curriculum/<tag>_best.pt
```

### 📊 2. Noise Robustness Evaluation

After training, evaluate the model under different SNR values.

Run from the curriculum experiment folder:

```bash
python -m src.eval_noise_sweep --data_dir ../../data --config ./configs/curriculum_train.yaml --ckpt ../../runs/curriculum/curriculum_best.pt
```
This script will:

Test the trained model at SNR = 20, 10, 0, –5 dB

Append results to:

```bash
runs/acc_snr.csv
```

### 📈 3. Plotting the Accuracy–SNR Curve

Use the unified plotting script:

```bash
python experiments/plot_acc_snr.py --csv runs/acc_snr.csv --out runs/acc_snr.png --title "Baseline vs AWGN vs Curriculum"
```

This generates:

```bash
runs/acc_snr.png
```

The plot will contain all experiments whose results are recorded in the CSV, such as:

baseline

awgn_train

awgn_train_v2

curriculum_train

### ✔ Expected Result Trend

Curriculum learning typically provides:

Slightly worse performance at very high noise (–5 dB) compared to AWGN v2

Clear improvement at moderate/high SNR (10–20 dB)

Smooth and stable SNR–Accuracy curve

```bash
python -m src.eval_noise_sweep --data_dir ../../data --config ./configs/se_reverb.yaml --ckpt ../../runs/se_reverb/se_reverb_best.pt
```
