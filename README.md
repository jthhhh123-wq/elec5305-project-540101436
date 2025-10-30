# 🗣️ Keyword Spotting in Noisy Environments  

### ELEC5305 – Sound Synthesis Deep Dive  
**Author:** Jianing Zhang (SID 540101436)  
**Supervisor:** Dr. Craig Jin  
**Institution:** University of Sydney  
**Project Stage:** Feedback 2 Progress Submission  

---

## 1️⃣ Overview  

This repository presents the **current progress** of a keyword spotting (KWS) project that investigates how **Convolutional Neural Networks (CNNs)** perform in noisy environments.  
The goal is to recognize short speech commands using the *Google Speech Commands v0.02* dataset (Warden, 2018),  
and evaluate performance under different **Signal-to-Noise Ratios (SNRs)**.  

This report represents the **Feedback 2 stage** — focusing on building a baseline, analyzing results, and defining next-phase improvements.  

---

## 2️⃣ Research Question & Motivation  

> **Research Question:**  
> How can a small-footprint CNN model maintain robust keyword recognition performance under noisy acoustic conditions?

With the growth of **voice-controlled IoT systems** and **embedded assistants**,  
achieving reliable recognition in real-world noise is a key challenge.  
Compact CNN architectures are attractive due to their low computation cost,  
but they suffer from poor noise generalization (Li et al., 2022).  

This project aims to benchmark a baseline CNN’s robustness to Gaussian noise  
and explore enhancement techniques to improve low-SNR performance.  

---

## 3️⃣ Baseline Setup  

### Dataset  
- **Source:** [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209)  
- **Selected Classes:** `yes`, `no`, `stop`, `go`, `up`, `down`, `left`, `right`, `on`, `off`  
- **Sampling Rate:** 16 kHz  

### Baseline Reference  
- Architecture adapted from [KWS-20 Benchmark](https://michel-meneses.github.io/sidi-kws/#method).  
- Simplified 2-layer CNN model trained on log-Mel spectrogram features.  

---

## 4️⃣ Data Preprocessing  

| Step | Description |
|------|--------------|
| **1. Resampling** | Audio files resampled to 16 kHz and amplitude-normalized. |
| **2. Framing** | 25 ms Hamming windows, 10 ms hop size. |
| **3. Feature Extraction** | 40-band **log-Mel spectrograms** computed via `librosa`. |
| **4. Normalization** | Each sample standardized before CNN input. |

Resulting tensors: `(1, n_mels, T)`  
These features are passed directly into the KWSCNN architecture.

---

## 5️⃣ Model Architecture  

| Layer | Type | Output Shape | Activation |
|--------|------|---------------|-------------|
| 1 | Conv2D + BatchNorm | (16, 20, 40) | ReLU |
| 2 | Conv2D + BatchNorm | (32, 10, 20) | ReLU |
| 3 | Fully Connected | (10,) | Softmax |

**Optimizer:** Adam (lr=1e-3)  
**Loss:** Cross-Entropy  
**Epochs:** 20  
**Batch Size:** 128  

---

## 6️⃣ Training & Evaluation  

### Training Command
```bash
python train_baseline.py \
  --data_root ./speech_commands_v0.02 \
  --classes yes no stop go up down left right on off \
  --epochs 20 --batch_size 128 --lr 1e-3 \
  --save_path models/baseline_cnn_final.pt

