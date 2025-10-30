# 🗣️ Keyword Spotting in Noisy Environments  

**ELEC5305 – Sound Synthesis Deep Dive**  
**Author:** Jianing Zhang (SID 540101436)  
**Supervisor:** Dr. Craig Jin  
**Institution:** University of Sydney  
**Project Stage:** Feedback 2 Progress Report  

---

## 1️⃣ Project Overview  

This repository presents the current progress of a **Keyword Spotting (KWS)** system designed to recognize short speech commands using a **Convolutional Neural Network (CNN)**.  
The study evaluates how well a compact CNN can maintain accuracy under different **noise conditions**, using the *Google Speech Commands v0.02* dataset (Warden, 2018).  

This version (Feedback 2) focuses on:  
1. Building a baseline CNN architecture for KWS.  
2. Testing noise robustness under Gaussian noise at multiple Signal-to-Noise Ratios (SNRs).  
3. Analyzing performance degradation using quantitative and visual metrics.  

The next stage will introduce **data augmentation**, **attention-based architectures**, and **real-world noise testing**.

---

## 2️⃣ Research Question  

> **How can a small-footprint CNN model maintain reliable keyword recognition in noisy acoustic environments?**

Voice-controlled devices require efficient and robust models.  
While CNNs perform well on clean data, their accuracy drops significantly under background noise.  
This project benchmarks baseline CNN performance under Gaussian noise and sets up a framework for future robustness improvements.

---

## 3️⃣ Dataset and Baseline Setup  

| Item | Description |
|------|--------------|
| **Dataset** | [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209) |
| **Classes** | yes, no, stop, go, up, down, left, right, on, off |
| **Sampling Rate** | 16 kHz |
| **Features** | Log-Mel spectrograms (40 Mel filters, 25 ms window, 10 ms hop) |
| **Baseline Reference** | Architecture adapted from [KWS-20 Benchmark](https://michel-meneses.github.io/sidi-kws/#method) |

---

## 4️⃣ Preprocessing  

All `.wav` files are converted into log-Mel spectrograms for CNN input.  
This step is automatically handled by `features.py`.

| Step | Operation |
|------|------------|
| 1 | Resample to 16 kHz |
| 2 | Apply 25 ms Hamming window, 10 ms hop |
| 3 | Compute 40-band log-Mel spectrogram |
| 4 | Normalize amplitude per sample |

---

## 5️⃣ Model Architecture  

| Layer | Type | Output | Activation |
|--------|------|---------|-------------|
| 1 | Conv2D + BatchNorm | (16, 20, 40) | ReLU |
| 2 | Conv2D + BatchNorm | (32, 10, 20) | ReLU |
| 3 | Fully Connected | (10,) | Softmax |

**Optimizer:** Adam (lr = 1e-3)  
**Loss:** Cross-Entropy  
**Batch Size:** 128  
**Epochs:** 20  

---

## 6️⃣ Training and Evaluation  

### Training Command  
```bash
python train_baseline.py \
  --data_root ./speech_commands_v0.02 \
  --classes yes no stop go up down left right on off \
  --epochs 20 --batch_size 128 --lr 1e-3 \
  --save_path models/baseline_cnn_final.pt

