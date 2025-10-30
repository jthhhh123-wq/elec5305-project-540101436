# 🗣️ Keyword Spotting in Noisy Environments

**Unit:** ELEC5305 – Sound Synthesis Deep Dive  
**Student:** Jianing Zhang (SID 540101436)  
**Supervisor:** Dr Craig Jin  
**Stage:** Project Feedback 2 (Work in Progress)

---

## 1️⃣ Introduction

This project explores a **Keyword Spotting (KWS)** system that can recognize short speech commands under **noisy environments** using a **Convolutional Neural Network (CNN)**. The purpose is to examine how noise affects recognition accuracy and to design a lightweight, noise-robust baseline for future improvements.

This submission (Feedback 2) focuses on:

1. Building and training a baseline CNN using the Google Speech Commands v0.02 dataset.  
2. Evaluating its **robustness to Gaussian noise** at multiple Signal-to-Noise Ratios (SNRs).  
3. Generating quantitative and visual results for preliminary analysis.

The next phase will include **SpecAugment**, **front-end denoising**, and **attention-based CNN architectures**.

---

## 2️⃣ Literature Background

Prior work showed that CNN-based small-footprint keyword spotting is practical for always-on devices such as smart speakers and mobile phones (Sainath et al., 2015; Warden, 2018). However, later studies also reported that **noise and channel mismatch** can quickly degrade recognition accuracy, especially for short commands (Li et al., 2022).

Typical directions to improve robustness include:

- **Data augmentation** with synthetic / real noise (Park et al., 2019);  
- **Feature normalization / denoising front-ends** to make Mel features more stable (Reddy et al., 2021);  
- **Compact CNN / CRNN architectures** that keep latency low for edge devices.

This project first reproduces a clean **baseline CNN** so that later improvements can be compared against it.

---

## 3️⃣ Research Question

> **How can a compact CNN model maintain reliable keyword recognition under noisy acoustic conditions?**

So in Feedback 2 the goal is **not** to beat SOTA, but to:
- measure how fast accuracy drops when SNR decreases,
- visualise which commands get confused,
- and prepare a place in the repo where later augmentation results can be added.

---

## 4️⃣ Dataset and Feature Extraction

| Item | Description |
|------|-------------|
| **Dataset** | Google Speech Commands v0.02 (Warden, 2018) |
| **Selected classes (10)** | `yes, no, stop, go, up, down, left, right, on, off` |
| **Sample rate** | 16 kHz |
| **Feature** | 40-bin **log-Mel spectrogram** |
| **Window / hop** | 25 ms window, 10 ms hop |
| **Script** | `code/features.py` |

All `.wav` files are loaded, (re)sampled to 16 kHz, converted to log-Mel spectrograms and padded to the longest frame in the batch.

---

## 5️⃣ Repository Structure

```text
.
├── code/
│   ├── train_baseline.py      # training loop, dataset split, save best .pt
│   ├── evaluate_noise.py      # test under multiple SNRs, draw confusion matrices
│   ├── model.py               # KWSCNN: 2 conv blocks + FC
│   ├── features.py            # log-Mel extraction
│   └── utils.py               # set_seed, helpers
├── results/
│   ├── accuracy_vs_snr.png
│   ├── confusion_30dB.png
│   ├── confusion_20dB.png
│   ├── confusion_10dB.png
│   ├── confusion_0dB.png
│   └── confusion_-5dB.png
├── requirements.txt
├── Feedback2.pdf              # current report
└── README.md
