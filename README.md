# 🗣️ Keyword Spotting in Noisy Environments

**Unit:** ELEC5305 – Sound Synthesis Deep Dive  
**Student:** Jianing Zhang (SID 540101436)  
**Supervisor:** Dr Craig Jin  
**Stage:** Project Feedback 2 (Work in Progress)

---

## 1️⃣ Introduction

This project explores a **Keyword Spotting (KWS)** system that can recognize short speech commands under **noisy environments** using a **Convolutional Neural Network (CNN)**.  
The purpose is to examine how noise affects recognition accuracy and to design a lightweight, noise-robust baseline for future improvements.

This submission (Feedback 2) focuses on:
1. Building and training a baseline CNN using the Google Speech Commands v0.02 dataset.  
2. Evaluating its **robustness to Gaussian noise** at multiple Signal-to-Noise Ratios (SNRs).  
3. Generating quantitative and visual results for preliminary analysis.

The next phase will include **SpecAugment**, **front-end denoising**, and **attention-based CNN architectures**.

---

## 2️⃣ Literature Background

Prior work (Sainath et al., 2015; Warden, 2018) established CNN-based KWS as an efficient solution for on-device voice activation like “Hey Siri” or “OK Google.”  
However, background noise remains a critical issue — even small distortions can reduce word accuracy dramatically (Li et al., 2022).  
Recent research improves robustness through:
- **Data augmentation:** adding synthetic or real noise (Park et al., 2019);  
- **Feature normalization:** improving signal-to-feature mapping stability (Reddy et al., 2021);  
- **Compact models:** balancing accuracy and low power consumption for embedded hardware.

This project replicates a simple CNN baseline as a foundation to later test these robustness techniques.

---

## 3️⃣ Research Question

> **How can a compact CNN model maintain reliable keyword recognition under noisy acoustic conditions?**

This question guides the work: measure degradation, visualize confusion, and prepare for later augmentation strategies.

---

## 4️⃣ Dataset and Feature Extraction

| Item | Description |
|------|--------------|
| **Dataset** | Google Speech Commands v0.02 (Warden, 2018) |
| **Classes (10)** | yes, no, stop, go, up, down, left, right, on, off |
| **Sample Rate** | 16 kHz |
| **Feature Type** | 40-bin log-Mel spectrograms |
| **Preprocessing** | 25 ms window, 10 ms hop, normalized amplitude |

Feature extraction is handled by `code/features.py`.

---

## 5️⃣ Repository Structure

```text
.
├── code/
│   ├── train_baseline.py      # model training
│   ├── evaluate_noise.py      # add Gaussian noise and evaluate
│   ├── model.py               # defines KWSCNN architecture
│   ├── features.py            # extract log-Mel features
│   └── utils.py               # helper functions
├── results/
│   ├── accuracy_vs_snr.png
│   ├── confusion_30dB.png
│   ├── confusion_20dB.png
│   ├── confusion_10dB.png
│   ├── confusion_0dB.png
│   └── confusion_-5dB.png
├── requirements.txt
├── Feedback2.pdf
└── README.md
"""
============================================================
 Keyword Spotting in Noisy Environments
 ELEC5305 – Sound Synthesis Deep Dive
 Author: Jianing Zhang (SID 540101436)
 Supervisor: Dr Craig Jin
 Project Stage: Feedback 2
============================================================

This file summarises the key components of the current project:
- Baseline CNN model architecture
- Training and evaluation workflow
- Discussion and conclusion
- References for Feedback 2 submission
============================================================
"""

# ----------------------------------------------------------
# 🧠 BASELINE CNN MODEL
# ----------------------------------------------------------

"""
The baseline model (KWSCNN) is a compact 2-layer CNN used for 
Keyword Spotting (KWS) tasks.

Architecture Summary:
------------------------------------------------------------
| Layer | Type                | Output Shape   | Activation |
|-------|---------------------|----------------|-------------|
| 1     | Conv2D + BatchNorm  | (16, 20, 40)   | ReLU        |
| 2     | Conv2D + BatchNorm  | (32, 10, 20)   | ReLU        |
| 3     | Fully Connected      | (10,)          | Softmax     |
------------------------------------------------------------
Optimizer: Adam (lr = 1e-3)
Loss Function: CrossEntropyLoss
Batch Size: 128
Epochs: 20
"""


# ----------------------------------------------------------
# ⚙️ TRAINING AND EVALUATION WORKFLOW
# ----------------------------------------------------------

"""
Step 1️⃣ – Train the baseline model
-----------------------------------
python code/train_baseline.py \
  --data_root ./speech_commands_v0.02 \
  --classes yes no stop go up down left right on off \
  --epochs 20 --batch_size 128 --lr 1e-3 \
  --save_path models/baseline_cnn_final.pt

This trains a 2-layer CNN using 10 command classes and saves the
final checkpoint to "models/baseline_cnn_final.pt".


Step 2️⃣ – Evaluate noise robustness
-----------------------------------
python code/evaluate_noise.py \
  --data_root ./speech_commands_v0.02 \
  --model_path models/baseline_cnn_final.pt \
  --classes yes no stop go up down left right on off \
  --plot_confusion_all True \
  --results_dir results

This adds Gaussian white noise at multiple Signal-to-Noise Ratios:
SNR = [30, 20, 10, 0, -5] dB
and outputs:
  - accuracy_vs_snr.png
  - confusion_30dB.png
  - confusion_20dB.png
  - confusion_10dB.png
  - confusion_0dB.png
  - confusion_-5dB.png
"""


# ----------------------------------------------------------
# 📊 DISCUSSION
# ----------------------------------------------------------

"""
Observations:
-------------
- The model performs well at 30–20 dB (clean to mild noise)
  but accuracy decreases sharply below 10 dB.
- Misclassification increases between phonetically similar
  classes such as "go/no" and "on/off" under heavy noise.

Interpretation:
---------------
These results confirm findings from Li et al. (2022) that
noise robustness remains a challenge in small-footprint
KWS systems. The model's simplicity makes it efficient but
less adaptive to spectral masking and overlapping phonemes.

Planned improvements:
---------------------
1. Apply SpecAugment and noise-injection augmentation.
2. Add feature normalization or denoising front-ends.
3. Explore CNN-LSTM or attention-based temporal models.
4. Extend evaluation to real environmental noise recordings.
"""


# ----------------------------------------------------------
# ✅ CONCLUSION
# ----------------------------------------------------------

"""
This experiment establishes a functional KWS baseline model
and provides clear benchmarks for noise-robustness analysis.

- Accuracy (clean): 68.5%
- Accuracy (−5 dB): 37.8%

The sharp decline confirms the need for augmentation and
architecture enhancements in the next stage.

In the final project, additional experiments will include:
- SpecAugment and DNS noise datasets
- Feature-level denoising
- Attention-based CNN for temporal stability
"""


# ----------------------------------------------------------
# 📚 REFERENCES
# ----------------------------------------------------------

"""
1. Warden, P. (2018).
   "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition."
   arXiv:1804.03209.

2. Li, J., Deng, L., & Gong, Y. (2022).
   "Noise-Robust Automatic Speech Recognition: A Review."
   IEEE/ACM Transactions on Audio, Speech, and Language Processing.

3. Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019).
   "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition."
   Interspeech.

4. Reddy, C. K. A. et al. (2021).
   "DNS Challenge: Improving Noise Suppression Models."
   Interspeech.
"""

# ----------------------------------------------------------
# END OF FILE
# ----------------------------------------------------------
