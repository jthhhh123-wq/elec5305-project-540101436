---

## 7️⃣ Results and Discussion  

After 20 epochs of training, the baseline CNN achieved good performance under clean audio conditions but showed a clear drop in accuracy as noise increased.

### 🔹 Quantitative Results

| **SNR (dB)** | **Accuracy (%)** |
|--------------:|----------------:|
| 30 | 68.5 |
| 20 | 63.4 |
| 10 | 56.7 |
| 0  | 46.2 |
| −5 | 37.8 |

**Observation:**  
The model performs well in clean or mild noise (≥20 dB) but rapidly loses accuracy below 10 dB, indicating limited robustness to strong background interference.

---

### 🔹 Figure 1 – Accuracy vs. SNR  
Shows overall performance trend across different noise levels.

<img src="results/accuracy_vs_snr.png" width="500"/>

---

### 🔹 Figures 2–6 – Confusion Matrices under Different Noise Levels  

These matrices illustrate how misclassification increases under noise, especially between acoustically similar commands (“go/no” and “on/off”).

**30 dB (clean)**  
<img src="results/confusion_30dB.png" width="300"/>

**20 dB**  
<img src="results/confusion_20dB.png" width="300"/>

**10 dB**  
<img src="results/confusion_10dB.png" width="300"/>

**0 dB**  
<img src="results/confusion_0dB.png" width="300"/>

**−5 dB (high noise)**  
<img src="results/confusion_-5dB.png" width="300"/>

---

### 🔹 Discussion  

These results align with previous studies showing that **noise robustness remains a key challenge** in small-footprint KWS systems (Li et al., 2022).  
At low SNRs, overlapping phonetic cues and spectral masking cause the CNN to confuse similar temporal patterns.  

Future improvements will include:  
- **SpecAugment** and additive noise augmentation (Park et al., 2019).  
- **Denoising or feature normalization front-ends** (Reddy et al., 2021).  
- **CNN-LSTM or attention-based architectures** for better temporal modeling.  
- Real-world environmental noise evaluation.

---

## 8️⃣ References  

- Li, J., Deng, L., & Gong, Y. (2022). *Noise-Robust Automatic Speech Recognition: A Review.* IEEE/ACM T-ASLP, 30, 1532–1550.  
- Park, D. S. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech 2019.  
- Reddy, C. K. A. et al. (2021). *DNS Challenge: Improving Noise Suppression Models.* Interspeech 2021.  
- Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209.  

---

## 🧠 Notes  

This report represents **the current development stage (Feedback 2)** of the KWS project.  
The final submission will include expanded data augmentation, model optimization, and real environmental noise testing.

---
