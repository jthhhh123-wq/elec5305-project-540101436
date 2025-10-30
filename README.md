# Keyword Spotting in Noisy Environments  

## Project Overview  
This repository documents the **current progress** of a keyword spotting (KWS) system developed for *ELEC5305 – Sound Synthesis Deep Dive* at the University of Sydney.  
The project aims to design, train, and evaluate a **Convolutional Neural Network (CNN)** that can recognize short speech commands from the *Google Speech Commands v0.02* dataset (Warden, 2018).  

This stage (Feedback 2 submission) focuses on:  
1. Reviewing state-of-the-art literature and defining the **research question**.  
2. Building and training a **baseline CNN** for KWS.  
3. Evaluating its **noise robustness** under Gaussian noise at multiple SNR levels.  
4. Providing example input/output cases for demonstration.  

Further development (for the **final submission**) will extend to **data augmentation**, **improved architectures**, and **real-world environmental noise testing**.  

---

## 1. Research Question  
> How can a compact CNN model maintain reliable keyword recognition performance in noisy acoustic environments?

This question explores the trade-off between model simplicity and robustness — a core challenge for real-time, on-device speech recognition.  

---

## 2. Background and Literature Review  
Keyword spotting systems enable low-power, always-on interfaces such as *Hey Siri* and *OK Google*.  
While CNN-based models have achieved strong accuracy in clean conditions, they often fail under noise or far-field settings (Li et al., 2022).  

Recent works addressing these challenges include:  
- **SpecAugment** for spectrogram augmentation to enhance noise robustness (Park et al., 2019).  
- **Denoising and normalization front-ends** to improve SNR tolerance (Reddy et al., 2021).  
- **Compact CNN and hybrid CNN-LSTM** models for on-device deployment (Sainath et al., 2015).  
- **KWS-20 baseline** providing a structured benchmark for small-footprint KWS (Meneses et al., 2020).  

This project builds upon these approaches by evaluating a CNN baseline’s degradation pattern under controlled **Gaussian noise**,  
serving as a foundation for later augmentation and robustness improvements.

---

## 3. Baseline Model and Dataset  

**Dataset:**  
- [Google Speech Commands v0.02](https://arxiv.org/abs/1804.03209) (Warden, 2018)  
- 10 selected command classes: `yes, no, stop, go, up, down, left, right, on, off`  

**Baseline Reference:**  
- Structure inspired by the [KWS-20 project](https://michel-meneses.github.io/sidi-kws/#method).  
- Adapted with simplified 2-layer CNN and log-Mel feature extraction pipeline.  

**Model Architecture:**  
- 2 convolutional layers + batch normalization  
- 1 fully connected classifier  
- Trained using Adam optimizer (lr = 1e-3, batch size = 128, epochs = 20)

---

## 4. My Contribution  

**Implemented:**
- `evaluate_noise.py` — custom evaluation script to test performance under multiple SNRs (30 → −5 dB).  
- Automated generation of:
  - `accuracy_vs_snr.png`
  - `confusion_*.png` for each noise level.  
- Enhanced `train_baseline.py` with:
  - Learning rate, batch size, and epoch configurability.  
  - Checkpoint saving for best-performing models.  
- Added reproducible **training + testing commands** in README.  
- Prepared **audio examples** for future GitHub demonstration.

---

## 5. Repository Structure  

| File | Description |
|------|--------------|
| `features.py` | Extracts **log-Mel spectrograms** from .wav inputs. |
| `model.py` | Defines the **KWSCNN** baseline architecture. |
| `train_baseline.py` | Trains the CNN model on selected commands. |
| `evaluate_noise.py` | Evaluates trained model under Gaussian noise. |
| `results_feedback2/` | Contains output plots (accuracy + confusion matrices). |
| `speech_commands_v0.02/` | Dataset directory. |
| `requirements.txt` | Lists dependencies (torch, librosa, numpy, matplotlib, seaborn). |

---

## 6. How to Run  

### Step 1️⃣ – Environment Setup
```bash
pip install -r requirements.txt
