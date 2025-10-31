# 🗣️ Keyword Spotting in Noisy Environments

**ELEC5305 – Sound Synthesis Deep Dive**  
**Author:** Jianing Zhang (SID: 540101436)  
**Supervisor:** Dr. Craig Jin | **TA:** Reza Ghanavi  
**University of Sydney**

---

## 🎯 Project Overview

This repository presents the **Feedback 2 stage** of a keyword spotting (KWS) system designed to recognize short speech commands in **noisy environments** using a **Convolutional Neural Network (CNN)**.  

The system is trained and evaluated on the **Google Speech Commands v0.02** dataset (Warden, 2018), targeting **10 command words** (*yes, no, stop, go, up, down, left, right, on, off*).  

### Current Stage (Feedback 2)
1. Implement and train a **baseline CNN model**.  
2. Evaluate **noise robustness** using additive Gaussian noise (SNR = 30 → −5 dB).  
3. Analyze performance degradation with visual and quantitative metrics.  

### Final Stage (Planned)
- Integrate **data augmentation (SpecAugment)**.  
- Add **attention-based or GRU-enhanced architectures**.  
- Evaluate on **real environmental noise** datasets.

---

## 🧩 Repository Structure

| File | Description |
|------|--------------|
| `features.py` | Extracts **log-Mel spectrograms** for CNN input. |
| `model.py` | Defines the **KWSCNN baseline** (2 conv + FC layers). |
| `train_baseline.py` | Handles training, validation, and checkpoint saving. |
| `evaluate_noise.py` | Adds Gaussian noise, tests across SNRs, and plots results. |
| `models/` | Trained weights (e.g., `baseline_cnn_final.pt`). |
| `results_feedback2/` | Accuracy vs. SNR plots and confusion matrices. |
| `speech_commands_v0.02/` | Dataset directory with 10 command classes. |

---

## ⚙️ Workflow

### Step 1️⃣ – Feature Extraction
Each `.wav` file (16 kHz, mono) is converted into **log-Mel spectrograms**  
(25 ms window, 10 ms hop, 40 Mel filters):

```bash
python features.py --data_root ./speech_commands_v0.02
```

### Step 2️⃣ – Model Training
Train the baseline CNN:

```bash
python train_baseline.py \
  --data_root ./speech_commands_v0.02 \
  --classes yes no stop go up down left right on off \
  --epochs 20 --batch_size 128 --lr 1e-3 \
  --save_path models/baseline_cnn_final.pt
```

### Step 3️⃣ – Noise Evaluation
Evaluate model robustness under different noise levels:

```bash
python evaluate_noise.py \
  --data_root ./speech_commands_v0.02 \
  --model_path models/baseline_cnn_final.pt \
  --plot_confusion_all True \
  --results_dir results_feedback2
```

Outputs:
- `accuracy_vs_snr.png`
- `confusion_30db.png` → `confusion_-5db.png`

---

## 📊 Experimental Results

| **SNR (dB)** | **Accuracy (%)** |
|---------------|------------------|
| 30 | 68.5 |
| 20 | 63.4 |
| 10 | 56.7 |
| 0  | 46.2 |
| −5 | 37.8 |

### Key Findings
- Accuracy remains stable up to 20 dB but drops sharply below 10 dB.  
- Confusion increases between acoustically similar pairs (*go/no*, *on/off*).  
- Below 0 dB, predictions become near-random.

📈 **Figure 1 – Accuracy vs. SNR:** shows clear degradation as noise increases.  
🧩 **Figures 2–6 – Confusion Matrices:** visualize misclassification trends.

---

## 💬 Discussion

The baseline CNN achieves ~69 % accuracy on clean data but loses robustness under strong noise, consistent with previous KWS studies (Li et al., 2022).  
It relies heavily on **shallow spectral cues** that are easily masked by noise.

### Identified Issues
- **Class-specific vulnerability:** short or spectrally simple commands (e.g., *up*, *on/off*) are most affected.  
- **Phonetic overlap:** confusion rises between *go/no* as SNR decreases.  

---

## 🚀 Planned Improvements (Final Stage)

| Enhancement | Purpose |
|--------------|----------|
| **SpecAugment** (Park et al., 2019) | Increase data diversity and generalization |
| **Noise front-end** (Reddy et al., 2021) | Denoising via spectral subtraction or learnable filters |
| **CNN-GRU / Attention Modules** | Capture temporal and contextual dependencies |
| **Real-world noise tests** | Evaluate beyond synthetic Gaussian noise |

---

## 🧪 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/jthhhh123-wq/elec5305-project-540101436.git
cd elec5305-project-540101436

# 2. Set up environment
conda create -n kws python=3.10 -y
conda activate kws
pip install -r requirements.txt

# 3. Train baseline model
python train_baseline.py --data_root ./speech_commands_v0.02

# 4. Evaluate under noise
python evaluate_noise.py --model_path models/baseline_cnn_final.pt
```

---

## 📘 References

1. Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209.  
2. Sainath, T. N., & Parada, C. (2015). *CNNs for Small-Footprint Keyword Spotting.* Interspeech 2015.  
3. Park, D. S., Chan, W., Zhang, Y., et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech 2019.  
4. Li, J., Deng, L., & Gong, Y. (2022). *Noise-Robust Automatic Speech Recognition: A Review.* IEEE/ACM T-ASLP, 30, 1532–1550.  
5. Reddy, C. K. A., Dubey, H., Koishida, K., & Viswanathan, V. (2021). *DNS Challenge: Improving Noise Suppression Models.* Interspeech 2021.  

---

## 👨‍💻 Author

**Jianing Zhang**  
Master of Professional Engineering (Electrical)  
The University of Sydney – *ELEC5305 Project (Feedback 2)*

📧 GitHub: [jthhhh123-wq](https://github.com/jthhhh123-wq)

---
