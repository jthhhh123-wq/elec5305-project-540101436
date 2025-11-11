# 🎙️ Noise-Robust Keyword Spotting with ConvMixer

This project implements a **Keyword Spotting (KWS)** system based on **ConvMixer** for recognizing short speech commands under noisy environments.  
It includes a clean **baseline**, a **noise-augmented (AWGN)** experiment, and an improved **AWGN_v2** version.

---

## ⚙️ Environment Setup

### 1. Create and activate conda environment
```bash
conda create -n kws python=3.10
conda activate kws

### 2. Install dependencies
pip install torch torchaudio matplotlib pyyaml


