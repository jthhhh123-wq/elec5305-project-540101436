import os
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ====== Paths ======
EXAMPLE_PATH = r"./data/SpeechCommands/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"
FIG_DIR = "figures"

# ====== Mel-spectrogram parameters (same as training) ======
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 320

mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)

def to_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram (dB scale)."""
    spec = mel_fn(waveform)
    spec_db = torchaudio.functional.amplitude_to_DB(
        spec,
        multiplier=10.0,
        amin=1e-10,
        db_multiplier=0.0,
    )
    return spec_db


def add_awgn(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to a waveform at a given SNR (in dB)."""
    noise = torch.randn_like(waveform)
    sig_power = waveform.pow(2).mean()
    noise_power = noise.pow(2).mean()
    scale = (sig_power / (10 ** (snr_db / 10)) / noise_power).sqrt()
    noisy = waveform + scale * noise
    return noisy


def simple_reverb(waveform: torch.Tensor, decay: float = 0.9, length: int = 80) -> torch.Tensor:
    """
    Apply a simple convolutional reverb (no SoX, works on Windows):
    an exponentially decaying impulse response is convolved with the signal.
    This is only for visualization, not an exact match to training.
    """
    # waveform: (1, T)
    ir = torch.pow(decay, torch.arange(length, dtype=waveform.dtype))
    ir = ir / ir.sum()                          # normalize
    ir = ir.view(1, 1, -1)                      # (out_channels, in_channels, kernel_size)

    x = waveform.unsqueeze(0)                   # (batch=1, ch=1, T)
    y = F.conv1d(x, ir, padding=length - 1)     # keep length similar
    y = y.squeeze(0)                            # (1, T')
    return y


def plot_single_mel(mel_db: torch.Tensor, title: str, out_path: str):
    """Plot a single mel-spectrogram and save to disk."""
    plt.figure(figsize=(8, 4))
    plt.imshow(
        mel_db.squeeze().numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Mel frequency bins")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved to: {out_path}")


def main():
    # ====== 1. Load clean waveform ======
    print(f"Loading: {EXAMPLE_PATH}")
    waveform, sr = torchaudio.load(EXAMPLE_PATH)
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        print(f"Resampled to {SAMPLE_RATE} Hz, new shape: {waveform.shape}")

    # ====== 2. Figure 1: clean mel-spectrogram (your original figure) ======
    clean_mel_db = to_log_mel(waveform)
    plot_single_mel(
        clean_mel_db,
        title="Mel-Spectrogram of 'yes'",
        out_path=os.path.join(FIG_DIR, "melspec_yes.png"),
    )

    # ====== 3. Create waveform variants ======
    # Noisy (0 dB), Curriculum-style hardest stage (-5 dB),
    # and SE+Reverb approximation (reverb applied on -5 dB noisy signal).
    noisy_wave = add_awgn(waveform, snr_db=0.0)
    curr_wave = add_awgn(waveform, snr_db=-5.0)
    se_input = add_awgn(waveform, snr_db=-5.0)
    se_reverb_wave = simple_reverb(se_input, decay=0.9, length=80)

    # ====== 4. Convert all to log-mel spectrograms ======
    noisy_mel_db = to_log_mel(noisy_wave)
    curr_mel_db = to_log_mel(curr_wave)
    se_reverb_mel_db = to_log_mel(se_reverb_wave)

    # ====== 5. Plot three variants side-by-side ======
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def show_mel(ax, mel_db, title):
        ax.imshow(mel_db.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Mel bins")

    show_mel(axes[0], noisy_mel_db, "Noisy (0 dB AWGN)")
    show_mel(axes[1], curr_mel_db, "Curriculum stage (−5 dB AWGN)")
    show_mel(axes[2], se_reverb_mel_db, "SE+Reverb (approx.)")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "melspec_variants_noisy_curriculum_sereverb.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Variant figure saved to: {out_path}")


if __name__ == "__main__":
    main()
