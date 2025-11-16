import os
import torchaudio
import matplotlib.pyplot as plt


example_path = r"./data/SpeechCommands/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"

print(f"Loading: {example_path}")
waveform, sr = torchaudio.load(example_path)
print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")


mel_fn = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=320,
    n_mels=64
)

mel = mel_fn(waveform)


mel_db = torchaudio.functional.amplitude_to_DB(
    mel,
    multiplier=10.0,
    amin=1e-10,
    db_multiplier=0.0
)

print("Mel spectrogram shape:", mel_db.shape)


plt.figure(figsize=(8, 4))
plt.imshow(
    mel_db.squeeze().numpy(),
    origin="lower",
    aspect="auto",
    cmap="magma"
)
plt.colorbar(label="dB")
plt.title("Mel-Spectrogram of 'yes'")
plt.xlabel("Time frames")
plt.ylabel("Mel frequency bins")
plt.tight_layout()


os.makedirs("figures", exist_ok=True)
out_path = os.path.join("figures", "melspec_yes.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Figure saved to: {out_path}")


