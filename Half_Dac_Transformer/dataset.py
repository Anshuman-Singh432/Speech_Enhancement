import os
import torch
import torchaudio
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """
    PURPOSE:
    Loads paired noisy-clean speech.

    INPUT:
    noisy wav
    clean wav

    OUTPUT:
    noisy_waveform  -> (1, T)
    clean_waveform  -> (1, T)
    """

    def __init__(self, noisy_dir, clean_dir, sample_rate=16000):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate

        self.files = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.files)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)

        # Convert stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.sample_rate
            )

        return wav

    def __getitem__(self, idx):
        filename = self.files[idx]

        noisy_path = os.path.join(self.noisy_dir, filename)
        clean_path = os.path.join(self.clean_dir, filename)

        noisy = self.load_audio(noisy_path)
        clean = self.load_audio(clean_path)

        # Match lengths
        min_len = min(noisy.shape[-1], clean.shape[-1])

        noisy = noisy[..., :min_len]
        clean = clean[..., :min_len]

        return noisy, clean
