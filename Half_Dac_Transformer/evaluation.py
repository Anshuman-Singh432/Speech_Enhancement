# ============================================================
# evaluate.py
# ============================================================
#
# PURPOSE:
# Evaluate enhanced speech against clean speech.
#
# Computes:
# 1. PESQ
# 2. STOI
# 3. SI-SDR
#
# INPUT:
# Enhanced speech folder
# Clean speech folder
#
# OUTPUT:
# Average metrics across test set
#
# ============================================================


# import os
# import torch
# import torchaudio
# import numpy as np
# from tqdm import tqdm

# from pesq import pesq
# from pystoi import stoi


# # ============================================================
# # PATHS
# # ============================================================

# ENHANCED_DIR = "./enhanced_outputs"

# TEST_CLEAN = "/beegfs/work_fast/data/shared/urgent2025_challenge/nonblindtestset/clean_16k_soxi"


# # ============================================================
# # SI-SDR FUNCTION
# # ============================================================

# def compute_si_sdr(reference, estimation, eps=1e-8):
#     """
#     PURPOSE:
#     Compute Scale-Invariant Signal-to-Distortion Ratio.

#     WHY?
#     Measures speech enhancement quality.

#     HIGHER = BETTER

#     INPUT:
#     reference:
#         clean waveform
#         shape: (samples)

#     estimation:
#         enhanced waveform
#         shape: (samples)

#     OUTPUT:
#     scalar SI-SDR value
#     """

#     # Convert to numpy
#     reference = reference.astype(np.float64)
#     estimation = estimation.astype(np.float64)

#     # --------------------------------------------
#     # Zero-mean normalization
#     # --------------------------------------------

#     reference = reference - np.mean(reference)
#     estimation = estimation - np.mean(estimation)

#     # --------------------------------------------
#     # Projection of estimation onto reference
#     # --------------------------------------------

#     reference_energy = np.sum(reference ** 2) + eps

#     scale = np.sum(reference * estimation) / reference_energy

#     projection = scale * reference

#     # --------------------------------------------
#     # Noise/error component
#     # --------------------------------------------

#     noise = estimation - projection

#     # --------------------------------------------
#     # SI-SDR computation
#     # --------------------------------------------

#     ratio = np.sum(projection ** 2) / (
#         np.sum(noise ** 2) + eps
#     )

#     si_sdr = 10 * np.log10(ratio + eps)

#     return si_sdr


# # ============================================================
# # LOAD AUDIO FUNCTION
# # ============================================================

# def load_audio(path, target_sr=16000):
#     """
#     PURPOSE:
#     Load waveform and ensure:
#     - mono audio
#     - correct sample rate

#     OUTPUT:
#     waveform:
#         shape: (samples)
#     """

#     wav, sr = torchaudio.load(path)

#     # --------------------------------------------
#     # Convert stereo → mono
#     # --------------------------------------------

#     if wav.shape[0] > 1:
#         wav = wav.mean(dim=0, keepdim=True)

#     # --------------------------------------------
#     # Resample if needed
#     # --------------------------------------------

#     if sr != target_sr:

#         wav = torchaudio.functional.resample(
#             wav,
#             sr,
#             target_sr
#         )

#     # Remove channel dimension
#     # (1, samples) → (samples)

#     wav = wav.squeeze(0)

#     return wav.numpy(), target_sr


# # ============================================================
# # MAIN EVALUATION
# # ============================================================

# def main():

#     # --------------------------------------------------------
#     # Get all enhanced files
#     # --------------------------------------------------------

#     enhanced_files = sorted(
#         os.listdir(ENHANCED_DIR)
#     )

#     print(f"Found {len(enhanced_files)} enhanced files")

#     # --------------------------------------------------------
#     # Metric accumulators
#     # --------------------------------------------------------

#     total_pesq = 0
#     total_stoi = 0
#     total_sisdr = 0

#     valid_pesq_count = 0

#     # --------------------------------------------------------
#     # Loop through files
#     # --------------------------------------------------------

#     for filename in tqdm(enhanced_files):

#         enhanced_path = os.path.join(
#             ENHANCED_DIR,
#             filename
#         )

#         clean_path = os.path.join(
#             TEST_CLEAN,
#             filename
#         )

#         # Skip if clean file missing
#         if not os.path.exists(clean_path):
#             print(f"Missing clean file: {filename}")
#             continue

#         # ----------------------------------------------------
#         # Load audio
#         # ----------------------------------------------------

#         enhanced, sr = load_audio(enhanced_path)

#         clean, _ = load_audio(clean_path)

#         # ----------------------------------------------------
#         # Match lengths
#         # ----------------------------------------------------

#         min_len = min(len(enhanced), len(clean))

#         enhanced = enhanced[:min_len]
#         clean = clean[:min_len]

#         # ----------------------------------------------------
#         # PESQ
#         # ----------------------------------------------------

#         try:

#             pesq_score = pesq(
#                 sr,
#                 clean,
#                 enhanced,
#                 'wb'      # wideband
#             )

#             total_pesq += pesq_score

#             valid_pesq_count += 1

#         except Exception as e:

#             print(f"PESQ failed for {filename}")

#         # ----------------------------------------------------
#         # STOI
#         # ----------------------------------------------------

#         stoi_score = stoi(
#             clean,
#             enhanced,
#             sr,
#             extended=False
#         )

#         total_stoi += stoi_score

#         # ----------------------------------------------------
#         # SI-SDR
#         # ----------------------------------------------------

#         sisdr_score = compute_si_sdr(
#             clean,
#             enhanced
#         )

#         total_sisdr += sisdr_score

#     # --------------------------------------------------------
#     # Number of evaluated files
#     # --------------------------------------------------------

#     n = len(enhanced_files)

#     # --------------------------------------------------------
#     # Average metrics
#     # --------------------------------------------------------

#     avg_pesq = total_pesq / max(valid_pesq_count, 1)

#     avg_stoi = total_stoi / n

#     avg_sisdr = total_sisdr / n

#     # --------------------------------------------------------
#     # Print results
#     # --------------------------------------------------------

#     print("\n================================================")
#     print("FINAL EVALUATION RESULTS")
#     print("================================================")

#     print(f"PESQ   : {avg_pesq:.4f}")

#     print(f"STOI   : {avg_stoi:.4f}")

#     print(f"SI-SDR : {avg_sisdr:.4f}")

#     print("================================================")


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     main()



import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi


# ============================================================
# PATHS
# ============================================================

ENHANCED_DIR = "./enhanced_outputs"

TEST_CLEAN = "/beegfs/work_fast/data/shared/urgent2025_challenge/nonblindtestset/clean_16k_soxi"


# ============================================================
# SI-SDR FUNCTION
# ============================================================

def compute_si_sdr(reference, estimation, eps=1e-8):
    """
    PURPOSE:
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    Capped at -50dB to remove extreme mathematical artifacts while keeping 
    true negative trends accurate.
    """
    reference = reference.astype(np.float64)
    estimation = estimation.astype(np.float64)

    # Zero-mean normalization
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)

    # Projection of estimation onto reference
    reference_energy = np.sum(reference ** 2) + eps
    scale = np.sum(reference * estimation) / reference_energy
    projection = scale * reference

    # Noise/error component
    noise = estimation - projection

    # SI-SDR computation
    projection_energy = np.sum(projection ** 2)
    noise_energy = np.sum(noise ** 2) + eps

    if projection_energy < eps:
        return -50.0

    ratio = projection_energy / noise_energy
    si_sdr = 10 * np.log10(ratio + eps)

    # ADJUSTED FIX: Set floor to -50.0 to prevent inflation of bad outputs
    return max(-50.0, si_sdr)


# ============================================================
# LOAD AUDIO FUNCTION
# ============================================================

def load_audio(path, target_sr=16000):
    """
    PURPOSE:
    Load waveform and ensure mono audio, correct sample rate, and 1D format.
    """
    wav, sr = torchaudio.load(path)

    # Convert stereo -> mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # flatten() safely replaces squeeze(0) to guarantee a 1D array
    wav = wav.flatten()

    return wav.numpy(), target_sr


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    if not os.path.exists(ENHANCED_DIR):
        print(f"CRITICAL: Enhanced folder path missing: {ENHANCED_DIR}")
        return

    # Target valid extensions only to ignore system files (.DS_Store)
    valid_extensions = (".wav", ".flac", ".ogg")
    enhanced_files = sorted([
        f for f in os.listdir(ENHANCED_DIR) 
        if f.lower().endswith(valid_extensions)
    ])

    print(f"Found {len(enhanced_files)} valid enhanced audio files")

    # Metric accumulators
    total_pesq = 0.0
    total_stoi = 0.0
    total_sisdr = 0.0

    # CRITICAL FIX 1: Track all metric counts independently to avoid division biases
    valid_pesq_count = 0
    valid_stoi_count = 0
    valid_sisdr_count = 0

    for filename in tqdm(enhanced_files, desc="Evaluating"):
        enhanced_path = os.path.join(ENHANCED_DIR, filename)
        clean_path = os.path.join(TEST_CLEAN, filename)

        if not os.path.exists(clean_path):
            print(f"Missing clean file target match: {filename}")
            continue

        try:
            enhanced, sr = load_audio(enhanced_path)
            clean, _ = load_audio(clean_path)

            min_len = min(len(enhanced), len(clean))
            
            # Guard against edge-case empty audio clips
            if min_len == 0:
                print(f"Skipping zero-length file: {filename}")
                continue

            enhanced = enhanced[:min_len]
            clean = clean[:min_len]

            # ── PESQ Calculation ──
            try:
                pesq_score = pesq(sr, clean, enhanced, 'wb')
                total_pesq += pesq_score
                valid_pesq_count += 1
            except Exception:
                print(f"PESQ processing failed for {filename}")

            # ── STOI Calculation ──
            try:
                stoi_score = stoi(clean, enhanced, sr, extended=False)
                total_stoi += stoi_score
                valid_stoi_count += 1
            except Exception:
                print(f"STOI processing failed for {filename}")

            # ── SI-SDR Calculation ──
            try:
                sisdr_score = compute_si_sdr(clean, enhanced)
                total_sisdr += sisdr_score
                valid_sisdr_count += 1
            except Exception:
                print(f"SI-SDR processing failed for {filename}")

        except Exception as e:
            print(f"Failed parsing pipeline for {filename}: {e}")
            continue

    # Handle completely empty evaluation sets safely
    if max(valid_pesq_count, valid_stoi_count, valid_sisdr_count) == 0:
        print("No valid files evaluated across any metrics.")
        return

    # CRITICAL FIX 2: Compute averages based strictly on independent successful counts
    avg_pesq = total_pesq / max(valid_pesq_count, 1)
    avg_stoi = total_stoi / max(valid_stoi_count, 1)
    avg_sisdr = total_sisdr / max(valid_sisdr_count, 1)

    print("\n================================================")
    print("FINAL EVALUATION RESULTS")
    print("================================================")
    print(f"Successful PESQ Files   : {valid_pesq_count}")
    print(f"Successful STOI Files   : {valid_stoi_count}")
    print(f"Successful SI-SDR Files : {valid_sisdr_count}")
    print("------------------------------------------------")
    print(f"PESQ (Wideband)         : {avg_pesq:.4f}  (Range: -0.5 to 4.5)")
    print(f"STOI                    : {avg_stoi:.4f}  (Range: 0.0 to 1.0)")
    print(f"SI-SDR (dB)             : {avg_sisdr:.4f}")
    print("================================================")


if __name__ == "__main__":
    main()
