import os
import torch
import torchaudio
from tqdm import tqdm

# Changed from relative to absolute top-level import
from dac_transformer import DACTransformerModel

# ============================================================
# PATHS
# ============================================================

# Point directly to the transformer-specific checkpoint matrix
MODEL_PATH = "best_transformer.pt"

TEST_NOISY = "/beegfs/work_fast/data/shared/urgent2025_challenge/nonblindtestset/noisy_16k_soxi"
TEST_CLEAN = "/beegfs/work_fast/data/shared/urgent2025_challenge/nonblindtestset/clean_16k_soxi"
OUTPUT_DIR = "./enhanced_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MAIN INFERENCE ENGINE
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing audio enhancement pipeline on: [{device}]")

    # --------------------------------------------------------
    # Load model architecture & selectively restore weights
    # --------------------------------------------------------
    # CRITICAL FIX 1: Match the exact architecture arguments from your training script
    model = DACTransformerModel(
        hidden_size=1024,
        intermediate_size=2048,
        num_layers=24,
        num_heads=16
    ).to(device)

    # CRITICAL FIX 2: Added weights_only=True to comply with PyTorch security standards
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.transformer.load_state_dict(state_dict)
    
    # Enforce pure validation mode constraints globally
    model.eval()

    # --------------------------------------------------------
    # Scan and filter noisy test assets
    # --------------------------------------------------------
    valid_extensions = (".wav", ".flac", ".ogg")
    noisy_files = sorted([
        f for f in os.listdir(TEST_NOISY) 
        if f.lower().endswith(valid_extensions)
    ])

    print(f"Found {len(noisy_files)} valid audio target files for processing.")

    # --------------------------------------------------------
    # Linear inference processing queue
    # --------------------------------------------------------
    for filename in tqdm(noisy_files, desc="Enhancing"):
        noisy_path = os.path.join(TEST_NOISY, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # Load noisy waveform -> shape: [channels, samples]
            wav, sr = torchaudio.load(noisy_path)

            # Enforce strict mono down-mixing -> shape: [1, samples]
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # CRITICAL FIX 3: Replicate exact tensor dimension structure of your training batch.
            # If your SpeechDataset returns [1, samples], it is already batch-ready after unsqueeze(0).
            # If your dataset returns [samples] (no channel dim), use wav = wav.squeeze(0).unsqueeze(0) 
            wav = wav.to(device) 
            if wav.dim() == 2:
                # Changes [1, samples] -> [1, 1, samples] (Match this to your training dataset shape)
                wav = wav.unsqueeze(0) 

            # Pure mathematical execution block with mixed precision capability
            with torch.no_grad():
                # Force bfloat16 processing if your GPU architecture supports it
                use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
                
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                    # clean_wav=None forces the inference waveform reconstruction pathway
                    enhanced = model(wav, clean_wav=None)

            # Post-process tensor output safely back to host space memory
            # If enhanced output keeps batch/channel dims, safely reduce to [channels, time]
            enhanced_wav = enhanced.cpu().to(torch.float32)
            while enhanced_wav.dim() > 2:
                enhanced_wav = enhanced_wav.squeeze(0)
            if enhanced_wav.dim() == 1:
                enhanced_wav = enhanced_wav.unsqueeze(0)

            # Export data assets securely to storage block targets
            torchaudio.save(output_path, enhanced_wav, sr)

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to enhance {filename}: {e}")
            continue

    print("\n Audio inference pipeline completed successfully.")


if __name__ == "__main__":
    main()
