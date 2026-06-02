import os
import sys

# Crucial memory optimization for deep 24-layer networks
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from dataset import SpeechDataset
from dac_transformer import DACTransformerModel

# High-performance hardware acceleration flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

TRAIN_NOISY = "/beegfs/work_fast/data/shared/urgent2025_challenge/simulation_train/noisy_16k"
TRAIN_CLEAN = "/beegfs/work_fast/data/shared/urgent2025_challenge/simulation_train/clean_16k"

VAL_NOISY = "/beegfs/work_fast/data/shared/urgent2025_challenge/offical_validation/noisy_16k"
VAL_CLEAN = "/beegfs/work_fast/data/shared/urgent2025_challenge/offical_validation/clean_16k"


def collate_fn(batch):
    noisy_list = []
    clean_list = []

    lengths = [x[0].shape[-1] for x in batch]
    max_len = max(lengths)

    for noisy, clean in batch:
        pad_len = max_len - noisy.shape[-1]

        # Explicit 1D padding along the time axis
        noisy = torch.nn.functional.pad(noisy, (0, pad_len))
        clean = torch.nn.functional.pad(clean, (0, pad_len))

        noisy_list.append(noisy)
        clean_list.append(clean)

    return torch.stack(noisy_list), torch.stack(clean_list)


def validate(model, loader, device, amp_dtype):
    model.eval()
    total_loss = 0

    # Clean backend configuration for SDPA execution routing
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.no_grad():
        
        for noisy, clean in loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # Route validation through FlashAttention and AMP
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    loss = model(noisy, clean)

            total_loss += loss.item()

    return total_loss / len(loader)


def save_performance_plots(epochs_tracked, train_losses, val_losses, train_accs, val_accs):
    """
    Generates and saves dual-panel plots for loss and structural accuracy.
    """
    plt.figure(figsize=(12, 5))

    # Panel 1: Training & Validation Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_tracked, train_losses, label="Train Loss", color="tab:blue", marker="o")
    plt.plot(epochs_tracked, val_losses, label="Val Loss", color="tab:orange", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Score")
    plt.title("Regression Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Acoustic Accuracy Metric curves 
    plt.subplot(1, 2, 2)
    plt.plot(epochs_tracked, train_accs, label="Train Accuracy", color="tab:green", marker="o")
    plt.plot(epochs_tracked, val_accs, label="Val Accuracy", color="tab:red", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy Vector")
    plt.title("Acoustic Reconstruction Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("training_performance.png", dpi=300)
    plt.close()
    print("[PLOT ENGINE] Performance visualization metrics saved to 'training_performance.png'.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extract major version element from capability tuple
    use_bfloat16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    
    print(f"Device: {device} | AMP Precision: {amp_dtype}")

    train_dataset = SpeechDataset(TRAIN_NOISY, TRAIN_CLEAN)
    val_dataset = SpeechDataset(VAL_NOISY, VAL_CLEAN)

    PHYSICAL_BATCH_SIZE = 2
    ACCUMULATION_STEPS = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,          
        persistent_workers=True   
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )

    model = DACTransformerModel(
        hidden_size=512,
        intermediate_size=1024,
        num_layers=12,
        num_heads=8
    ).to(device)

    # Enable gradient checkpointing on the internal transformer block if available
    if hasattr(model.transformer, "gradient_checkpointing_enable"):
        model.transformer.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    trainable_params = [p for p in model.transformer.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=1e-4
    )

    epochs = 100
    best_val = float("inf")

    # TRACKING VARIABLES
    global_step = 0
    checkpoint_interval = 100000
    patience = 5
    patience_counter = 0

    # PLOT HISTORY BUFFERS
    history_epochs = []
    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    # Clean backend configuration for SDPA execution routing
    from torch.nn.attention import SDPBackend, sdpa_kernel

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0
        
        optimizer.zero_grad(set_to_none=True)

        for step, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    loss = model(noisy, clean) / ACCUMULATION_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
                
                if (step + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                
                if (step + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # --- MONITOR 100K PERFORMANCE MILESTONE ---
            global_step += 1
            if global_step == checkpoint_interval:
                print(f"\n[INTERRUPT] Reached exactly step {global_step}. Pausing to verify performance...")
                step_100k_val = validate(model, val_loader, device, amp_dtype)
                print(f">>> Performance at Step {global_step} | Validation Loss: {step_100k_val:.4f} <<<")

                torch.save(model.transformer.state_dict(), f"transformer_step_{global_step}.pt")
                print(f"Mid-epoch checkpoint saved as: 'transformer_step_{global_step}.pt'")
                model.train()
            # if global_step == checkpoint_interval:
            #     print(
            #            f"\n[INTERRUPT] Reached exactly step {global_step}. "
            #                            f"Running detailed diagnostics..."
            #                                                          )

            #     model.eval()

            #     with torch.no_grad():
            #         #
            #         # Grab a single validation batch
            #         #
            #         noisy_val, clean_val = next(iter(val_loader))

            #         noisy_val = noisy_val.to(device, non_blocking=True)
            #         clean_val = clean_val.to(device, non_blocking=True)

            #         #
            #         # DAC latent statistics
            #         #
            #         noisy_z = model.encode(noisy_val)
            #         clean_z = model.encode(clean_val)

            #         enhanced_z = model.predict_latents(noisy_val)

            #         baseline_mse = F.mse_loss(noisy_z, clean_z)
            #         model_mse = F.mse_loss(enhanced_z, clean_z)

            #         print("\n========== DAC LATENT STATISTICS ==========")
            #         print(f"Mean : {clean_z.mean().item():.4f}")
            #         print(f"Std  : {clean_z.std().item():.4f}")
            #         print(f"Min  : {clean_z.min().item():.4f}")
            #         print(f"Max  : {clean_z.max().item():.4f}")

            #         print("\n========== LATENT MSE ==========")
            #         print(f"Baseline MSE : {baseline_mse.item():.4f}")
            #         print(f"Model MSE    : {model_mse.item():.4f}")
            #         print(
            #               f"Improvement  : "
            #               f"{(baseline_mse - model_mse).item():.4f}"
            #               )

            #         #
            #         # Full validation loss
            #         #
            #     step_100k_val = validate(model, val_loader, device, amp_dtype )

            #     print(
            #              f"\n>>> Performance at Step {global_step}"
            #               f" | Validation Loss: {step_100k_val:.4f} <<<"
            #                 )

            #     torch.save(
            #             model.transformer.state_dict(),
            #                 f"transformer_step_{global_step}.pt"
            #                     )

            #     print(
            #                   f"Mid-epoch checkpoint saved as "
            #                      f"'transformer_step_{global_step}.pt'"
            #                       )

            #     model.train()

            # Accumulate unscaled raw loss value for accurate telemetry tracking
            actual_step_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += actual_step_loss
            pbar.set_description(f"Loss: {actual_step_loss:.4f}")

        train_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, device, amp_dtype)

        # Mathematical mapping from continuous loss values to a 0.0-1.0 bounded reconstruction accuracy matrix
        train_accuracy = 1.0 / (1.0 + train_loss)
        val_accuracy = 1.0 / (1.0 + val_loss)

        # Append data to plotting vectors
        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_accuracy)
        history_val_acc.append(val_accuracy)

        # Output graph visualization files automatically on each completed epoch 
        save_performance_plots(history_epochs, history_train_loss, history_val_loss, history_train_acc, history_val_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

        # --- EARLY STOPPING ENGINE ---
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0  # Reset patience on improvement
            torch.save(model.transformer.state_dict(), "best_transformer.pt")
            print("Best transformer weights saved.")
        else:
            patience_counter += 1
            print(f"No improvement in Val Loss. Patience counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n[EARLY STOP] Triggered! Validation loss failed to improve for {patience} epochs. Terminating.")
                break


if __name__ == "__main__":
    main()
