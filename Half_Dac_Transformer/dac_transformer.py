import torch
from torch import nn

import dac
from dac.model import DAC
from transformer import TransformerEnhancer


class DACTransformerModel(nn.Module):
    """
    PURPOSE:
    Full speech enhancement system.

    PIPELINE:

    Noisy Speech
    → DAC Encoder
    → Input Projection (1024 → hidden_size)
    → Transformer
    → Output Projection (hidden_size → 1024)
    → DAC Decoder
    → Enhanced Speech

    TRAINING TARGET:
    Predict clean DAC embeddings.
    """

    def __init__(
        self,
        hidden_size=512,
        intermediate_size=1024,
        num_layers=12,
        num_heads=8,
        freeze_dac=True
    ):
        super().__init__()

        # Load pretrained DAC
        model_path = dac.utils.download(model_type="16khz")
        self.dac = DAC.load(model_path)

        # Freeze DAC
        if freeze_dac:
            for p in self.dac.parameters():
                p.requires_grad = False

        # --- STRUCTURAL FIX: Channel Projection Layers ---
        # Maps DAC output (1024) to your smaller transformer hidden size (e.g., 512)
        self.input_projection = nn.Linear(1024, hidden_size)
        # Maps transformer hidden size back to DAC target dimension (1024) for loss/decoder
        self.output_projection = nn.Linear(hidden_size, 1024)

        self.transformer = TransformerEnhancer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads
        )

    def encode(self, wav):
        """
        PURPOSE:
        Convert waveform → DAC embeddings

        INPUT:
        wav:
        (B, 1, samples)

        OUTPUT:
        embeddings:
        (B, T, 1024)
        """

        with torch.no_grad():
            z, codes, latents, _, _ = self.dac.encode(wav)

        # z:
        # (B, 1024, T)

        z = z.transpose(1, 2)

        # (B, T, 1024)

        return z

    def decode(self, z):
        """
        PURPOSE:
        Convert embeddings → waveform

        INPUT:
        z:
        (B, T, 1024)

        OUTPUT:
        waveform:
        (B, 1, samples)
        """

        z = z.transpose(1, 2)

        # (B, 1024, T)

        wav = self.dac.decode(z)

        return wav

    def forward(self, noisy_wav, clean_wav=None):
        """
        PURPOSE:
        Training forward pass.

        INPUT:
        noisy_wav:
        (B, 1, samples)

        clean_wav:
        (B, 1, samples)

        OUTPUT:
        loss OR enhanced waveform
        """

        # ---------------------------------
        # Encode noisy speech
        # ---------------------------------
        # 1. Encode noisy speech (Output has requires_grad=False)
        noisy_z = self.encode(noisy_wav)

        # CRITICAL FIX: Enable gradient tracking on the input tensor
        # This allows gradient checkpointing to function correctly!
        if self.training:
            noisy_z = noisy_z.clone().requires_grad_()

        # noisy_z: (B, T, 1024)

        # ---------------------------------
        # Transformer enhancement
        # ---------------------------------
        # Project from 1024 channels down to transformer size (e.g., 512)
        transformer_input = self.input_projection(noisy_z)  # (B, T, hidden_size)

        # Process with your fast, downsized transformer block
        transformer_output = self.transformer(transformer_input)  # (B, T, hidden_size)

        # Project back up to 1024 channels to match clean target expectations
        enhanced_z = self.output_projection(transformer_output)  # (B, T, 1024)
        # delta_z = self.output_projection(transformer_output)
        # enhanced_z = noisy_z + delta_z

        # ---------------------------------
        # Inference mode
        # ---------------------------------
        if clean_wav is None:
            enhanced_wav = self.decode(enhanced_z)
            return enhanced_wav

        # ---------------------------------
        # Encode clean speech
        # ---------------------------------
        # 4. Encode clean speech
        with torch.no_grad():
            clean_z = self.encode(clean_wav).detach()

        # clean_z: (B, T, 1024)

        # ---------------------------------
        # Embedding MSE loss
        # ---------------------------------
        # Safe slicing to handle edge-case padding variations from DAC strides
        T_min = min(enhanced_z.size(1), clean_z.size(1))
        
        loss = torch.nn.functional.mse_loss(  
            enhanced_z[:, :T_min, :],  # (B, T_min, 1024)
            clean_z[:, :T_min, :]      # (B, T_min, 1024)
        )

        return loss
    

    # def predict_latents(self, noisy_wav):
    #     noisy_z = self.encode(noisy_wav)
    #     transformer_input = self.input_projection(noisy_z)
    #     transformer_output = self.transformer(transformer_input)
    #     enhanced_z = self.output_projection(transformer_output)
    #     return enhanced_z
