import torch
from torch.utils.checkpoint import checkpoint
from torch import nn


class TransformerBlock(nn.Module):
    """
    PURPOSE:
    Single transformer block.

    FLOW:
    Input
    → MultiHeadAttention
    → FeedForward

    INPUT SHAPE:
    (B, T, D)

    OUTPUT SHAPE:
    (B, T, D)

    WHERE:
    B = batch size
    T = sequence length
    D = hidden size = 1024
    """

    def __init__(
        self,
        hidden_size=512,
        intermediate_size=1024,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):

        # x shape:
        # (B, T, 1024)

        residual = x

        x = self.norm1(x)

        attn_out, _ = self.attn(x, x, x)

        # attn_out:
        # (B, T, 1024)

        x = residual + attn_out

        residual = x

        x = self.norm2(x)

        ff_out = self.ff(x)

        # ff_out:
        # (B, T, 1024)

        x = residual + ff_out

        return x




class TransformerEnhancer(nn.Module):
    """
    24-layer Transformer for speech enhancement in the DAC latent space.
 
    Takes noisy DAC embeddings and maps them to enhanced embeddings of
    the same shape, which are then passed to the DAC decoder.
 
    Args:
        hidden_size (int):       Codec embedding dimension D. Default: 1024.
        intermediate_size (int): FFN inner dimension. Default: 2048.
        num_layers (int):        Number of TransformerBlocks. Default: 24.
        num_heads (int):         Attention heads per block. Default: 16.
        dropout (float):         Dropout applied inside each block. Default: 0.1.
        use_gradient_checkpointing (bool):
            Trades compute for memory by re-running the forward pass during
            backprop instead of storing all activations. Recommended for
            24-layer models on constrained GPU memory. Default: True.
 
    Input:  (B, T, 1024)  — noisy DAC embeddings
    Output: (B, T, 1024)  — enhanced DAC embeddings
    """
 
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
 
        self.use_gradient_checkpointing = use_gradient_checkpointing
 
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
 
        self.final_norm = nn.LayerNorm(hidden_size)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1024)
 
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                # checkpoint() requires inputs to have requires_grad=True to
                # trigger recomputation; use_reentrant=False is the modern,
                # safer API that avoids issues with in-place ops and DDP.
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                # At inference time checkpointing gives no benefit —
                # run the layer normally for speed.
                x = layer(x)
 
        x = self.final_norm(x)
 
        return x  # (B, T, 1024)
 