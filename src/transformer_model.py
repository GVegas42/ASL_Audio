import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths=None):
        # x: (B, T, D_in)
        x = self.input_proj(x)  # (B, T, D)
        x = self.pos_enc(x)

        # Build padding mask before transpose: True = padding (ignore in attention)
        if lengths is not None:
            B, T = x.size(0), x.size(1)
            src_key_padding_mask = (
                torch.arange(T, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
            )  # (B, T)
        else:
            src_key_padding_mask = None

        # Transformer expects (T, B, D)
        x = x.transpose(0, 1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (T, B, D)

        x = x.transpose(0, 1)  # (B, T, D)

        # Simple pooling: mean over time (ignoring padding if lengths provided)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.to(x.dtype)
            summed = (x * mask.unsqueeze(-1)).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = summed / denom
        else:
            pooled = x.mean(dim=1)

        out = self.classifier(pooled)
        return out
