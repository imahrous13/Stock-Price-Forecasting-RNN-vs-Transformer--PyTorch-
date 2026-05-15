from __future__ import annotations

import torch
import torch.nn as nn


class CausalTransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        output_size: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 256,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

        self.register_buffer(
            "_causal_mask_cache",
            torch.empty(0),
            persistent=False,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (
            self._causal_mask_cache.shape == (seq_len, seq_len)
            and self._causal_mask_cache.device == device
        ):
            return self._causal_mask_cache
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        self._causal_mask_cache = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input seq_len={seq_len} > max_seq_len={self.max_seq_len}"
            )

        x = self.input_projection(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_embedding(positions)
        x = self.input_dropout(x)

        mask = self._causal_mask(seq_len, x.device)
        x = self.encoder(x, mask=mask, is_causal=True)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
