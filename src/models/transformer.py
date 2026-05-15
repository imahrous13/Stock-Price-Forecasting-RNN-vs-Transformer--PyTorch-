import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000):

        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        output_size: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        max_seq_len: int = 5000
    ):
        
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = x[-1, :, :]
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
        
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x



