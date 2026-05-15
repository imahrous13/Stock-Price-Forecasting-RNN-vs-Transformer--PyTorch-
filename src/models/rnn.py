import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNN(nn.Module):
    
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        bidirectional: bool = False
    ):
        
        super(VanillaRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out, _ = self.rnn(x)
        
        out = out[:, -1, :]
        
        if self.use_batch_norm:
            out = self.batch_norm(out)
        
        out = self.dropout(out)
        
        out = self.fc(out)
        
        return out



