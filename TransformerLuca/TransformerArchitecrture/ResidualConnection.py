from TransformerLuca.TransformerArchitecrture.LayerNormalization import LayerNormalization
import torch
import torch.nn as nn

"""
Take an input x and add it to itself after it's normalised and processed by a layer
"""

class ResidualConnection(nn.Module):

        def __init__(self, device, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(device=device,features=features)

        def forward(self, x, sublayer):
              return x + self.dropout(sublayer(self.norm(x)))