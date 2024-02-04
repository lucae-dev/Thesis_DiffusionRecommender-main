import torch
import torch.nn as nn
import LayerNormalization

"""
Take an input x and add it to itself after it's normalised and processed by a layer
"""

class ResidualConnection(nn.Module):

        def __init__(self, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization()

        def froward(self, x, sublayer):
              return x + self.dropout(sublayer(self.norm(x)))