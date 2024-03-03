from TransformerLuca.TransformerArchitecrture.FeedForwardBlock import FeedForwardBlock
from TransformerLuca.TransformerArchitecrture.LayerNormalization import LayerNormalization
from TransformerLuca.TransformerArchitecrture.MultiHeadAttentionBlock import MultiHeadAttentionBlock
from TransformerLuca.TransformerArchitecrture.ResidualConnection import ResidualConnection
import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, device, features:int, attention_block: MultiHeadAttentionBlock, feed_forward_blcok: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_blcok
        self.residual_connections = nn.ModuleList([ResidualConnection(device=device,features=features, dropout=dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):

    def __init__(self, device, features: int, layers: nn.ModuleList)->None:
        super().__init_()
        self.layers = layers
        self.norm = LayerNormalization(device=device, features=features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)