import torch
import torch.nn as nn

"""
Simply two linear layer with a Relu activation function in the middle:

y output of the layer
x input of the layer

y= max(0, W1x+b1)*W2 +b2

where:
- ReLu = max(0, input) by definiton
- W1, b1 are the parameters of the first linear layer
- W2, b2 are the parameters of the second linear layer

"""

class FeedForwardBlock(nn.module):
    
    def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # firts layer with W1,b1. d_model dimension of imput, dff intermediary dimension
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2, b2

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
