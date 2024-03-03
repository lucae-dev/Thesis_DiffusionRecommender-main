import torch
import torch.nn as nn

"""
This layer takes an input, of multiple vectors x1, x2... xn for example, and normlizes the content of each one of them, removing the mean mu  and dividing by the standard deviation delta(adjusted to avoid division with 0 or close to 0)

x'j output of the layer Normalization
xj input
muj mean across xj
deltaj standard deviation of xj
epsi 


x'j = (xj - muj)/(sqrt(deltaj^2 +epsi))

Of course we want to be able to exploit this layer to be expressive, we do it by adding to the mix the two learnable parameters alpha and bias. This way the model will not be costrained by values between 0 and 1 but can learn to amplify some


x''j = x'j*alpha + bias



"""

class LayerNormalization(nn.Module):
    def __init__(self, device, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features, device = device)) #initialiazed to 1 since it's multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(features, device = device)) #initialiazed to 0 since it's the additive parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias #!? should it be sqrt(std^2 +eps) ?
    