#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/10/2022

@author: Maurizio Ferrari Dacrema
"""

from torch import nn
import torch
import math
import matplotlib
import matplotlib.pyplot as plt


class SinusoidalPositionalEncoding(object):
    """
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """
    def __init__(self, embedding_size, device):
        super(SinusoidalPositionalEncoding).__init__()
        self.embedding_size = embedding_size

        w_t_exp = -4*2/self.embedding_size * torch.arange(math.floor(self.embedding_size / 2), device=device)
        self.w_t = torch.pow(10, w_t_exp)

    def get_encoding(self, position):
        # Each position will be multiplied for all w_t, the result is |batch|x|position_encoding|
        encoding = torch.outer(position, self.w_t)

        if len(self.w_t)*2<self.embedding_size:
            encoding = torch.cat((encoding.sin(), encoding.cos(), torch.zeros((len(position),1), device=self.w_t.device)), dim=-1)
        else:
            encoding = torch.cat((encoding.sin(), encoding.cos()), dim=-1)

        return encoding




if __name__ == '__main__':

    device = torch.device("cpu:0")
    embedding_size = 2000
    noise_steps = 1000
    encoding = SinusoidalPositionalEncoding(embedding_size = embedding_size, device = device)

    encoding_data = torch.empty((noise_steps, embedding_size))
    for step in range(noise_steps):
        encoding_data[step,:] = encoding.get_encoding(torch.tensor([step]))

    plt.imshow(encoding_data.numpy(), cmap='PuRd')
    plt.savefig("E:/PyCharmWorkspace/RecSysFramework_private/result_experiments/", dpi = 2400, bbox_inches='tight')



