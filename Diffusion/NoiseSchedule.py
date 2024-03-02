#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/10/2022

@author: Maurizio Ferrari Dacrema
"""

from torch import nn
import torch
import torch.nn.functional as F



class LinearNoiseSchedule(object):
    def __init__(self, start_beta, end_beta, device, noise_timesteps = 1000):
        super(LinearNoiseSchedule).__init__()
        dtype = torch.float32 if device.type == 'mps' else torch.float64
        self._beta_values = torch.linspace(start_beta, end_beta, noise_timesteps, dtype = dtype, device=device)
        alpha = 1. - self._beta_values
        alpha_prod = torch.cumprod(alpha, axis=0, dtype=torch.float)
        self._alpha_values = torch.clamp(alpha_prod, min=0.0, max=1.0)
        alpha_values_prev = F.pad(self._alpha_values[:-1], (1, 0), value = 1.)
        self._posterior_variance = self._beta_values * (1. - alpha_values_prev) / (1. - self._alpha_values)

        self._posterior_log_variance_clipped = torch.log(self._posterior_variance.clamp(min = 1e-20))


        self._posterior_mean_c_x_start = self._beta_values * torch.sqrt(alpha_values_prev) / (1. - alpha_prod)
        self._posterior_mean_c_x_t = (1. - alpha_values_prev) * torch.sqrt(self._alpha_values) / (1. - alpha_prod)

    def get_beta(self, step):
        return self._beta_values[step]

    def get_a_signed(self, step):
        return self._alpha_values[step]




