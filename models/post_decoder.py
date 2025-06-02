import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
    SIG,
)

class PostDecoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims1,
        encoder_hidden_dims2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim1 = encoder_hidden_dims1
        self.hidden_t_dim2 = encoder_hidden_dims2
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, encoder_hidden_dims1, bias=False),
            linear(encoder_hidden_dims1, encoder_hidden_dims2, bias=False), 
            linear(encoder_hidden_dims2, output_dims, bias=False), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x