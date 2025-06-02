import torch
from typing import List, Union
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
LRELU_SLOPE = 0.02

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)


def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)


def norm_layer1d(norm, num_channels):
    if norm == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm == 'instance':
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == 'layer':
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError('%s not recognized.' % norm)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x

class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_sizes: Union[int, list]=3, strides=1,
                 norm=None, activation=None, padding_mode='replicate',
                 padding=None):
        super(Conv3DBlock, self).__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_sizes, strides, padding=padding,
            padding_mode=padding_mode)

        if activation is None:
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'tanh':
            nn.init.xavier_uniform_(self.conv3d.weight,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'lrelu':
            nn.init.kaiming_uniform_(self.conv3d.weight, a=LRELU_SLOPE,
                                     nonlinearity='leaky_relu')
            nn.init.zeros_(self.conv3d.bias)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(self.conv3d.weight, nonlinearity='relu')
            nn.init.zeros_(self.conv3d.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            raise NotImplementedError('Norm not implemented.')
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x



class PreEncoder(nn.Module):
    """
    transform 5 channel to 1 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential( 
            linear(input_dims, output_dims), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x

class MotionEncoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims, bias=False), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x

class VoxelFeatEncoder(nn.Module):
    """
    transform 9 channel to 512 channel
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims, bias=False), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        F = x.shape[2]
        x = x.view(B*T, F)
        x = self.feat_embed(x) 
        x = x.view(B, T, x.shape[-1])

        return x




class LocEncoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., out_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.out_layer = out_layer() if out_layer is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


class GLIPEncoder(nn.Module):
    """
    B,T,256,7,12->B,T,512
    """

    def __init__(
        self,
        input_dims_conv,
        output_dims_conv,
        input_dims,
        output_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_dims_conv = input_dims_conv
        self.output_dims_conv = output_dims_conv


        self.conv1x1 = nn.Conv2d(input_dims_conv, output_dims_conv, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)

        self.feat_embed = nn.Sequential(
             linear(input_dims, output_dims, bias=False), 
        )

    def forward(self, x):

        B = x.shape[0]
        T = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        x = x.view(B*T, C, H, W)
        x = self.conv1x1(x)
        x = x.view(B*T*self.output_dims_conv, -1)
        x = self.feat_embed(x)
        x = x.view(B, T, -1)

        return x

class VoxelEncoder(nn.Module):
    """
    [8, 40, 100, 100, 100, 4]    B,T,100, 100, 100, 4->B,T,100,100,100,64->20, 20, 20, 64
    """

    def __init__(
        self,
        input_dims,
        output_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.voxel_patch_size = 5
        self.voxel_patch_stride = 5

        self.input_preprocess = Conv3DBlock(
            self.input_dims, self.output_dims//2, kernel_sizes=1, strides=1,
            norm=None, activation="lrelu",
        )

        self.patchify = Conv3DBlock(
            self.output_dims//2, self.output_dims,
            kernel_sizes=5, strides=2,
            norm=None, activation="lrelu")

        self.patchify_add1 = Conv3DBlock(
            self.output_dims, self.output_dims,
            kernel_sizes=3, strides=2,
            norm=None, activation="lrelu")

        self.patchify_add2 = Conv3DBlock(
            self.output_dims, self.output_dims,
            kernel_sizes=3, strides=2,
            norm=None, activation="lrelu")


    def forward(self, x):
        
        # Opt: encode only occupancy or occ with RGB
        x = x[:,:,:,:,:,0:1]
        B = x.shape[0]
        T = x.shape[1]
        x = x.view(B*T,*x.shape[2:])
        x = x.permute(0,4,1,2,3)

        x = self.input_preprocess(x)
        x = self.patchify(x)
        x = self.patchify_add1(x)
        x = self.patchify_add2(x)
        x = x.view(B, T, *x.shape[1:])
        return x