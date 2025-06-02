"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file contains classes of denoising models for HTP diffusion and Egomotion diffusion.
"""

import torch
from mambapy.mamba import Mamba, MambaConfig, Mamba_homo

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

from einops import rearrange

def get_pad_mask(seq, pad_idx=0):
    if seq.dim() != 2:
        raise ValueError("<seq> has to be a 2-dimensional tensor!")
    if not isinstance(pad_idx, int):
        raise TypeError("<pad_index> has to be an int!")

    return (seq != pad_idx).unsqueeze(1)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
           self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attention = ScaledDotProductAttention(temperature=qk_scale or head_dim ** 0.5)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, mask=None):
        B, Nq, Nk, Nv, C = q.shape[0], q.shape[1], k.shape[1], v.shape[1], q.shape[2]
        if self.with_qkv:
            q = self.proj_q(q).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = self.proj_k(k).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = self.proj_v(v).reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
            v = v.reshape(B, Nv, self.num_heads, C // self.num_heads).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attn = self.attention(q, k, v, mask=mask)
        x = x.transpose(1, 2).reshape(B, Nq, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class TransformerNetModel(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout


        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.enc_dec_attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, emb_motion, memory_mask=None, trg_mask=None):
        tgt_2 = self.norm1(tgt)
        tgt = tgt + self.drop_path(self.enc_dec_attn(q=self.norm2(tgt), k=emb_motion, v=emb_motion, mask=None))  
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return tgt

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, N, mask=None):

        if mask is not None:
            src_mask = rearrange(mask, 'b n t -> b (n t)', b=B, n=N, t=T)
            src_mask = get_pad_mask(src_mask, 0)
        else:
            src_mask = None
        x2 = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x2, k=x2, v=x2, mask=src_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class HOIMamba_homo(nn.Module):
    def __init__(
        self,
        d_model=1024,
        input_dims=1024,
        output_dims=1024,
        hidden_t_dim=1024,
        n_layers=1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
        self.mamba_encoder = Mamba_homo(config)

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = d_model

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        self.register_buffer("position_ids", torch.arange(1000).expand((1, -1)))
        self.position_embeddings = nn.Embedding(1000, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
    

        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.Tanh(), nn.Linear(self.hidden_size, self.output_dims))


    def forward(self, x_r, timesteps, motion_feat_encoded, valid_mask=None):

        valid_mask = None # deprecated
        if valid_mask is not None:
            valid_mask = get_pad_mask(valid_mask, pad_idx=0)

        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        emb_xr = x_r
        seq_length = x_r.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_inputs_r = self.position_embeddings(position_ids) + emb_xr + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)
        motion_length = motion_feat_encoded.size(1)
        position_ids_motion = self.position_ids[:, : motion_length]
        emb_motion = self.position_embeddings(position_ids_motion) + motion_feat_encoded
        emb_motion = self.LayerNorm(emb_motion)
        emb_inputs_r = self.mamba_encoder(emb_inputs_r, emb_motion)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        if self.output_dims != self.hidden_size:
            h_r = self.output_down_proj(emb_inputs_r)
        else:
            h_r = emb_inputs_r
        h_r = h_r.type(x_r.dtype)
        return h_r


class HOIMambaTransformer(nn.Module):
    def __init__(
        self,
        d_model=1024,
        input_dims=1024,
        output_dims=1024,
        hidden_t_dim=1024,
        n_layers=1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
        self.mamba_encoder = Mamba(config)
        self.mamba_encoder_add = Mamba(config)

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = d_model // 2

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        self.register_buffer("position_ids", torch.arange(1000).expand((1, -1)))
        self.position_embeddings = nn.Embedding(1000, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
    
        self.dropout_value=0.1
        drop_path_rate = 0.1
        self.depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.denoised_transformer = nn.ModuleList([DecoderBlock(dim=self.hidden_size, num_heads=4, mlp_ratio=4, qkv_bias=False, qk_scale=None,
        drop=self.dropout_value, attn_drop=0., drop_path=dpr[i])
        for i in range(self.depth)])
        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                nn.Tanh(), nn.Linear(self.hidden_size, self.output_dims))


    def forward(self, x_r, timesteps, motion_feat_encoded_raw, valid_mask=None):

        motion_feat_encoded = motion_feat_encoded_raw[0]
        occ_feat_encoded =  motion_feat_encoded_raw[1]

        valid_mask = None # deprecated
        if valid_mask is not None:
            valid_mask = get_pad_mask(valid_mask, pad_idx=0)

        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim)) 
        emb_xr = x_r

        seq_length = x_r.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_inputs_r = self.position_embeddings(position_ids) + emb_xr + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        motion_length = motion_feat_encoded.size(1)
        position_ids_motion = self.position_ids[:, : motion_length]
        emb_motion = self.position_embeddings(position_ids_motion) + motion_feat_encoded
        emb_motion = self.LayerNorm(emb_motion)

        occ_length = occ_feat_encoded.size(1)
        position_ids_occ = self.position_ids[:, : occ_length]
        emb_occ = self.position_embeddings(position_ids_occ) + occ_feat_encoded
        emb_occ = self.LayerNorm(emb_occ)

        emb_inputs_r = self.mamba_encoder(emb_inputs_r, emb_motion)
        emb_inputs_r = emb_inputs_r[:,:,:1024]
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        for blk in self.denoised_transformer:
            emb_inputs_r = emb_inputs_r + blk(emb_inputs_r, emb_occ, memory_mask=None, trg_mask=valid_mask)

        if self.output_dims != self.hidden_size:
            h_r = self.output_down_proj(emb_inputs_r)
        else:
            h_r = emb_inputs_r
        h_r = h_r.type(x_r.dtype)
        return h_r