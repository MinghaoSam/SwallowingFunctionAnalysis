from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


logger = logging.getLogger(__name__)


class Position_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,patchsize, img_size, batch_size):
        super().__init__()
        img_size = _pair(img_size)  # _pair is a function that returns a tuple of 2 elements
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14*14=196

        self.patch_embeddings = Conv2d(in_channels=batch_size,
                                       out_channels=batch_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, batch_size))  # trainable parameter
        self.dropout = Dropout(0.5)

    def forward(self, x):
        if x is None:  # x -> (C, B, H, W)
            return None
        x = self.patch_embeddings(x)  # (C, B=8, n_patches^(1/2), n_patches^(1/2)) hidden = in_channels
        x = x.flatten(2)  # (C, B, n_patches) start_dim=2
        x = x.transpose(-1, -2)  # (C, n_patches, B)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, hidden, n_patch = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # sqrt(196) = 14
        # x = x.permute(0, 2, 1)  # (B, n_patch, hidden) -> (B, hidden, n_patch)
        x = x.contiguous().view(B, hidden, h, w)  # (B, hidden, n_patch) -> (B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x) # upsample is bilinear interpolation
        out = self.conv(x)  # 1x1 conv
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    '''
    Temporal Attention with Relative Position Bias
    '''
    def __init__(self,batch_size=8):
        super(Attention_org, self).__init__()
        self.KV_size = batch_size
        self.batch_size = batch_size
        self.num_attention_heads = 4  # 4

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(self.num_attention_heads):  # num_heads = 4
            query = nn.Linear(batch_size, batch_size, bias=False)  # linear transformation, Q
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)  # 8->8, K
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)  # 8->8, V
            self.query.append(copy.deepcopy(query))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out = nn.Linear(batch_size, batch_size, bias=False)
        self.attn_dropout = Dropout(0.5)
        self.proj_dropout = Dropout(0.5)

        coords = torch.stack(torch.meshgrid([torch.arange(batch_size), torch.arange(batch_size)]))
        relative_coords = coords[0] - coords[1] + batch_size - 1
        relative_coords = relative_coords.view(-1, batch_size)  # [B, B]  8*8
        self.register_buffer('relative_position_index', relative_coords)  # register buffer to save it in the model, trainable=False
        self.relative_position_table = nn.Parameter(
            torch.zeros(2 * batch_size - 1, self.num_attention_heads))  # trainable parameter  [15, 4*8*8]  [2b-1, num_heads * b * b]



    def forward(self, emb):
        multi_head_Q_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb is not None:
            for query in self.query:
                Q = query(emb.permute(0, 2, 1))  # (196, B, C) -> (196, C, B), Q
                Q = Q.permute(0, 2, 1)
                multi_head_Q_list.append(Q)
        for key in self.key:
            K = key(emb.permute(0, 2, 1))  # (196, B, C) -> (196, B, C), K
            K = K.permute(0, 2, 1)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb.permute(0, 2, 1))  # (196, B, C) -> (196, B, C), V
            V = V.permute(0, 2, 1)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q = torch.stack(multi_head_Q_list, dim=1) if emb is not None else None  # stack of 4 heads alone a new dimension, (196, 4, B, C)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)  # (196, 4, B, C)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)  # (196, 4, B, C)

        multi_head_K = multi_head_K.transpose(-1, -2) if emb is not None else None  # (196, 4, B, C) -> (196, 4, C, B)

        attention_scores = torch.matmul(multi_head_Q, multi_head_K) if emb is not None else None  # (196, 4, B, C) * (196, 4, C, B) -> (196, 4, B, B)

        attention_scores = attention_scores / math.sqrt(self.KV_size) if emb is not None else None
        # attention_scores_flatten = attention_scores.flatten(1)
        # print(f'attention_scores_flatten: {attention_scores_flatten.shape}')

        relative_temporal_bias = self.relative_position_table[self.relative_position_index.view(-1)].view(self.batch_size,self.batch_size,-1)  # (B, B, 4)
        relative_temporal_bias = relative_temporal_bias.permute(2, 0, 1).unsqueeze(0)  # (1, 4, B, B)

        # print(f'rela_pos: {relative_temporal_bias.shape}')

        attention_scores = attention_scores + relative_temporal_bias  # (196, 4, B, B)

        attention_scores = attention_scores.view(-1, self.num_attention_heads, self.KV_size, self.KV_size)  # (196, 4, B, B)

        attention_probs = self.softmax(self.psi(attention_scores)) if emb is not None else None  # attention_probs = softmax(attention_scores/ sqrt(d_k))
        # print(attention_probs4.size())

        attention_probs = self.attn_dropout(attention_probs) if emb is not None else None

        # multi_head_V = multi_head_V.transpose(-1, -2)  # (196, 4, C, B) -> (196, 4, B, C)
        context_layer = torch.matmul(attention_probs, multi_head_V) if emb is not None else None  # (196, 4, B, B) * (196, 4, B, C) -> (196, 4, B, C)

        context_layer = context_layer.permute(0, 2, 3, 1).contiguous() if emb is not None else None  # (196, 4, B, C) -> (196, B, C, 4), deep copy
        context_layer = context_layer.mean(dim=3) if emb is not None else None  # (196, B, C, 4) -> (196, B, C) mean along the head dim

        O1 = self.out(context_layer.permute(0, 2, 1)) if emb is not None else None  # out is a linear layer, output dim = input dim
        O1 = O1.permute(0, 2, 1)
        O1 = self.proj_dropout(O1) if emb is not None else None

        return O1




class Mlp(nn.Module):
    def __init__(self,in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.5)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    '''
    Transformer Block
    '''
    def __init__(self, batch_size=8):
        super(Block_ViT, self).__init__()
        expand_ratio = 4  # config sets the value to 4
        self.attn_norm = LayerNorm(batch_size,eps=1e-6)  # 8
        self.temporal_attn = Attention_org(batch_size)
        self.ffn_norm = LayerNorm(batch_size,eps=1e-6)
        self.ffn = Mlp(batch_size,batch_size*expand_ratio)


    def forward(self, emb):
        org = emb  # (196, B, C)

        cx = self.attn_norm(emb.permute(0, 2, 1)) if emb is not None else None  # (196, B, C)
        cx = cx.permute(0, 2, 1)
        cx = self.temporal_attn(cx)  # (196, B, C) -> (196, B, C)
        cx = org + cx if emb is not None else None  # (196, B, C)

        org = cx
        x = self.ffn_norm(cx.permute(0, 2, 1)) if emb is not None else None  # layer norm

        x = self.ffn(x) if emb is not None else None  # (196, B, C)

        x = x.permute(0, 2, 1)
        x = x + org if emb is not None else None  # residual connection

        return x


class Encoder(nn.Module):
    '''
    Encoder is made up of self-attn and feed forward (defined below)
    '''
    def __init__(self, batch_size=8):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(batch_size,eps=1e-6)
        for _ in range(4):  # num_layers = 4
            layer = Block_ViT(batch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb):
        attn_weights = []
        for layer_block in self.layer:
            emb = layer_block(emb)  # (196, B, C) -> (196, B, C)
        emb = self.encoder_norm(emb.permute(0, 2, 1)) if emb is not None else None
        emb = emb.permute(0, 2, 1)
        return emb


class TemporalTransformer(nn.Module):
    def __init__(self, img_size, n_channels, patchSize=16):
        super().__init__()
        self.embeddings = Position_Embeddings(patchSize, img_size=img_size, batch_size=8)
        self.encoder = Encoder(8)
        self.reconstruct = Reconstruct(n_channels, n_channels, kernel_size=1,scale_factor=(patchSize, patchSize))

    def forward(self,en):
        # print(en.shape)

        en_p = en.permute(1, 0, 2, 3) # (B, C, H, W) -> (C, B, H, W)
        # print(en.shape)
        emb = self.embeddings(en_p)  # (C, B, H, W) -> (C, 196, B)
        emb_p = emb.permute(1, 2, 0) # (C, 196, B) -> (196, B, C)
        # print(emb.shape)


        encoded = self.encoder(emb_p)  # (196, B, C) -> (196, B, C)
        # print(encoded.shape)
        encoded = encoded.permute(1, 2, 0)  # (196, B, C) -> (B, C, 196)
        # print(encoded.shape)
        x = self.reconstruct(encoded) if en is not None else None  # (B, C, 196) -> (B, C, 224, 224)


        x = x + en  if en is not None else None


        return x