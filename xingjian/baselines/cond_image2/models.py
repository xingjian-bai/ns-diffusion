import sys
sys.path.append('../')
from utils import *
from dataset_clevr_ryan import BoundingBox

########################################################################################
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

# from one_box_dataset import AdaptedDataset, collate_adapted

import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = True
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        """
        Args:
            x: image tensor of shape [batch_size, channels, height, width]
            time: time embedding tensor of shape [batch_size, time_dim]
        """
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class Unet_conditional(Unet):
    def __init__(self, cond_dim, dim = 64):
        """
            dim: the dim inside Unet
            cond_dim: the dim of the condition
        """
        super().__init__(dim)
        self.cond_dim = cond_dim
        time_dim = dim * 4
        self.cond_mlp = nn.Sequential(
            # SinusoidalPosEmb(dim),
            nn.Linear(cond_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.no_cond = (cond_dim == 0)



    def forward(self, cond, x, time, x_self_cond=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'
        
        # print("shape of x: ", x.shape)
        # print("shape of time: ", time.shape)
        t = self.time_mlp(time)
        # print("shape of t: ", t.shape)
        
        if not self.no_cond:
            # print("shape of cond: ", cond.shape)
            cond_out = self.cond_mlp(cond.float())
            # print(f"shape of cond_out: {cond_out.shape}")
            t += cond_out

        # below is normal unet forward
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
        # output is pred noise



class BiCondUnet(nn.Module):
    def __init__(self, 
                 composition_type = "add",
                 no_rel = False, no_obj = False, no_global = False,
                 mask_bbox = False,):
        super().__init__()
        self.no_rel = no_rel
        self.no_obj = no_obj
        self.no_global = no_global
        self.mask_bbox = mask_bbox
        self.rel_denoise = Unet_conditional(4 + 4 + 4 + 4 + 1)
        self.obj_denoise = Unet_conditional(4 + 4)
        self.global_denoise = Unet_conditional(0)
        self.composition_type = composition_type

        print("initialized BiCondUnet")
    
    def name(self):
        return f"Bi-CondUnet{self.composition_type}" + \
                "[" + \
                ("r" if not self.no_rel else "") + \
                ("o" if not self.no_obj else "") + \
                ("g" if not self.no_global else "") + \
                "]" + \
                ("" if not self.mask_bbox else "[mask]")
    
    def forward(self, conds, x, time, x_self_cond=None):
        """
        """
        (objects, relations, relations_ids, obj_masks, rel_masks) = conds
        # print(f"in forward, obj_masks has shape {len(obj_masks)} * {obj_masks[0].shape}")
        # print(f"in forward, rel_masks has shape {len(rel_masks)} * {rel_masks[0].shape}")

        batch_size = len(objects)
        # print("obj_num:", obj_num)
        output_batch = torch.zeros_like(x)

        if self.composition_type != "add":
            raise NotImplementedError
        
        if self.composition_type == "cfg":
            raise NotImplementedError


        if not self.no_obj:
            for batch_id in range(batch_size):
                object_all = objects[batch_id]
                xs = x[batch_id].unsqueeze(0).repeat(len(objects[batch_id]), 1, 1, 1)
                times = time[batch_id].repeat(len(objects[batch_id]))

                obj_outs = self.obj_denoise(object_all, xs, times)
                
                if self.mask_bbox:
                    mask = obj_masks[batch_id].unsqueeze(1) # shape obj_num, 1, 128, 128
                    # print("shape of obj mask: ", mask.shape)
                    obj_outs *= mask.expand_as(obj_outs)
                
                output_batch[batch_id] += obj_outs.sum(dim=0)

                # for obj_id in range(len(objects[batch_id])):
                #     object = objects[batch_id][obj_id]
                #     obj_out = self.obj_denoise(object, x[batch_id], time[batch_id])
                    
                #     if self.mask_bbox:
                #         mask = obj_masks[batch_id][obj_id].unsqueeze(0)
                #         print("shape of mask: ", mask.shape, ", which should be 1 128 128")
                #         obj_out *= mask.expand_as(obj_out)
                    
                #     output_batch[batch_id] += obj_out
        
        if not self.no_rel:
            for batch_id in range(batch_size):
                relation_all = relations[batch_id]
                xs = x[batch_id].unsqueeze(0).repeat(len(relations[batch_id]), 1, 1, 1)
                times = time[batch_id].repeat(len(relations[batch_id]))

                
                object_as = objects[batch_id][relations_ids[batch_id][:, 0]]
                object_bs = objects[batch_id][relations_ids[batch_id][:, 1]]
                relation_tensors = relations[batch_id][:, -1].unsqueeze(-1)
                # print("shapes of object_as, object_bs, relation_tensors:", object_as.shape, object_bs.shape, relation_tensors.shape)
                relation_combined = torch.cat((object_as, object_bs, relation_tensors), dim=1)
                # print("shape of relation_combined:", relation_combined.shape)
                # print("shape of xs:", xs.shape, "shape of times:", times.shape)
                rel_outs = self.rel_denoise(relation_combined, xs, times)

                if self.mask_bbox:
                    mask = rel_masks[batch_id].unsqueeze(1) # shape rel_num, 1, 128, 128
                    # print("shape of rel mask: ", mask.shape)
                    mask_str = mask_to_string(mask[0][0])
                    # print("example of rel mask:\n", mask_str)
                    rel_outs *= mask.expand_as(rel_outs)
                
                output_batch[batch_id] += rel_outs.sum(dim=0)

                # for rel_id in range(len(relations[batch_id])):
                #     relation = relations[batch_id][rel_id]
                #     (a, b) = relations_ids[batch_id][rel_id]
                #     object_a = objects[batch_id, a]
                #     object_b = objects[batch_id, b]
                #     relation_tensor = torch.tensor([relation]).to(x.device)
                #     # concat the two bbox
                #     print("pre combine,", object_a, object_b, relation_tensor)
                #     relation_combined = torch.cat((object_a, object_b, relation_tensor), dim=0)
                #     print("shape of the combined bbox:", relation_combined.shape, "shape of x[]:", x[batch_id].shape, "shape of time[]:", time[batch_id].shape)

                #     rel_out = self.rel_denoise(relation_combined, x[batch_id], time[batch_id])
                    
                #     if self.mask_bbox:
                #         mask = rel_masks[batch_id][rel_id].unsqueeze(0)
                #         print("shape of mask: ", mask.shape, ", which should be 1 128 128")
                #         rel_out *= mask.expand_as(rel_out)
                    
                #     output_batch[batch_id] += rel_out
        
        if not self.no_global:
            global_out = self.global_denoise(None, x, time)
            output_batch += global_out

        return output_batch
