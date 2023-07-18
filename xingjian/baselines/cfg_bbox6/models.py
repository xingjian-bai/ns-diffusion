import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

from one_box_dataset import AdaptedDataset, collate_adapted


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


def swish(x):
    return x * torch.sigmoid(x)


class Denoise(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        h = 512

        fourier_dim, time_dim = 128, 128

        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.fc1 = nn.Linear(inp_dim + out_dim, h)

        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim)

        self.t_map_fc2 = nn.Linear(time_dim, 2 * h)
        self.t_map_fc3 = nn.Linear(time_dim, 2 * h)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, cond, out, t):
        """
        Args
            cond: (batch, inp_dim)
            out: (batch, out_dim)
            t: (batch, 1)
        """
        t_emb = self.time_mlp(t)

        fc2_gain, fc2_bias = torch.chunk(self.t_map_fc2(t_emb), 2, dim=-1)
        fc3_gain, fc3_bias = torch.chunk(self.t_map_fc3(t_emb), 2, dim=-1)

        # output shapes of cond and out
        # print(f"cond shape: {cond.shape}, out shape: {out.shape}")
        x = torch.cat((cond, out), dim=-1)
        h = swish(self.fc1(x))
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)

        output = self.fc4(h)

        return output

class BiDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_denoise = Denoise(4, 4)
        self.rel_denoise = Denoise(9, 8)
        self.out_dim = 4
    
    def forward(self, conditions, positions, t):
        """
        obj_cond: (batch, obj_num, 4)
        """
        obj_cond, rel_cond, rel_ids = conditions
        # brute force
        output_batch = torch.zeros_like(positions)

        batch_size = len(obj_cond)
        # print("before for loop")
        for i in range(batch_size): # batch
            obj_cond_datum = obj_cond[i]
            positions_datum = positions[i]
            t_datum = torch.stack([t[i] for _ in range(obj_cond_datum.shape[0])])


            obj_out = self.obj_denoise(obj_cond_datum, positions_datum, t_datum)
            output_batch[i] += obj_out


            rel_cond_datum = rel_cond[i]
            positions_datum = torch.stack([torch.cat([positions[i, a], positions[i, b]], dim=-1) for (a, b) in rel_ids[i]])
            t_datum = torch.stack([t[i] for _ in range(rel_cond_datum.shape[0])])
            rel_out = self.rel_denoise(rel_cond_datum, positions_datum, t_datum)

            for (j, rel_out_e) in enumerate(rel_out):
                (a, b) = rel_ids[i][j]
                output_batch[i, a] += rel_out_e[:4]
                output_batch[i, b] += rel_out_e[4:]
                
        return output_batch


class DiffusionWrapper(nn.Module):
    def __init__(self, ebm):
        super(DiffusionWrapper, self).__init__()
        self.ebm = ebm
        self.inp_dim = ebm.inp_dim
        self.out_dim = ebm.out_dim

    def forward(self, inp, opt_out, t, return_energy=False, return_both=False):
        opt_out.requires_grad_(True)
        opt_variable = torch.cat([inp, opt_out], dim=-1)

        energy = self.ebm(opt_variable, t)

        if return_energy:
            return energy

        opt_grad = torch.autograd.grad([energy.sum()], [opt_out], create_graph=True)[0]

        if return_both:
            return energy, opt_grad
        else:
            return opt_grad