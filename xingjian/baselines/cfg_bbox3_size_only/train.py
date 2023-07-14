#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/13/2023
#
# Distributed under terms of the MIT license.

import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

from diffusion_1d import Trainer1D, GaussianDiffusion1D


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
        t_emb = self.time_mlp(t)

        fc2_gain, fc2_bias = torch.chunk(self.t_map_fc2(t_emb), 2, dim=-1)
        fc3_gain, fc3_bias = torch.chunk(self.t_map_fc3(t_emb), 2, dim=-1)

        x = torch.cat((cond, out), dim=-1)
        h = swish(self.fc1(x))
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)

        output = self.fc4(h)

        return output


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


class SingleBoxDataset(Dataset):
    def __init__(self):
        self.ds = self.randomly_generate()

    def randomly_generate(self, n=10000):
        lists = []
        for i in range(n):
            is_big = random.random() < 0.5
            if is_big:
                size = (0.5, 0.5)
            else:
                size = (0.25, 0.25)

            # range: (-1, 1)
            x = random.random() * (2 - size[0]) - 1 + size[0] / 2
            y = random.random() * (2 - size[1]) - 1 + size[1] / 2
            lists.append((1 - is_big, is_big, x, y, size[0], size[1]))
        return torch.tensor(lists, dtype=torch.float32)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        fd = self.ds[index]
        return fd[:2], fd[2:]

class AdaptedDataset(Dataset):
    def __init__(self):
        import sys
        sys.path.append('../')
        from dataset import RelationalDataset1O
        self.data = RelationalDataset1O()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, bboxes, generated_prompt, raw_image, annotated_image_tensor = self.data[index]
        
        object = objects[0][3:4]
        label = bboxes[0]

        # print(f"in getitem: {object=}, {label=}")
        return object, label, generated_prompt, annotated_image_tensor

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))


parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')

parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--data-workers', type=int, default=8, help='number of workers to use for data loading')


if __name__ == "__main__":
    FLAGS = parser.parse_args()

    # dataset = SingleBoxDataset()
    dataset = AdaptedDataset()
    def collate_fn(batch):
        object_batch = []
        label_batch = []
        prompt_batch = []
        image_batch = []
        for (object, label, prompt, image) in batch:
            object_batch.append(object)
            label_batch.append(label)
            prompt_batch.append(prompt)
            image_batch.append(image)
        return torch.stack(object_batch), torch.stack(label_batch), prompt_batch, image_batch

    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.data_workers, collate_fn=collate_fn)
    metric = 'visualize'

    model = Denoise(1, 4)
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 32,
        objective = 'pred_noise',  # Alternative pred_x0
        timesteps = 100, # number of steps
        sampling_timesteps = 100, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
    )

    trainer = Trainer1D(
        diffusion,
        dataset_type = 'CLEVR_1O',
        dataloader = dataloader,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-4,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        data_workers = FLAGS.data_workers,
        amp = False,                      # turn on mixed precision
        metric = metric,
        wandb = True,
    )

    trainer.train()

