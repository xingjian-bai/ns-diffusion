import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

from trainer import Trainer1D
from diffuser import GaussianDiffusion1D
from one_box_dataset import AdaptedDataset, collate_adapted
from models import BiDenoise


parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')
parser.add_argument('--batch_size', default=128, type=int, help='size of batch of input to use')
parser.add_argument('--data-workers', type=int, default=1, help='number of workers to use for data loading')
parser.add_argument('--dataset', type=str, default='CLEVR_1O', help='dataset to use')
parser.add_argument('--wandb', default=False, action='store_true', help='use wandb')
parser.add_argument('--metric', type=str, default='visualize', help='metric to use')
parser.add_argument('--beta_schedule', type=str, default='cosine', help='beta schedule to use')
parser.add_argument('--name', type=str, help='name of experiment')

if __name__ == "__main__":
    FLAGS = parser.parse_args()
    

    if FLAGS.dataset == 'CLEVR_2O':
        dataset = AdaptedDataset()
        dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.data_workers, collate_fn=collate_adapted)
        model = BiDenoise()

        diffusion = GaussianDiffusion1D(
            model,
            seq_length = 32,
            beta_schedule = FLAGS.beta_schedule,
            objective = 'pred_noise',  # Alternative pred_x0
            timesteps = 100, # number of steps
            sampling_timesteps = 100, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
        )

        trainer = Trainer1D(
            diffusion,
            dataset_type = FLAGS.dataset,
            dataloader = dataloader,
            train_batch_size = FLAGS.batch_size,
            train_lr = 1e-4,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 1,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            data_workers = FLAGS.data_workers,
            amp = False,                      # turn on mixed precision
            metric = FLAGS.metric,
            wandb = FLAGS.wandb,
            name = FLAGS.name
        )
    
    else:
        raise ValueError('Dataset not supported')

    trainer.train()

