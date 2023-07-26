# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from trainer import Trainer
from PIL import Image
from models import Unet
from diffuser import GaussianDiffusion
import argparse
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
# from image_adapted_dataset import
# from models import BiDenoise

parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')
# parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
# parser.add_argument('--data-workers', type=int, default=1, help='number of workers to use for data loading')
parser.add_argument('--dataset', type=str, default='CLEVR_1O', help='dataset to use')
parser.add_argument('--wandb', default=False, action='store_true', help='use wandb')
# parser.add_argument('--metric', type=str, default='visualize', help='metric to use')
# parser.add_argument('--beta_schedule', type=str, default='cosine', help='beta schedule to use')
parser.add_argument('--name', type=str, help='name of experiment')
# parser.add_argument('--no_obj', default=False, action='store_true', help='use wandb')
# parser.add_argument('--rel_only', default=False, action='store_true', help='use wandb')
# parser.add_argument('--obj_only', default=False, action='store_true', help='use wandb')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb_drawer = wandb.init(
                project="diffusion_image",
                name=f'{args.name}-{args.dataset}',
                save_code=True,
            )
    else:
        wandb_drawer = None


    from dataset_image_adopted import AdaptedDataset
    dataset = AdaptedDataset(dataset=args.dataset)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)

    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )
    print("in main, prepared model")

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )
    print("in main, prepared diffusion")


    trainer = Trainer(
        diffusion,
        dataset_type = args.dataset,
        dataloader = dataloader,
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        wandb_drawer = wandb_drawer,
        name = args.name,
    )
    print("in main, prepared trainer")

    trainer.train()

# import argparse
# import random
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data.dataset import Dataset
# from torch.utils.data import Dataset, DataLoader

# from trainer import Trainer1D
# from diffuser import GaussianDiffusion1D
# from one_box_dataset import AdaptedDataset, collate_adapted
# from models import BiDenoise


# parser = argparse.ArgumentParser(description='Train Diffusion Reasoning Model')
# parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
# parser.add_argument('--data-workers', type=int, default=1, help='number of workers to use for data loading')
# parser.add_argument('--dataset', type=str, default='CLEVR_2O', help='dataset to use')
# parser.add_argument('--wandb', default=False, action='store_true', help='use wandb')
# parser.add_argument('--metric', type=str, default='visualize', help='metric to use')
# parser.add_argument('--beta_schedule', type=str, default='cosine', help='beta schedule to use')
# parser.add_argument('--name', type=str, help='name of experiment')
# parser.add_argument('--no_obj', default=False, action='store_true', help='use wandb')
# parser.add_argument('--rel_only', default=False, action='store_true', help='use wandb')
# parser.add_argument('--obj_only', default=False, action='store_true', help='use wandb')


# def train_on(FLAGS, dataset_name, model, steps=1000000, wandb_drawer = None, global_step = 0):
#     dataset = AdaptedDataset(dataset=dataset_name)
#     dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.data_workers, collate_fn=collate_adapted)
        
#     diffusion = GaussianDiffusion1D(
#         model,
#         seq_length = 32,
#         beta_schedule = FLAGS.beta_schedule,
#         objective = 'pred_noise',  # Alternative pred_x0
#         timesteps = 100, # number of steps
#         sampling_timesteps = 100, # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
#         obj_num = dataset.obj_num,
#     )
#     trainer = Trainer1D(
#         diffusion,
#         dataset_type = dataset_name,
#         real_dataset_name = FLAGS.dataset,
#         dataloader = dataloader,
#         train_batch_size = FLAGS.batch_size,
#         train_lr = 1e-4,
#         train_num_steps = steps,         # total training steps
#         gradient_accumulate_every = 1,    # gradient accumulation steps
#         ema_decay = 0.995,                # exponential moving average decay
#         data_workers = FLAGS.data_workers,
#         amp = False,                      # turn on mixed precision
#         metric = FLAGS.metric,
#         wandb_drawer = wandb_drawer,
#         name = FLAGS.name,
#         starting_step = global_step
#     )
#     global_step += steps
#     trainer.train()
    

# if __name__ == "__main__":
#     FLAGS = parser.parse_args()

    

#     print(f"FLAGS: {FLAGS}")
#     model = BiDenoise(obj_is_special_rel = FLAGS.no_obj,
#         rel_only = FLAGS.rel_only,
#         obj_only = FLAGS.obj_only)
    
#     if FLAGS.wandb:
#         import wandb
#         wandb_drawer = wandb.init(
#                 project="diffusion_bbox",
#                 name=f'{FLAGS.name}--{model.BiDenoise_name()}--{FLAGS.dataset}',
#                 save_code=True,
#             )
#     else:
#         wandb_drawer = None
    
#     if FLAGS.dataset == 'CLEVR_2O' or FLAGS.dataset == 'CLEVR_3O' or FLAGS.dataset == 'CLEVR_4O':
#         train_on(FLAGS, FLAGS.dataset, model, wandb_drawer = wandb_drawer)
        
#     elif FLAGS.dataset == "mixed234":
#         global_step = 0
#         while True:
#             train_on(FLAGS, "CLEVR_2O", model,steps=10000, wandb_drawer = wandb_drawer, global_step = global_step)
#             global_step += 1000
#             train_on(FLAGS, "CLEVR_3O", model,steps=10000, wandb_drawer = wandb_drawer, global_step = global_step)
#             global_step += 1000
#             train_on(FLAGS, "CLEVR_4O", model,steps=10000, wandb_drawer = wandb_drawer, global_step = global_step)
#             global_step += 1000
#     else:
#         raise ValueError('Dataset not supported')
    
#     if FLAGS.wandb:
#         wandb_drawer.finish()
    
    
    

