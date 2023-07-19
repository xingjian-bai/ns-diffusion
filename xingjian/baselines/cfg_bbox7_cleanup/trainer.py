import math
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from tabulate import tabulate
from PIL import Image, ImageDraw

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../')
from dataset import BoundingBox
from utils import *

from tqdm.auto import tqdm
import wandb
from diffuser import GaussianDiffusion1D


# def _custom_exception_hook(type, value, tb):
#     if hasattr(sys, 'ps1') or not sys.stderr.isatty():
#         # we are in interactive mode or we don't have a tty-like
#         # device, so we call the default hook
#         sys.__excepthook__(type, value, tb)
#     else:
#         import traceback, ipdb
#         # we are NOT in interactive mode, print the exception...
#         traceback.print_exception(type, value, tb)
#         # ...then start the debugger in post-mortem mode.
#         ipdb.post_mortem(tb)


# def hook_exception_ipdb():
#     """Add a hook to ipdb when an exception is raised."""
#     if not hasattr(_custom_exception_hook, 'origin_hook'):
#         _custom_exception_hook.origin_hook = sys.excepthook
#         sys.excepthook = _custom_exception_hook


# def unhook_exception_ipdb():
#     """Remove the hook to ipdb when an exception is raised."""
#     assert hasattr(_custom_exception_hook, 'origin_hook')
#     sys.excepthook = _custom_exception_hook.origin_hook

# hook_exception_ipdb()


# constants

# helpers functions



# gaussian diffusion trainer class



# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        *,
        dataset_type: str = 'CLEVR_2O',
        dataset: Dataset = None,
        dataloader: DataLoader = None,
        train_batch_size = 128,
        eval_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 100,
        num_samples = 25,
        data_workers = None,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        metric = 'mse',
        cond_mask = False,
        wandb = False,
        name = 'default',
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.wandb = wandb
        self.name = name

        self.accelerator.native_amp = amp
        self.dataset_type = dataset_type

        # model

        self.model = diffusion_model

        # Conditioning on mask

        self.cond_mask = cond_mask

        # sampling and training hyperparameters
        # self.out_dim = self.model.out_dim

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # Evaluation metric.
        self.metric = metric
        self.data_workers = data_workers

        if self.data_workers is None:
            self.data_workers = cpu_count()

        # dataset
        if dataset_type == 'CLEVR_1O' or dataset_type == 'CLEVR_2O' or dataset_type == 'CLEVR_3O' or dataset_type == 'CLEVR_4O':
            dl = dataloader
        elif dataset_type == 'simple':
            dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = False, num_workers = self.data_workers)
        else:
            print(f"attention! Dataset unorthodox!")
            raise NotImplementedError

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def get_train_name(self):
        return f'{self.name}--{self.model.model.BiDenoise_name()}--{self.dataset_type}'


    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'{self.get_train_name()}-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'{self.get_train_name()}-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if self.wandb:
            wandb.init(
                project="diffusion_bbox",
                name=self.get_train_name(),
                save_code=True,
            )

        end_time = time.time()
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:
            
            tqdm_update_freq = 100 

            while self.step < self.train_num_steps:
                total_loss = 0.

                end_tiem = time.time()
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.dataset_type == 'CLEVR_1O':
                        inp, label, prompts, images = data
                        inp, label = inp.float().to(device), label.float().to(device)
                        # print(f"all data obtained: {inp.shape}, {label.shape}, {len(prompts)}, {len(images)}, {images[0].shape}")
                        mask = None
                    elif self.dataset_type == 'CLEVR_2O' or self.dataset_type == 'CLEVR_3O' or self.dataset_type == 'CLEVR_4O':
                        objects, relations, label, prompts, images, relations_ids = data
                        inp = (objects, relations, relations_ids)
                        mask = None
                    # elif self.dataset_type == 'simple':
                    #     inp, label = data
                    #     mask = None
                    #     prompts = None
                    #     images = []
                    #     for lab in label:
                    #         images.append(BoundingBox(lab).draw(color = (255, 0, 0)))
                    #     # print("check labels: ", label)
                    else:
                        raise NotImplementedError

                    data_time = time.time() - end_time; end_time = time.time()

                    with self.accelerator.autocast():
                        loss, (loss_denoise, loss_energy) = self.model(inp, label, mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                
                if self.step % tqdm_update_freq == 0:  # update progress bar every `update_freq` steps
                    pbar.set_description(f'loss: {total_loss:.4f} loss_denoise: {loss_denoise:.4f} loss_energy: {loss_energy:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')
                    pbar.update(tqdm_update_freq)
                if self.wandb:
                    wandb.log({"loss": total_loss, "step": self.step}, step = self.step)
                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update ()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()
                        print(f"In eval with step={self.step}")
                        inp = (inp[0][:self.eval_batch_size], inp[1][:self.eval_batch_size], inp[2][:self.eval_batch_size]) if self.dataset_type != 'CLEVR_1O' else inp[:self.eval_batch_size]
                        label = label[:self.eval_batch_size]
                        mask = mask[:self.eval_batch_size] if mask is not None else None
                        prompts = prompts[:self.eval_batch_size] if prompts is not None else None
                        images = images[:self.eval_batch_size]

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(
                                inp, label, mask,
                                batch_size=self.eval_batch_size), range(1)))

                        all_samples = torch.cat(all_samples_list, dim = 0)
                        mse_error = (all_samples - label).pow(2).mean()

                        if self.metric == 'visualize':
                            rows, bboxes = [], []
                            for i in range(all_samples.size(0)):
                                obj_cond, rel_cond, rel_id_cond = inp
                                this_out = all_samples[i]
                                # this_out is actually a list of boxes!
                                bbox = []
                                for e in this_out:
                                    bbox.append(BoundingBox(e.tolist()))
                                bboxes.append(bbox)
                                rows.append(((obj_cond[i].tolist(), rel_cond[i].tolist(), rel_id_cond[i].tolist()), this_out.tolist(), label[i].tolist()))
                            print(tabulate(rows))

                            
                            if self.wandb:
                                for i, image in enumerate(images):
                                    if isinstance(image, torch.Tensor):
                                        image = tensor_to_pil(image)
                                    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
                                    for j, bbox in enumerate(bboxes[i]):
                                        image = bbox.draw(image, color=colours[j % len(colours)])
                                    images[i] = image
                                    if self.dataset_type == 'CLEVR_1O':
                                        size = "small" if inp[i][3] == 0 else "large"
                                        print(f"MOD output: {size=}, bbox_size=({label[i][2]}, {label[i][3]})")
                                wandb.log({"images": [wandb.Image(image) for image in images]}, step = self.step)
                        else:
                            raise NotImplementedError
                        # if self.metric == 'mse':
                        #     all_samples = torch.cat(all_samples_list, dim = 0)
                        #     mse_error = (all_samples - label).pow(2).mean()
                        #     print("mse_error: ", mse_error)
                        # elif self.metric == 'bce':
                        #     assert len(all_samples_list) == 1

                        #     summary = binary_classification_accuracy_4(all_samples_list[0], label)
                        #     rows = [[k, v] for k, v in summary.items()]
                        #     print(tabulate(rows))

                        self.save(milestone)

                

        accelerator.print('training complete')

        if self.wandb:
            wandb.finish()

    # def eval(self, single_image_eval, data_loader, show_off_mode = False, repeat = 8, wandb_drawer = None):
    #     accelerator = self.accelerator
    #     device = accelerator.device

    #     scores = []
            
    #     #make dataset into dataloader
    #     for (data_id, data) in enumerate(data_loader):
    #         objects, relations, relations_ids, labels, images = data
    #         mask = None
    #         inp = (objects, relations, relations_ids)

    #         objects = [obj.to(device) for obj in objects]
    #         relations = [relation.to(device) for relation in relations]
    #         labels = [l.to(device) for l in labels]
    #         images = [image.to(device) for image in images]
    #         relations_ids = [relation_id.to(device) for relation_id in relations_ids]
            

    #         if show_off_mode:
    #             objects = [item for item in objects for _ in range(repeat)][:self.eval_batch_size]
    #             relations = [item for item in relations for _ in range(repeat)][:self.eval_batch_size]
    #             relations_ids = [item for item in relations_ids for _ in range(repeat)][:self.eval_batch_size]
    #             labels = [item for item in labels for _ in range(repeat)][:self.eval_batch_size]
    #             images = [item for item in images for _ in range(repeat)][:self.eval_batch_size]

    #         inp = (objects, relations, relations_ids)
    #         print("dealing with data_id: ", data_id)
    #         self.ema.ema_model.eval()
    #         with torch.no_grad():
    #             all_samples_list = list(map(lambda n: self.ema.ema_model.sample(
    #                 inp, labels, mask,
    #                 batch_size=self.eval_batch_size), range(1)))

    #         all_samples = torch.cat(all_samples_list, dim = 0)

    #         # print("shape of all samples: ", all_samples.shape)
    #         # print("len of obj_cond: ", len(obj_cond))
    #         # print("len of rel_cond: ", len(rel_cond))
    #         # print("len of rel_id_cond: ", len(rel_id_cond))

    #         for i in range(len(relations)):
    #             score = single_image_eval(all_samples[i], relations[i], relations_ids[i])
    #             scores.append(score)
            
    #         if show_off_mode:
    #             bboxes = [[BoundingBox(e.tolist()) for e in this_out] for this_out in all_samples]
    #             for i, image in enumerate(images):
    #                 if isinstance(image, torch.Tensor):
    #                     image = tensor_to_pil(image)
    #                 colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # red, green, blue, yellow, cyan, magenta
    #                 for j, bbox in enumerate(bboxes[i]):
    #                     image = bbox.draw(image, color=colours[j % len(colours)])
    #                 images[i] = image
    #             if wandb_drawer is not None:
    #                 print("wandb log images")
    #                 wandb_drawer.log({"images": [wandb.Image(image) for image in images]}, step = data_id)
    #                 # wandb.log({"images": [wandb.Image(image) for image in images]}, step = data_id)
    #                 break
    #     if not show_off_mode:
    #         avg_score = sum(scores) / len(scores)
    #         print("acc: ", avg_score)
    #         if wandb_drawer is not None:
    #             wandb.log({"acc": avg_score}, step = self.step)
    #         return avg_score
                
        




