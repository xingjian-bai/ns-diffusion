import sys
sys.path.append('../')
from utils import *
########################################################################################
import math
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path
from tabulate import tabulate
from PIL import Image, ImageDraw

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('../')
from dataset_clevr_ryan import *
from utils import *

from tqdm.auto import tqdm
import wandb
import random



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

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        # dataset_folder = None,
        dataset_type: str = 'CLEVR_2O',
        real_dataset_name = None,
        dataset: Dataset = None,
        dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
        train_batch_size = 64,
        eval_batch_size = 8,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 1000000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 250,
        save_model_every = 5000,
        num_samples = 25,
        results_folder = './results',
        weights_folder = './weights',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        wandb_drawer = None,
        name = 'default-trainer',
        starting_step = 0,
        obj_num = None
    ):
        super().__init__()

        if obj_num == None:
            self.obj_num = 2
        else:
            self.obj_num = obj_num

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        
        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_model_every = save_model_every

        self.batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        ####################################################################################################
        if real_dataset_name is None:
            real_dataset_name = dataset_type
        self.real_dataset_name = real_dataset_name

        self.wandb_drawer = wandb_drawer
        self.name = name
        self.dataset_type = dataset_type
        if dataset_type == 'CLEVR_1O' or dataset_type == 'CLEVR_2O' or dataset_type == 'CLEVR_3O' or dataset_type == 'CLEVR_4O':
            print(f"in trainer, loaded {dataset_type}")
            dl = dataloader
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
            
            eval_dl = eval_dataloader
            eval_dl = self.accelerator.prepare(eval_dl)
            self.eval_dl = cycle(eval_dl)
        else:
            print(f"attention! Dataset unorthodox! cannot performan training then")
            raise NotImplementedError
        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        # dl = self.accelerator.prepare(dl)
        # self.dl = cycle(dl)
        self.step = starting_step
        ####################################################################################################

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.weights_folder = Path(weights_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.weights_folder.mkdir(exist_ok = True)


        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            print("fid is not suitable in this task")
            raise NotImplementedError
            if not self.model.is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        print(f"in trainer, finished init")

    @property
    def device(self):
        return self.accelerator.device

    def get_train_name(self):
        return f'{self.name}-{self.real_dataset_name}'


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

        torch.save(data, str(self.weights_folder / f'{self.get_train_name()}-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.weights_folder / f'{self.get_train_name()}-{milestone}.pt'), map_location=device)

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
        end_time = time.time()

        step_limit = self.train_num_steps + self.step
        with tqdm(initial = self.step, total = step_limit, disable = not accelerator.is_main_process, dynamic_ncols = True) as pbar:

            tqdm_update_freq = 1 

            while self.step < step_limit:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)

                    if self.dataset_type == 'CLEVR_1O' or self.dataset_type == 'CLEVR_2O' or self.dataset_type == 'CLEVR_3O' or self.dataset_type == 'CLEVR_4O':
                        # in image generation, all needed are images, objects, bboxes, relations, relations_ids
                        images, objects, bboxes, relations, relations_ids, obj_masks, rel_masks = data
                        images = images.to(device)
                        objects = [object.to(device) for object in objects]
                        bboxes = [bbox.to(device) for bbox in bboxes]
                        relations = [relation.to(device) for relation in relations]
                        relations_ids = [relation_id.to(device) for relation_id in relations_ids]
                        obj_masks = [obj_mask.to(device) for obj_mask in obj_masks]
                        rel_masks = [rel_mask.to(device) for rel_mask in rel_masks]

                        objects_combined = [torch.cat((object, bbox), dim=-1) for object, bbox in zip(objects, bboxes)]
                    else:
                        raise NotImplementedError
                    
                    data_time = time.time() - end_time; end_time = time.time()

                    with self.accelerator.autocast():
                        conds = (objects_combined, relations, relations_ids, obj_masks, rel_masks)
                        loss = self.model(conds, images)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                nn_time = time.time() - end_time; end_time = time.time()
                
                if self.step % tqdm_update_freq == 0:  # update progress bar every `update_freq` steps
                    pbar.set_description(f'loss: {total_loss:.4f} data_time: {data_time:.2f} nn_time: {nn_time:.2f}')
                    pbar.update(tqdm_update_freq)
                if self.wandb_drawer:
                    self.wandb_drawer.log({"loss": total_loss, "step": self.step}, step = self.step)
                self.step += 1


                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step % self.save_model_every == 0:
                        # save model
                        print("saving the weights...")
                        self.save(self.step)

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        print(f"in trainer, entered eval with step={self.step}")

                        def visualize(conds, text = "generated", images = None, bboxes = None):
                            print(f"visualizing {text} images")
                            combined_objects, relations, relations_ids, obj_masks, rel_masks = conds
                            combined_objects = combined_objects[:self.eval_batch_size]
                            relations = relations[:self.eval_batch_size]
                            relations_ids = relations_ids[:self.eval_batch_size]
                            obj_masks = obj_masks[:self.eval_batch_size]
                            rel_masks = rel_masks[:self.eval_batch_size]
                            conds = (combined_objects, relations, relations_ids, obj_masks, rel_masks)
                            if not self.wandb_drawer:
                                return
                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                all_sampled_images = self.ema.ema_model.sample(conds, batch_size=self.eval_batch_size)
                            # MUST convert to PIL, to avoid normalization
                            visualized_images = []
                            for (i, image) in enumerate(all_sampled_images):
                                image = tensor_to_pil(image)

                                from dataset_clevr_ryan import draw_scene_graph
                                object = combined_objects[i]
                                # print("needed: ", object[:, :-4])
                                scene_graph = draw_scene_graph(object[:, :-4].long(), relations[i], relations_ids[i])
                                image = combine_images(image, scene_graph)
                                if images is not None:
                                    original_image = tensor_to_pil(images[i])
                                else:
                                    original_image = Image.new("RGB", (128, 128), (224, 224, 224))
                                if bboxes is not None:
                                    for bbox_data in bboxes[i]:
                                        bbox = BoundingBox(bbox_data)
                                        original_image = bbox.draw(original_image)
                                image = combine_images(image, original_image)
                                visualized_images.append(image)

                            self.wandb_drawer.log({f"{text}": [wandb.Image(image) for image in visualized_images]}, step = self.step)
                        
                        visualize(conds, "generated", images = images, bboxes = bboxes)

                        eval_data = next(self.eval_dl)
                        eval_images, eval_objects, eval_bboxes, eval_relations, eval_relations_ids, eval_obj_masks, eval_rel_masks = eval_data
                        eval_images = eval_images.to(device)
                        eval_objects = [object.to(device) for object in eval_objects]
                        eval_bboxes = [bbox.to(device) for bbox in eval_bboxes]
                        eval_relations = [relation.to(device) for relation in eval_relations]
                        eval_relations_ids = [relation_id.to(device) for relation_id in eval_relations_ids]
                        eval_obj_masks = [obj_mask.to(device) for obj_mask in eval_obj_masks]
                        eval_rel_masks = [rel_mask.to(device) for rel_mask in eval_rel_masks]

                        eval_objects_combined = [torch.cat((object, bbox), dim=-1) for object, bbox in zip(eval_objects, eval_bboxes)]
                        eval_cond = (eval_objects_combined, eval_relations, eval_relations_ids, eval_obj_masks, eval_rel_masks)
                        visualize(eval_cond, "eval", images = eval_images, bboxes = eval_bboxes)

                        ################RANDOM################
                        
                        # random_combined_objects = torch.zeros((self.eval_batch_size, self.obj_num, 8))
                        # random_relations = []
                        # random_relations_ids = []
                        if self.obj_num == 2:
                            rel_num = 1 # TODO
                        else:
                            rel_num = 0
                        for i in range(self.eval_batch_size):
                            for j in range(self.obj_num):
                                bbox = gen_rand_bbox()
                                eval_bboxes[i][j] = torch.tensor(bbox.normalized_output(pos_type="cwh", normalized_range = (-1, 1))).to(device)
                                eval_objects[i][j] = torch.tensor(gen_rand_object()).to(device)
                                eval_obj_masks[i][j] = bbox.get_mask().to(device)
                            for j in range(rel_num):
                                rand_relation_id = np.random.randint(0, 6)
                                eval_relations[i][j] = torch.tensor([0,0,0,0,0,0,0,0,rand_relation_id]).to(device)
                                eval_relations_ids[i][j][0] = 0
                                eval_relations_ids[i][j][1] = 1
                                mask0 = eval_obj_masks[i][j][0]
                                mask1 = eval_obj_masks[i][j][1]
                                eval_rel_masks[i][j] = mask_OR(mask0, mask1)
                        

                        eval_objects_combined = [torch.cat((object, bbox), dim=-1) for object, bbox in zip(eval_objects, eval_bboxes)]
                        eval_cond = (eval_objects_combined, eval_relations, eval_relations_ids, eval_obj_masks, eval_rel_masks)
                        visualize(eval_cond, "random", bboxes = eval_bboxes)



                        # for i in range(self.eval_batch_size):
                        #     for j in range(self.obj_num):
                        #         random_bbox = torch.tensor(gen_rand_bbox().normalized_output(pos_type="cwh", normalized_range = (-1, 1)))
                        #         random_object = torch.tensor(gen_rand_object())
                        #         object_features = torch.cat((random_object, random_bbox), dim=-1)
                        #         random_combined_objects[i, j] = object_features
                            
                        #     random_relations_datum = []
                        #     random_relations_ids_datum = []
                        #     # for j in range(rel_num):

                        #     random_relations.append(random_relations_datum)
                        #     random_relations_ids.append(random_relations_ids_datum)

                        # # convert to device
                        # random_combined_objects = random_combined_objects.to(device)
                        
                        # random_conds = (random_combined_objects, random_relations, random_relations_ids)
                        # # print(f"compare random_combined_object with objects_combined_eval: {random_combined_objects}, {objects_combined}, {eval_objects_combined}")
                        # visualize(random_conds, "random", bboxes = random_combined_objects[:, :, 4:])
                        ################RANDOM################


                        # if self.calculate_fid:
                        #     fid_score = self.fid_scorer.fid_score()
                        #     accelerator.print(f'fid_score: {fid_score}')
                        # if self.save_best_and_latest_only:
                        #     if self.best_fid > fid_score:
                        #         self.best_fid = fid_score
                        #         self.save("best")
                        #     self.save("latest")
                        # else:
                        #     self.save(milestone)

        accelerator.print('training complete')
