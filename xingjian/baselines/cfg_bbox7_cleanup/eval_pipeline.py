import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
import wandb
import sys
sys.path.append('../')
from dataset_clevr_ryan import RelationalDataset, BoundingBox
from utils import *
from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class RelationalDataset2O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_balanced.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_balanced_imgs/combined_file.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = args.num_upperbound)
    

# class RelationalDataset1O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj_imgs/1.npz'
#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
#         super().__init__(self.path, 
#                          uncond_image_type=uncond_image_type, 
#                          center_crop=center_crop, 
#                          pick_one_relation=pick_one_relation, 
#                          image_path=self.image_path, 
#                          split = "test", num_upperbound = args.num_upperbound)
                         

class RelationalDataset3O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = args.num_upperbound)
class RelationalDataset4O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = args.num_upperbound)

class RelationalDataset5O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_5objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_5objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = args.num_upperbound)

class RelationalDataset8O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_8objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_8objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = args.num_upperbound)


class RelationalDatasetxO(RelationalDataset):
    def __init__(self, x, uncond_image_type="original", center_crop=True, pick_one_relation=False, args = None, upperbound = 1000):
        super().__init__(data_path = "nothing",
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         split = "train", num_upperbound = args.num_upperbound if args is not None else upperbound,
                         generated_object_num = x, generated_data_num = 1000)

class EvalAdaptedDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
        return objects, relations, relations_ids, labels, annotated_image_tensor
    
def collate_adapted(batch):
    objects_batch = []
    relations_batch = []
    relations_ids_batch = []
    labels_batch = []
    image_batch = []
    
    for (objects, relations, relations_ids, labels, annotated_image_tensor) in batch:
        objects_batch.append(objects)
        relations_batch.append(relations)
        relations_ids_batch.append(relations_ids)
        labels_batch.append(labels)
        image_batch.append(annotated_image_tensor)
    return objects_batch, relations_batch, relations_ids_batch, torch.stack(labels_batch), torch.stack(image_batch)

from trainer import Trainer1D






from diffuser import GaussianDiffusion1D

class EvalInfo:
    def __init__(self):
        # list of size 6
        self.correct_relations = [0] * 6
        self.total_relations = [0] * 6
        self.relation_names = ["left", "right", "front", "behind", "above", "below"]
    
    def update(self, rel_id, result):
        self.correct_relations[rel_id] += result
        self.total_relations[rel_id] += 1
    
    def to_list(self):
        return [self.correct_relations[i] / self.total_relations[i] if self.total_relations[i] != 0 else 0 for i in range(6)]
    def print(self):
        for i in range(6):
            print(f"{self.relation_names[i]}: {self.correct_relations[i]} / {self.total_relations[i]}")

def eval(trainer, single_image_eval, data_loader, dataset_step, show_off_mode = False, wandb_drawer = None, args = None):
    accelerator = trainer.accelerator
    device = accelerator.device
    scores = []
    eval_info = EvalInfo()

    print("eval with show_off_mode: ", show_off_mode)
        
    #make dataset into dataloader
    for (data_id, data) in enumerate(data_loader):
        objects, relations, relations_ids, labels, images = data
        mask = None

        objects = [obj.to(device) for obj in objects]
        relations = [relation.to(device) for relation in relations]
        labels = [l.to(device) for l in labels]
        images = [image.to(device) for image in images]
        relations_ids = [relation_id.to(device) for relation_id in relations_ids]
        

        if show_off_mode:
            repeat = args.repeat
            objects = [item for item in objects for _ in range(repeat)][:trainer.eval_batch_size]
            relations = [item for item in relations for _ in range(repeat)][:trainer.eval_batch_size]
            relations_ids = [item for item in relations_ids for _ in range(repeat)][:trainer.eval_batch_size]
            labels = [item for item in labels for _ in range(repeat)][:trainer.eval_batch_size]
            images = [item for item in images for _ in range(repeat)][:trainer.eval_batch_size]

        inp = (objects, relations, relations_ids)
        # print("dealing with data_id: ", data_id)
        trainer.ema.ema_model.eval()
        with torch.no_grad():
            all_samples_list = list(map(lambda n: trainer.ema.ema_model.sample(
                inp, labels, mask,
                batch_size=trainer.eval_batch_size), range(1)))

        all_samples = torch.cat(all_samples_list, dim = 0)

        for i in range(len(relations)):
            score, eval_info = single_image_eval(all_samples[i], relations[i], relations_ids[i], eval_info = eval_info)
            scores.append(score)
        
        if show_off_mode:
            bboxes = [[BoundingBox(e.tolist()) for e in this_out] for this_out in all_samples]
            for i, image in enumerate(images):
                if isinstance(image, torch.Tensor):
                    image = tensor_to_pil(image)
                colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # red, green, blue, yellow, cyan, magenta
                for j, bbox in enumerate(bboxes[i]):
                    image = bbox.draw(image, color=colours[j % len(colours)])
                
                if args.with_scene_graph:
                    from dataset_clevr_ryan import draw_scene_graph
                    scene_graph = draw_scene_graph(objects[i], relations[i], relations_ids[i])
                    image = combine_images(image, scene_graph)

                images[i] = image
            if wandb_drawer is not None:
                wandb_drawer.log({"images": [wandb.Image(image) for image in images]}, step = dataset_step)
                break
    if not show_off_mode:
        if args.no_above_below:
            avg_score = (eval_info.correct_relations[0] + eval_info.correct_relations[1] + eval_info.correct_relations[2] + eval_info.correct_relations[3]) / (eval_info.total_relations[0] + eval_info.total_relations[1] + eval_info.total_relations[2] + eval_info.total_relations[3])
        else:
            avg_score = sum(scores) / len(scores)
        print("acc: ", avg_score)
        if wandb_drawer is not None:
            # separately log each relation's acc
            eval_info_list = eval_info.to_list()
            eval_info.print()
            for i in range(6):
                wandb_drawer.log({f"acc_{eval_info.relation_names[i]}": eval_info_list[i]}, step = dataset_step)
            wandb_drawer.log({"acc": avg_score}, step = dataset_step)
        return avg_score

def evaluate(dataset_type, metric, threshold = 0.5, wandb_drawer = None, args = None):
    print(f"evaluating model (epoch={args.epoch}) with data={dataset_type}, metric={metric}")
    if metric == 'relation_classifier':
        sys.path.append('../bbox_classifier/')
        from classifier import BboxClassifier
        metric_model = BboxClassifier()
        metric_model.load_state_dict(torch.load('../bbox_classifier/' + args.eval_model_pth))
        def single_image_eval(bboxes, relations, relations_ids, eval_info = EvalInfo()):
            # print("entered single image eval")
            correct_relations = 0
            for (i, rel) in enumerate(relations):
                (a, b) = relations_ids[i]
                a = a.item()
                b = b.item()
                rel_id = rel[-1]
                input = torch.concat([bboxes[a].cuda(), bboxes[b].cuda(), torch.tensor([rel_id]).cuda()])
                input = input.cuda()
                metric_model.cuda()
                pred = metric_model(input)[0].item()
                correct_relations += pred > threshold
                eval_info.update(rel_id, pred > threshold)

                # print(f"{i}th relation: {pred} vs {rel[0]}")
            return correct_relations / len(relations), eval_info
    else:
        raise NotImplementedError
    
    if dataset_type == 2 and args.pure_synthetic == False:
        dataset = RelationalDataset2O(args = args)
    elif dataset_type == 3 and args.pure_synthetic == False:
        dataset = RelationalDataset3O(args = args)
    elif dataset_type == 4 and args.pure_synthetic == False:
        dataset = RelationalDataset4O(args = args)
    elif dataset_type == 5 and args.pure_synthetic == False:
        dataset = RelationalDataset5O(args = args)
    elif dataset_type == 8 and args.pure_synthetic == False:
        dataset = RelationalDataset8O(args = args)
    else:
        # print("self generating dataset...")
        dataset = RelationalDatasetxO(dataset_type, args = args)
    
    from models import BiDenoise

    # print(f"check args.model_name: {args.model_name}")
    if args.model_name == "both":
        model = BiDenoise().cuda()
    elif args.model_name == "rel_only":
        model = BiDenoise(rel_only = True).cuda()
    elif args.model_name == "obj_only":
        model = BiDenoise(obj_only = True).cuda()
    else:
        raise NotImplementedError

    print(f"evaluating {args.model_name} on {dataset_type} objects")

    diffuser = GaussianDiffusion1D(model,seq_length = 32,
            beta_schedule = 'cosine',
            objective = 'pred_noise', 
            timesteps = 100,
            sampling_timesteps = 100,
            obj_num = dataset_type,
        ).cuda()
    trainer = Trainer1D(
            diffuser,
            dataset_type = args.train_dataset_type,
            wandb_drawer = False,
            eval_batch_size = args.batch_size,
            name = f"bbox-diffusion1.0",
        )
    
    trainer.load(args.epoch)
    # print(f"loaded model", trainer.device)

    eval_dataset = EvalAdaptedDataset(dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_adapted)
    
    eval(trainer, single_image_eval, eval_dataloader, dataset_type, show_off_mode = True, wandb_drawer = wandb_drawer, args = args)
    eval(trainer, single_image_eval, eval_dataloader, dataset_type, show_off_mode = False, wandb_drawer = wandb_drawer, args = args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval pipeline')
    parser.add_argument('--epoch', default=272, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch of input to use')
    parser.add_argument('--num_upperbound', default=200, type=int)
    parser.add_argument('--eval_model_pth', type=str, default='4-layer-DNN-48_multi_rels-400.pth')
    parser.add_argument('--metric', type=str, default='relation_classifier', help='metric to use')
    parser.add_argument('--model_name', type=str, default='both', help='name of the model tested')
    parser.add_argument('--wandb', default=False, action='store_true', help='use wandb')
    parser.add_argument('--no_above_below', default=False, action='store_true', help='use wandb')
    parser.add_argument('--pure_synthetic', default=False, action='store_true')
    parser.add_argument('--train_dataset_type', type=str, default='CLEVR_2O')
    parser.add_argument('--eval_from', default = 2, type = int)
    parser.add_argument('--eval_to', default = 9, type = int)
    parser.add_argument('--repeat', default = 4, type = int)
    parser.add_argument('--with_scene_graph', default=True, action='store_true')
    
    args, unknown = parser.parse_known_args()

    if args.wandb:
        wandb_drawer = wandb.init(
            project="diffusion_bbox_eval",
            name=f"{args.model_name}-{args.metric}-{args.num_upperbound}" + ("syn" if args.pure_synthetic else "") + f"-{args.train_dataset_type}" + f"-{args.epoch}",
            save_code=True,
        )
    else:
        wandb_drawer = None

    for obj_num in range(args.eval_from, args.eval_to):
        evaluate(obj_num, args.metric, wandb_drawer = wandb_drawer, args = args)

    if wandb_drawer is not None:
        wandb_drawer.finish()