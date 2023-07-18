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
from dataset import RelationalDataset

class RelationalDataset2O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_imgs/combined_file.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "test", num_upperbound = 100)
    

class RelationalDataset1O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "train", test_size = 0.1, num_upperbound = 100)
                         

class RelationalDataset3O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs_old.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "train", test_size = 0.1, num_upperbound = 100)
class RelationalDataset4O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_old.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, 
                         uncond_image_type=uncond_image_type, 
                         center_crop=center_crop, 
                         pick_one_relation=pick_one_relation, 
                         image_path=self.image_path, 
                         split = "train", test_size = 0.1, num_upperbound = 100)
class EvalAdaptedDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
        return objects, relations, relations_ids, labels, annotated_image_tensor
    
# class RepeatDataset(Dataset):
#     def __init__(self, dataset, repeat = 8):
#         self.data = dataset
#         self.repeat = repeat
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         # divide all indices by 8
#         start = index.start // self.repeat if index.start is not None else None
#         stop = index.stop // self.repeat if index.stop is not None else None
#         step = index.step // self.repeat if index.step is not None else None
#         index = slice(start, stop, step)


#         clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
#         return objects, relations, relations_ids, labels, annotated_image_tensor


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




parser = argparse.ArgumentParser(description='eval pipeline')
parser.add_argument('--batch_size', default=32, type=int, help='size of batch of input to use')
parser.add_argument('--dataset', type=str, default='CLEVR_2O', help='dataset to use')
parser.add_argument('--metric', type=str, default='relation_classifier', help='metric to use')
parser.add_argument('--model_name', type=str, default='relation_classifier', help='model to use')
args = parser.parse_args()

from diffuser import GaussianDiffusion1D
def evaluate(dataset_type, metric):
    
    if metric == 'relation_classifier':
        sys.path.append('../bbox_classifier/')
        from classifier import BboxClassifier
        metric_model = BboxClassifier()
        metric_model.load_state_dict(torch.load('../bbox_classifier/rel_classifier_epoch_100.pth'))
        def single_image_eval(bboxes, relations, relations_ids):
            # print("entered single image eval")
            correct_relations = 0
            total_relations = 0
            for (i, rel) in enumerate(relations):
                (a, b) = relations_ids[i]
                a = a.item()
                b = b.item()
                # print("?", bboxes[a],bboxes[b], rel)
                input = torch.concat([bboxes[a].cuda(), bboxes[b].cuda(), torch.tensor([rel[-1]]).cuda()])
                input = input.cuda()
                metric_model.cuda()
                pred = metric_model(input)[0].item()

                if pred > 0.5: # has relation
                    correct_relations += 1
                total_relations += 1
                # print(f"{i}th relation: {pred} vs {rel[0]}")
            return correct_relations / total_relations
    else:
        raise NotImplementedError
    
    if dataset_type == 'CLEVR_2O':
        dataset = RelationalDataset2O()
        obj_num = 2
    elif dataset_type == 'CLEVR_1O':
        dataset = RelationalDataset1O()
        obj_num = 1
    elif dataset_type == 'CLEVR_3O':
        dataset = RelationalDataset3O()
        obj_num = 3
    elif dataset_type == 'CLEVR_4O':
        dataset = RelationalDataset4O()
        obj_num = 4
    else:
        raise NotImplementedError
    
    from models import BiDenoise
    model = BiDenoise().cuda()
    diffuser = GaussianDiffusion1D(model,seq_length = 32,
            beta_schedule = 'cosine',
            objective = 'pred_noise', 
            timesteps = 100,
            sampling_timesteps = 100,
            obj_num = obj_num,
        ).cuda()
    trainer = Trainer1D(
            diffuser,
            dataset_type = dataset_type,
            wandb = False,
            eval_batch_size = args.batch_size,
            name = f"model_{dataset_type}_{metric}",
        )
    
    trainer.load(200)
    print(f"loaded model", trainer.device)
    
    

    eval_dataset = EvalAdaptedDataset(dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_adapted)
    

    wandb_drawer = wandb.init(
        project="diffusion_bbox_eval",
        name=f"model_{dataset_type}_{metric}",
        save_code=True,
    )
    trainer.eval(single_image_eval, eval_dataloader, show_off_mode = True, repeat = 8, wandb_drawer = wandb_drawer)
    acc = trainer.eval(single_image_eval, eval_dataloader, wandb_drawer = wandb_drawer)
if __name__ == '__main__':
    
    evaluate(args.dataset, args.metric)