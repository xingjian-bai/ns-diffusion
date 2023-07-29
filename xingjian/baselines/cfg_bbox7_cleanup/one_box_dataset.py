import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append('../')
from dataset_clevr_ryan import *



# class RelationalDataset4O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_old.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_imgs/1.npz'
#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
#         super().__init__(self.path, 
#                          uncond_image_type=uncond_image_type, 
#                          center_crop=center_crop, 
#                          pick_one_relation=pick_one_relation, 
#                          image_path=self.image_path, 
#                          split = "train", test_size = 0.1)


# class RelationalDataset2O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_balanced.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_balanced_imgs/combined_file.npz'
#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
#         super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)

# class RelationalDataset1O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj_imgs/1.npz'
#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
#         super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)
    


class AdaptedDataset(Dataset):
    def __init__(self, dataset="CLEVR_2O"):
        if dataset == "CLEVR_2O":
            self.data = RelationalDataset2O()
            self.obj_num = 2
        elif dataset == "CLEVR_1O":
            self.data = RelationalDataset1O()
            self.obj_num = 1
        elif dataset == "CLEVR_3O":
            self.data = RelationalDataset3O()
            self.obj_num = 3
        elif dataset == "CLEVR_4O":
            self.data = RelationalDataset4O()
            self.obj_num = 4
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
        return objects, relations, labels, generated_prompt, annotated_image_tensor, relations_ids

def collate_adapted(batch):
    objects_batch = []
    relations_batch = []
    labels_batch = []
    prompt_batch = []
    image_batch = []
    relations_ids_batch = []
    for (objects, relations, labels, prompt, image, relations_ids) in batch:
        objects_batch.append(objects)
        relations_batch.append(relations)
        labels_batch.append(labels)
        prompt_batch.append(prompt)
        image_batch.append(image)
        relations_ids_batch.append(relations_ids)
    return objects_batch, relations_batch, torch.stack(labels_batch), prompt_batch, image_batch, relations_ids_batch
