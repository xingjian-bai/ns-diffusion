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
from dataset import RelationalDataset
class RelationalDataset2O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_imgs/combined_file.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)

class RelationalDataset1O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
    image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj_imgs/1.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=False):
        super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)
    


class AdaptedDataset(Dataset):
    def __init__(self, dataset="2O"):
        if dataset == "2O":
            self.data = RelationalDataset2O()
        elif dataset == "1O":
            self.data = RelationalDataset1O()
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
