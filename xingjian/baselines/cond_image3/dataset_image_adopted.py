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

class AdaptedDataset(Dataset):
    def __init__(self, dataset="CLEVR_1O", split = "train", num_upperbound = None):
        super().__init__()
        if dataset == "CLEVR_1O":
            self.data = RelationalDataset1O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 1
        elif dataset == "CLEVR_2O":
            self.data = RelationalDataset2O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 2
        elif dataset == "CLEVR_3O":
            self.data = RelationalDataset3O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 3
        elif dataset == "CLEVR_4O":
            self.data = RelationalDataset4O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 4
        elif dataset == "CLEVR_5O":
            self.data = RelationalDataset5O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 5
        elif dataset == "CLEVR_8O":
            self.data = RelationalDataset8O(split=split, num_upperbound = num_upperbound)
            self.obj_num = 8
        elif dataset == "Mix123":
            self.data = Mix123(split=split, num_upperbound = num_upperbound)
            self.obj_num = None
        else:
            raise ValueError(f"dataset {dataset} not supported")
        # print(f"in adopted dataset, finished loading {len(self.data)} data")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids, obj_mask, rel_mask, annotated_image_tensor = self.data[index]
        # in image generation, all needed are images, objects, bboxes, relations, relations_ids
        # assert the type of returned items are all tensors
        assert isinstance(clean_image, torch.Tensor), f"clean_image is not a tensor, but {type(clean_image)}"
        assert isinstance(objects, torch.Tensor), f"objects is not a tensor, but {type(objects)}"
        assert isinstance(relations, torch.Tensor), f"relations is not a tensor, but {type(relations)}"
        assert isinstance(labels, torch.Tensor), f"labels is not a tensor, but {type(labels)}"
        assert isinstance(relations_ids, torch.Tensor), f"relations_ids is not a tensor, but {type(generated_prompt)}"
        assert isinstance(obj_mask, torch.Tensor), f"obj_mask is not a tensor, but {type(obj_mask)}"
        assert isinstance(rel_mask, torch.Tensor), f"rel_mask is not a tensor, but {type(rel_mask)}"

        return clean_image, objects, labels, relations, relations_ids, obj_mask, rel_mask

def collate_adapted(batch):
    clean_image_batch = []
    objects_batch = []
    bboxes_batch = []
    relations_batch = []
    relations_ids_batch = []
    obj_mask_batch = []
    rel_mask_batch = []

    for (clean_image, objects, labels, relations, relations_ids, obj_mask, rel_mask) in batch:
        clean_image_batch.append(clean_image)
        objects_batch.append(objects)
        bboxes_batch.append(labels)
        relations_batch.append(relations)
        relations_ids_batch.append(relations_ids)
        obj_mask_batch.append(obj_mask)
        rel_mask_batch.append(rel_mask)
    return torch.stack(clean_image_batch), objects_batch, bboxes_batch, relations_batch, relations_ids_batch, obj_mask_batch, rel_mask_batch
