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
from dataset_clevr_ryan import RelationalDataset1O, RelationalDataset2O


class AdaptedDataset(Dataset):
    def __init__(self, dataset="CLEVR_1O"):
        super().__init__()
        if dataset == "CLEVR_1O":
            self.data = RelationalDataset1O()
            sekf.obj_num = 1
        elif dataset == "CLEVR_2O":
            self.data = RelationalDataset2O()
            self.obj_num = 2
        else:
            raise ValueError(f"dataset {dataset} not supported")

        print(f"in adopted dataset, finished loading {len(self.data)} data")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
        # in image generation, all needed are images, objects, bboxes, relations, relations_ids
        # assert the type of returned items are all tensors
        assert isinstance(clean_image, torch.Tensor), f"clean_image is not a tensor, but {type(clean_image)}"
        assert isinstance(objects, torch.Tensor), f"objects is not a tensor, but {type(objects)}"
        assert isinstance(relations, torch.Tensor), f"relations is not a tensor, but {type(relations)}"
        assert isinstance(labels, torch.Tensor), f"labels is not a tensor, but {type(labels)}"
        assert isinstance(relations_ids, torch.Tensor), f"generated_prompt is not a tensor, but {type(generated_prompt)}"

        return clean_image, objects, labels, relations, relations_ids

def collate_adapted(batch):
    clean_image_batch = []
    objects_batch = []
    bboxes_batch = []
    relations_batch = []
    relations_ids_batch = []

    for (clean_image, objects, labels, relations, relations_ids) in batch:
        clean_image_batch.append(clean_image)
        objects_batch.append(objects)
        bboxes_batch.append(labels)
        relations_batch.append(relations)
        relations_ids_batch.append(relations_ids)
    return torch.stack(clean_image_batch), torch.stack(objects_batch), torch.stack(bboxes_batch), relations_batch, relations_ids_batch
