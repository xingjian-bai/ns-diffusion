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
from dataset_clevr_ryan import RelationalDataset1O


class AdaptedDataset(Dataset):
    def __init__(self, dataset="CLEVR_1O"):
        # if dataset == "CLEVR_2O":
        #     self.data = RelationalDataset2O()
        #     self.obj_num = 2
        # elif dataset == "CLEVR_1O":
        #     self.data = RelationalDataset1O()
        #     self.obj_num = 1
        # elif dataset == "CLEVR_3O":
        #     self.data = RelationalDataset3O()
        #     self.obj_num = 3
        # elif dataset == "CLEVR_4O":
        #     self.data = RelationalDataset4O()
        #     self.obj_num = 4
        # else:
        #     raise NotImplementedError

        assert dataset == "CLEVR_1O"
        self.data = RelationalDataset1O()

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

# def collate_adapted(batch):
#     objects_batch = []
#     relations_batch = []
#     labels_batch = []
#     prompt_batch = []
#     image_batch = []
#     relations_ids_batch = []
#     for (objects, relations, labels, prompt, image, relations_ids) in batch:
#         objects_batch.append(objects)
#         relations_batch.append(relations)
#         labels_batch.append(labels)
#         prompt_batch.append(prompt)
#         image_batch.append(image)
#         relations_ids_batch.append(relations_ids)
#     return objects_batch, relations_batch, torch.stack(labels_batch), prompt_batch, image_batch, relations_ids_batch
