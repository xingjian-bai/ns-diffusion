import argparse
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader

class SingleBoxDataset(Dataset):
    def __init__(self):
        self.ds = self.randomly_generate()
    
    def random_size(self, mean, std = 0.2):
        # w in normal distribution with mean and std
        w = random.normalvariate(mean, std)
        h = random.normalvariate(mean, std)
        w = max(min(w, 0.8), -0.8)
        h = max(min(h, 0.8), -0.8)
        return w, h


    def randomly_generate(self, n=10000):
        lists = []
        for i in range(n):
            is_big = random.random() < 0.5
            if is_big:
                size = self.random_size(0.0)
            else:
                size = self.random_size(-0.5)

            size = (size[0], size[0])

            # range: (-1, 1)
            size_ratio = [size[0] + 1, size[1] + 1]
            x = random.random() * (2 - size_ratio[0]) + size_ratio[0] / 2 - 1
            y = random.random() * (2 - size_ratio[1]) + size_ratio[1] / 2 - 1
            # print(f"generated xysize: {x=}, {y=}, {size=}")
            # print(f"scale in [0, 2]: {x + 1}, {y + 1}, {size_ratio[0]}, {size_ratio[1]}")
            

            lists.append((1 - is_big, is_big, x, y, size[0], size[1]))
        return torch.tensor(lists, dtype=torch.float32)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        fd = self.ds[index]
        return fd[:2], fd[2:]

class AdaptedDataset(Dataset):
    def __init__(self):
        import sys
        sys.path.append('../')
        from dataset import RelationalDataset2O
        self.data = RelationalDataset2O()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        clean_image, objects, relations, labels, generated_prompt, raw_image, annotated_image_tensor, relations_ids = self.data[index]
        

        # print(f"in getitem: {object=}, {label=}")
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
    return torch.stack(objects_batch), torch.stack(relations_batch), torch.stack(labels_batch), prompt_batch, image_batch, relations_ids_batch
