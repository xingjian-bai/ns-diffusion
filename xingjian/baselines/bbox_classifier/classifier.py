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
import numpy as np
import argparse
import numpy as np
import wandb

from torch.utils.data import random_split
from sklearn.metrics import accuracy_score



class BboxClassifier(nn.Module):
    def __init__(self, nn_size = 48, inp_dim = 4 + 4 + 1, out_dim = 1):
        super().__init__()
        self.nn_size = nn_size

        self.fc1 = nn.Linear(inp_dim, nn_size)
        self.fc2 = nn.Linear(nn_size, nn_size)
        self.fc3 = nn.Linear(nn_size, nn_size)
        self.fc4 = nn.Linear(nn_size, out_dim)
        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def get_name(self):
        return f"4-layer-DNN-{self.nn_size}"
import sys
sys.path.append('../')
from dataset import RelationalDataset
class RelationClassifierDataset(Dataset):
    def __init__(self, dataset_type = "train", args = None):
        self.datasets = []
        if dataset_type == "train":
            self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz', split = "train"))
            if args.train_on_multi_rels:
                self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs_old.npz', split = "train"))
                self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_old.npz', split = "train"))
                self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_5objs_old.npz', split = "train"))
                self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_8objs_old.npz', split = "train"))
                
        elif dataset_type == "2O-test":
            self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz', split = "test"))
        elif dataset_type == "3O-test":
            self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_3objs_old.npz', split = "test"))
        elif dataset_type == "4O-test":
            self.datasets.append(RelationalDataset('/viscam/projects/ns-diffusion/dataset/clevr_rel_4objs_old.npz', split = "test"))
        else:
            raise NotImplementedError
        self.dataset_type = dataset_type

        # enumerate all elements in the dataset
        self.inputs = []
        self.labels = []
        for dataset in self.datasets:
            for (id, raw_relations) in enumerate(dataset.all_raw_relations):
                bboxes = dataset.bboxes[id]
                for i in range (len(raw_relations)):
                    for j in range (len(raw_relations[i])):
                        if i >= j:
                            continue
                        for (k, val) in enumerate(raw_relations[i][j]):
                            input = torch.concat([bboxes[i].tensorize(), bboxes[j].tensorize(), torch.tensor([k])])
                            label = torch.tensor([1. if val else 0])
                            self.inputs.append(input)
                            self.labels.append(label)
                        
        print(f"prepared dataset {dataset_type}, len = {len(self.inputs)}")
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def trainer(model, train_dataset, test_datasets, epochs=400, batch_size=32, lr=0.001, device="cuda", use_wandb = False, args = None):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loaders = [DataLoader(test_dataset, batch_size=128, shuffle=False) for test_dataset in test_datasets]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer, you can adjust the learning rate

    if use_wandb:
        wandb_name = f"{model.get_name()}" + ("_multi_rels" if args.train_on_multi_rels else "")
        wandb.init(project="relational-classifier", name = wandb_name, save_code = True)

    # Training Loop
    for epoch in range(100):  # number of epochs, adjust as needed
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data[0].to(device), data[1].to(device)  # Move inputs and labels to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct_predictions += ((outputs > 0.5).float() == labels).sum().item()
            total_predictions += labels.size(0)
        train_accuracy = correct_predictions / total_predictions

        # Evaluate on test set
        model.eval()  # set model to evaluation mode
        accuracies = []
        for test_loader in test_loaders:
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float()
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
            accuracy = correct_predictions / total_predictions
            accuracies.append(accuracy)

        print(f"Epoch: {epoch+1}, train loss: {running_loss/len(train_loader)}, test accuracy: {accuracies}, train accuracy: {train_accuracy}")
        if use_wandb:
            wandb.log(
                        {"train_loss": running_loss/len(train_loader), 
                        f"{test_datasets[0].dataset_type}_accuracy": accuracies[0],
                        f"{test_datasets[1].dataset_type}_accuracy": accuracies[1],
                        f"{test_datasets[2].dataset_type}_accuracy": accuracies[2],
                        "train_accuracy": train_accuracy, 
                        "epoch": epoch+1}, 
                        step = epoch+1)

        model.train()  # set model back to training mode

        if epoch % 10 == 9:  # Save every 10 epochs
            torch.save(model.state_dict(), f"{wandb_name}-{epoch+1}.pth")

    if use_wandb:
        wandb.finish()

# train the model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_size", type=int, default=48)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--train_on_multi_rels", action="store_true")
    args = parser.parse_args()

    model = BboxClassifier(nn_size = args.nn_size)
    train_dataset = RelationClassifierDataset(args = args)
    test_datasets = []
    test_datasets.append(RelationClassifierDataset(dataset_type = "2O-test"))
    test_datasets.append(RelationClassifierDataset(dataset_type = "3O-test"))
    test_datasets.append(RelationClassifierDataset(dataset_type = "4O-test"))

    trainer(model, train_dataset, test_datasets, use_wandb = args.wandb, args = args)

