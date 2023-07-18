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

import sys
sys.path.append('../')
from dataset import RelationalDataset
class RelationClassifierDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

        # enumerate all elements in the dataset
        self.inputs = []
        self.labels = []
        self.num = len(self.data)
        
        for (id, raw_relations) in enumerate(self.data.all_raw_relations):
            bboxes = self.data.bboxes[id]
            for i in range (len(raw_relations)):
                for j in range (len(raw_relations[i])):
                    if i >= j:
                        continue
                    for (k, val) in enumerate(raw_relations[i][j]):
                        # print(f"bbox[i], bbox[j], k = {bboxes[i]}, {bboxes[j]}, {k}")
                        input = torch.concat([bboxes[i].tensorize(), bboxes[j].tensorize(), torch.tensor([k])])
                        # print(f"what? {raw_relations[i][j]}, {k}")
                        label = torch.tensor([1. if val else 0])
                        self.inputs.append(input)
                        self.labels.append(label)
                        
        print(f"prepared dataset, len = {len(self.inputs)}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def trainer(model, train_dataset, test_dataset, epochs=100, batch_size=32, lr=0.001, device="cuda", use_wandb = False):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer, you can adjust the learning rate

    if use_wandb:
        wandb.init(project="relational-classifier", name = f"{model.nn_size}-default", save_code = True)

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
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        print(f"c epoch {epoch+1}")

        accuracy = correct_predictions / total_predictions
        print(f"Epoch: {epoch+1}, train loss: {running_loss/len(train_loader)}, test accuracy: {accuracy}, train accuracy: {train_accuracy}")
        if use_wandb:
            wandb.log({"train_loss": running_loss/len(train_loader), "test_accuracy": accuracy, "train_accuracy": train_accuracy, "epoch": epoch+1}, step = epoch+1)

        model.train()  # set model back to training mode

        if epoch % 10 == 9:  # Save every 10 epochs
            torch.save(model.state_dict(), f"rel_classifier_epoch_{epoch+1}.pth")


# train the model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz')
    parser.add_argument("--nn_size", type=int, default=48)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    model = BboxClassifier(nn_size = args.nn_size)
    train_dataset = RelationClassifierDataset(RelationalDataset(args.data_path, split = "train"))
    test_dataset = RelationClassifierDataset(RelationalDataset(args.data_path, split = "test"))
    trainer(model, train_dataset, test_dataset, use_wandb = args.wandb)

