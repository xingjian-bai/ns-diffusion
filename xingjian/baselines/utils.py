import numpy as np
import argparse
import torch
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
def show_image(image):
    plt.imshow(image)
    plt.show()

# concat horizontally two Image into one
def combine_images(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
