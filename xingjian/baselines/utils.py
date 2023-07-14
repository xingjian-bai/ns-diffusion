import numpy as np
import argparse
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import math
import sys
import time

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

def tensor_to_pil(image_tensor: torch.Tensor) -> Image:
    # The transform expects the input to be in [C, H, W] format,
    # so transpose the tensor if it's not in that format
    # print(f"shape of image_tensor a: {image_tensor.shape}")
    if len(image_tensor.shape) == 3 and image_tensor.shape[0] != 3 and image_tensor.shape[0] != 4:
        image_tensor = image_tensor.permute(2, 0, 1)
    # print(f"shape of image_tensor b: {image_tensor.shape}")
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor)
    return image

def pil_to_tensor(image: Image) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    return image_tensor

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ['0', 'n', 'f']:
        return False
    elif x[0] in ['1', 'y', 't']:
        return True
    raise ValueError('Invalid value: {}'.format(x))




def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))