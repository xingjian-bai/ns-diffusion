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

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__



import matplotlib.pyplot as plt
def show_image(image):
    plt.imshow(image)
    plt.show()

# concat horizontally two Image into one
# def combine_images(im1, im2):
#     dst = Image.new('RGB', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst
def combine_images(im1, im2):
    max_height = max(im1.height, im2.height)
    im1_new_width = int(im1.width * max_height / im1.height)
    im2_new_width = int(im2.width * max_height / im2.height)
    im1_resized = im1.resize((im1_new_width, max_height))
    im2_resized = im2.resize((im2_new_width, max_height))

    # Create a new image that fits both the resized images
    new_width = im1_resized.width + im2_resized.width
    dst = Image.new('RGB', (new_width, max_height))
    dst.paste(im1_resized, (0, 0))
    dst.paste(im2_resized, (im1_resized.width, 0))
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


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t