import numpy as np
import argparse
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torchvision import transforms

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
