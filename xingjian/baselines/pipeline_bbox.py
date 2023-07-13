# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # print(f"shape inside SinusoidalPosEmb b = {emb.shape}")
        emb = x[:, None] * emb[None, :]
        # print(f"shape inside SinusoidalPosEmb c = {emb.shape}")
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print(f"shape of output of SinusoidalPosEmb = {emb.shape}")
        return emb


def swish(x):
    return x * torch.sigmoid(x)


class Diffusion1D(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        h = 512

        fourier_dim, time_dim = 128, 128

        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.fc1 = nn.Linear(inp_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, h)
        self.fc4 = nn.Linear(h, out_dim)

        self.t_map_fc2 = nn.Linear(time_dim, 2 * h)
        self.t_map_fc3 = nn.Linear(time_dim, 2 * h)

        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # print(f"in forward, shape of x = {x.shape}, shape of t = {t.shape}, shape of t_emb = {t_emb.shape}")

        fc2_gain, fc2_bias = torch.chunk(self.t_map_fc2(t_emb), 2, dim=-1)
        fc3_gain, fc3_bias = torch.chunk(self.t_map_fc3(t_emb), 2, dim=-1)

        h = swish(self.fc1(x))
        # print(f"shape here a : {h.shape}, {fc2_gain.shape}, {fc2_bias.shape}")
        #reshape fc2_gain and fc2_bias to match h
        fc2_gain = fc2_gain.reshape(h.shape[0], h.shape[1])
        fc2_bias = fc2_bias.reshape(h.shape[0], h.shape[1])
        h = swish(self.fc2(h) * (fc2_gain + 1) + fc2_bias)
        # print(f"shape here b : {h.shape}")

        fc3_gain = fc3_gain.reshape(h.shape[0], h.shape[1])
        fc3_bias = fc3_bias.reshape(h.shape[0], h.shape[1])
        h = swish(self.fc3(h) * (fc3_gain + 1) + fc3_bias)

        # print(f"shape here c: {h.shape}")
        output = self.fc4(h)

        # print(f"output forward, shape of output = {output.shape}")

        return output


from dataset import BoundingBox, Object, Relation


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_diffusion = Diffusion1D(4 + 4, 4)
        # self.rel_diffusion = Diffusion1D(1 + 9 + 9, 8)

    # def obj_denoise(self, coord: torch.Tensor, obj_features: torch.Tensor, t) -> torch.Tensor:
    #     input = torch.stack([coord, obj_features])
    #     return self.obj_diffusion(input, t)

    # def rel_denoise(rel: Relation, t) -> torch.Tensor[8]:
    #     rel = torch.cat([rel.color, rel.material, rel.shape, rel.size])
    #     coor1 = torch.tensor(rel.bbox1.bounds)
    #     coor2 = torch.tensor(rel.bbox2.bounds)
    #     inp = torch.cat([feature, coor1, coor2])

    #     return self.rel_diffusion(inp, rel.bbox1.time, t)

    def denoise(self, coords: torch.Tensor , objects: torch.Tensor, relations: torch.Tensor, t):
        # print(f"in denoise, t = {t} with type {type(t)}")
        # aggregated_noise = torch.zeros_like(coords)

        obj_denoise_input = torch.stack([torch.cat([coord, obj]) for coord, obj in zip(coords, objects)])
        # print(f"co, obj, and obj_input: {coords}, {objects}, {obj_denoise_input}")
        # repeat t for each object
        timesteps = t.repeat(len(coords), 1)
        obj_noise = self.obj_diffusion(obj_denoise_input, timesteps)

        # print(f"end of denoise, shape of obj_noise = {obj_noise.shape}, coords = {coords.shape}")

        return obj_noise


    def denoise_batch(self, coords_batch, objects_batch, relations_batch, timesteps):
        noises = []
        for i in range(len(coords_batch)):
            noises.append(
                self.denoise(
                    coords_batch[i],
                    objects_batch[i],
                    relations_batch[i],
                    timesteps[i],
                )
            )
        return torch.stack(noises)
    
    def forward(self, coords_batch, objects_batch, relations_batch, timesteps):
        return self.denoise_batch(coords_batch, objects_batch, relations_batch, timesteps)
        

from diffusers import DiffusionPipeline
from PIL import Image

class BboxPipeline():
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, denoiser, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.denoiser = denoiser

    
    @torch.no_grad()
    def __call__(
        self,
        objects: List[Object],
        relations: List[Relation] = [],
        num_objects: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> List[BoundingBox]:
        assert num_objects == len(objects), f"num_objects ({num_objects}) must match the length of objects ({len(objects)})"

        # random values from -1 to 1, with shape (num_objects, 4)
        coordinates = (torch.rand(num_objects, 4) * 2 - 1).cuda()
        objects_tensors = [obj.tensorize().cuda() for obj in objects]
        relations_tensors = [rel.tensorize().cuda() for rel in relations]

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in range(num_inference_steps):
            # convert t to tensor int
            t = torch.tensor(t, dtype=torch.int64).cuda()
            model_output = self.denoiser.denoise(coordinates, objects_tensors, relations_tensors, t)

            coordinates = self.scheduler.step(model_output, t, coordinates, generator=generator).prev_sample
            print(f"predicted coord after step {t}: {coordinates}")
        # create a white image in PIL.Image format

        # image = Image.new("RGB", (128, 128), (255, 255, 255))

        # turn coordinates to bounding boxes
        bboxes = []
        for i, coord in enumerate(coordinates):
            bbox = BoundingBox(coord)
            bbox.denormalize()
            bboxes.append(bbox)

        assert len(bboxes) == len(objects)
        return bboxes
