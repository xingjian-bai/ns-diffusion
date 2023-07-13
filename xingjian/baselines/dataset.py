import numpy as np
import argparse
import torch
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from utils import *

# a function random pick a colour
def random_colour():
    return tuple(np.random.randint(40, 200, 3))

class BoundingBox:
    def __init__(self, bounds, color=None):
        assert len(bounds) == 4
        self.bounds = bounds
        self.color = random_colour() if color is None else color 
    
    def tensorize(self):
        self.normalize()
        return torch.tensor(self.bounds)

    def normalize(self, resolution = 128):
        """
        From [0, 256] to (-1, 1}, from int type to float type
        """
        if max(self.bounds) >= 1:
            self.bounds = [float((x - resolution / 2) / (resolution / 2)) for x in self.bounds]


    def denormalize(self, resolution = 128):
        """
        From (-1, 1} to [0, 256], from float type to int type
        """
        if max(self.bounds) < 1:
            self.bounds = [int((x + 1) * (resolution / 2)) for x in self.bounds]

    def mse(self, other):
        self.normalize()
        return np.square(np.subtract(self.bounds, other.bounds)).mean()
    
    def draw(self, image = Image.new("RGB", (128, 128), (255, 255, 255))):
        self.denormalize()
        draw = ImageDraw.Draw(image)
        # print(f"before draw, bounds = {self.bounds}")
        draw.rectangle((self.bounds[0],self.bounds[1],self.bounds[2],self.bounds[3]) , outline=self.color, width=1)
        return image

    def __str__(self) -> str:
        return f"BBOX({self.bounds[0]}, {self.bounds[1]}) - ({self.bounds[2]}, {self.bounds[3]})"

def draw_bboxes(bboxes, image = Image.new("RGB", (128, 128), (255, 255, 255))):
    image = image.copy()
    for bbox in bboxes:
        # print(f"draw_bboxes: bbox = {bbox}")
        image = bbox.draw(image)
    return image

class Object:
    shapes = ["cube", "sphere", "cylinder", "none"]
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow", "none"]
    materials = ["rubber", "metal", "none"]
    sizes = ["small", "large", "none"]
    # Ryan's order : color, material, shape, size

    def __init__(self, color_idx, material_idx, shape_idx, size_idx):
        self.color_idx = color_idx
        self.material_idx = material_idx
        self.shape_idx = shape_idx
        self.size_idx = size_idx
    
    def tensorize(self):
        ret = torch.tensor([self.color_idx, self.material_idx, self.shape_idx, self.size_idx])
        assert ret.shape == (4,)
        return ret

    def equip_bbox(self, bbox):
        self.bbox = bbox
        self.bbox.denormalize()
    
    def __str__(self, mode="normal") -> str:
        if mode == "normal":
            return f"a {Object.sizes[self.size_idx]} {Object.colors[self.color_idx]} {Object.materials[self.material_idx]} {Object.shapes[self.shape_idx]}"
        else:
            raise NotImplementedError

class Relation:
    relations = ["left", "right", "front", "behind", "below", "above", "none"]
    rel_to_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below",
            "none": "none"
        }
    colours = [(255, 0, 0),     # Red
                (0, 255, 0),     # Green
                (0, 0, 255),     # Blue
                (255, 255, 0),   # Yellow
                (0, 255, 255),   # Cyan
                (255, 0, 255),   # Magenta
                (0, 0, 0),       # Black
    ]
    def __init__(self, relation_idx, obj1, obj2):
        self.relation_idx = relation_idx
        self.relation = Relation.relations[relation_idx]
        self.obj1 = obj1
        self.obj2 = obj2
        self.colour = Relation.colours[relation_idx]
    
    def tensorize(self):
        ret = torch.tensor([self.relation_idx, *self.obj1.tensorize(), *self.obj2.tensorize()])
        assert ret.shape == (9,)
        return ret
    
    def __str__(self) -> str:
        # relation = Relation.relations[self.relation_idx]
        # print(f"in __str__ of Relation, relation_idx = {self.relation_idx}")
        # print(f"?? {self.obj1}")
        # print(f"?? {Relation.rel_to_description}")
        # print(f"?? {self.obj2}")
        return f"{self.obj1} {Relation.rel_to_description[self.relation]} {self.obj2}"
    
    def draw_relation(self, image, thickness=1):
        # calculate the center of the bounding boxes
        c1x, c1y = (int((self.obj1.bbox.bounds[0] + self.obj1.bbox.bounds[2]) / 2), int((self.obj1.bbox.bounds[1] + self.obj1.bbox.bounds[3]) / 2))
        c2x, c2y = (int((self.obj2.bbox.bounds[0] + self.obj2.bbox.bounds[2]) / 2), int((self.obj2.bbox.bounds[1] + self.obj2.bbox.bounds[3]) / 2))

        draw = ImageDraw.Draw(image)
        draw.line((c1x, c1y, c2x, c2y), fill=self.colour, width=thickness)
        return image
    
def annotate(image, relations, objects):
    image_with_annotation = image.copy()
    for relation in relations:
        image_with_annotation = relation.draw_relation(image_with_annotation)
    for obj in objects:
        image_with_annotation = obj.bbox.draw(image_with_annotation)
    return image_with_annotation

def prompt(objects = [], relations = []):
    return ", AND ".join([str(rel) for rel in relations if rel.relation_idx != 6]) + " AND " + ", AND ".join([str(obj) for obj in objects])

def gen_rand_object():
    color_idx = np.random.randint(0, len(Object.colors))
    material_idx = np.random.randint(0, len(Object.materials))
    shape_idx = np.random.randint(0, len(Object.shapes))
    size_idx = np.random.randint(0, len(Object.sizes))
    return Object(color_idx, material_idx, shape_idx, size_idx)

def gen_rand_relation(obj1, obj2):
    relation_idx = np.random.randint(0, len(Relation.relations))
    return Relation(relation_idx, obj1, obj2)

def gen_rand_scene(num_objects, num_relations):
    objects = []
    relations = []
    for i in range(num_objects):
        objects.append(gen_rand_object())
    for i in range(num_relations):
        obj1_idx = i % len(objects)
        obj2_idx = np.random.randint(0, len(objects))
        while obj1_idx == obj2_idx:
            obj2_idx = np.random.randint(0, len(objects))
        relations.append(gen_rand_relation(objects[obj1_idx], objects[obj2_idx]))
    return objects, relations
def gen_prompt(num_objects, num_relations):
    objects, relations = gen_rand_scene(num_objects, num_relations)
    return prompt(relations = relations)


class RelationalDataset(Dataset):
    def formula_to_image(self, raw_relations, raw_objects):
        """
            Picks some suitable relations from raw_relations
        """
        relations = []
        for i in range (len(raw_relations)):
            for j in range (len(raw_relations[i])):
                if i > j:
                    continue 
                if self.pick_one_relation:
                    # randomly pick one non-zero index in raw_relations[i][j]
                    idx = np.random.choice(np.nonzero(raw_relations[i][j])[0]) if np.sum(raw_relations[i][j]) > 0 else 6
                    relations.append(Relation(idx, raw_objects[i], raw_objects[j]))
                else:
                    raise NotImplementedError
        return relations
                
    def __init__(
        self,
        data_path,
        uncond_image_type="original",
        center_crop=True,
        pick_one_relation=True,
    ):
        self.center_crop = center_crop
        self.data = np.load(data_path)
        self.pick_one_relation = pick_one_relation
        self.uncod_image_type = uncond_image_type

        # print(self.data.keys())

        self.objects = [[Object(*obj) for obj in objects] for objects in self.data['objects']] # list of list of objects
        self.bboxes = [[BoundingBox(bbox) for bbox in bboxes] for bboxes in self.data['bboxes']] # list of list of bboxes
        for objects, bboxes in zip(self.objects, self.bboxes):
            for obj, bbox in zip(objects, bboxes):
                obj.equip_bbox(bbox)
        
        if "images" in self.data.keys():
            self.images = [Image.fromarray(image) for image in self.data['images']]
        else:
            self.images = [draw_bboxes(bboxes) for bboxes in self.bboxes]

        self.relations = [self.formula_to_image(raw_relations, raw_objects) for raw_relations, raw_objects in zip(self.data['relations'], self.objects)]
        self.annotated_images = [annotate(image, relations, objects) for image, relations, objects in zip(self.images, self.relations, self.objects)]
        self.prompts = [prompt(relations = relations) for relations in self.relations]

        self.num = len(self.images)
        self.resolution = self.images[0].size[0]

        self.augmentations = transforms.Compose(
        [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        # print(f"start getitem {idx}")
        raw_image = self.images[idx]
        annotated_image = self.annotated_images[idx]

        if self.uncod_image_type == "original":
            clean_image = self.augmentations(raw_image.convert("RGB"))
        elif self.uncod_image_type == "annotated":
            clean_image = self.augmentations(annotated_image.convert("RGB"))
        else:
            raise NotImplementedError

        objects = torch.stack([object.tensorize() for object in self.objects[idx]])
        relations = torch.stack([relation.tensorize() for relation in self.relations[idx]])
        bboxes = torch.stack([bbox.tensorize() for bbox in self.bboxes[idx]])
        # print(f"in getitem, shapes are {clean_image.shape}, {objects.shape}, {relations.shape}, {bboxes.shape}")

        generated_prompt = self.prompts[idx]

        # put everythin in a dict
        sample = {
            "clean_image": clean_image,
            "objects": objects,
            "relations": relations,
            "bboxes": bboxes,
            "generated_prompt": generated_prompt,
            "raw_image": raw_image,
            "annotated_image": annotated_image,
        }

        return sample
    

class RelationalDataset2O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/toy_2obj.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=True):
        super().__init__(self.path, uncond_image_type, center_crop, pick_one_relation)



class RelationalDataset1O(RelationalDataset):
    path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
    def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=True):
        super().__init__(self.path, uncond_image_type, center_crop, pick_one_relation)

