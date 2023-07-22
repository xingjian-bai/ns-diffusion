import numpy as np
import argparse
import torch
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math

from utils import *

# a function random pick a colour
def random_colour():
    return tuple(np.random.randint(40, 200, 3))

class BoundingBox:
    def __init__(self, pos, pos_type = "cwh", color=None):
        """
        self.pos: [cx, cy, w, h]
        """
        if isinstance(pos, np.ndarray):
            pos = torch.tensor(pos)        
        if isinstance(pos, list):
            pos = torch.tensor(pos)
        
        assert len(pos) == 4, f"pos should be a list of length 4, but got {pos}"
        if pos_type == "lurb":
            self.pos = [(pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2, pos[2] - pos[0], pos[3] - pos[1]]
        elif pos_type == "cwh":
            self.pos = pos.clone()
        else:
            raise NotImplementedError
        self.color = random_colour() if color is None else color 
    
    def tensorize(self):
        self.normalize()
        return torch.tensor(self.pos)

    def normalize(self, resolution = 128):
        """
        From [0, 256] to (-1, 1}, from int type to float type
        """
        if self.pos[2] >= 1: # the width is in pixel
            self.pos[0] = float((self.pos[0] / (resolution / 2)) - 1)
            self.pos[1] = float((self.pos[1] / (resolution / 2)) - 1)
            self.pos[2] = float((self.pos[2] / (resolution / 2)) - 1)
            self.pos[3] = float((self.pos[3] / (resolution / 2)) - 1)

    def denormalize(self, resolution = 128):
        """
        From (-1, 1} to [0, 256], from float type to int type
        """
        if self.pos[2] < 1: # the width is in ratio
            self.pos[0] = int((self.pos[0] + 1) * resolution / 2)
            self.pos[1] = int((self.pos[1] + 1) * resolution / 2)
            self.pos[2] = int((self.pos[2] + 1) * resolution / 2)
            self.pos[3] = int((self.pos[3] + 1) * resolution / 2)


    def mse(self, other):
        self.normalize()
        return np.square(np.subtract(self.pos, other.pos)).mean()
    
    def draw(self, image = Image.new("RGB", (128, 128), (255, 255, 255)), color = None):
        self.denormalize()
        image = image.copy()
        draw = ImageDraw.Draw(image)
        #clip to [0, resolution
        left = self.pos[0] - self.pos[2] / 2
        top = self.pos[1] - self.pos[3] / 2
        right = self.pos[0] + self.pos[2] / 2
        bottom = self.pos[1] + self.pos[3] / 2

        # print("draw bbox: ", left, top, right, bottom, "from", self.pos)
        draw.rectangle([left, top, right, bottom], outline=self.color if color is None else color)
        return image

    def __str__(self) -> str:
        return f"BBOX(cx={self.pos[0]}, cy={self.pos[1]}, w={self.pos[2]}, h={self.pos[3]})"

def draw_bboxes(bboxes, image = Image.new("RGB", (128, 128), (255, 255, 255))):
    image = image.copy()
    for bbox in bboxes:
        # print(f"draw_bboxes: bbox = {bbox}")
        if isinstance(bbox, torch.Tensor):
            bbox = BoundingBox(bbox)

        image = bbox.draw(image)
    return image

class Object:
    shapes = ["cube", "sphere", "cylinder", "none"]
    sizes = ["small", "large", "none"]
    colors = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow", "none"]
    materials = ["rubber", "metal", "none"]
    # order: shape, size, color, material

    def __init__(self, shape_idx, size_idx, color_idx, material_idx):
        self.shape_idx = shape_idx
        self.size_idx = size_idx
        self.color_idx = color_idx
        self.material_idx = material_idx

        assert self.color_idx < len(Object.colors), f"color_idx should be less than {len(Object.colors)}, but got {self.color_idx}"
        assert self.material_idx < len(Object.materials), f"material_idx should be less than {len(Object.materials)}, but got {self.material_idx}"
        assert self.shape_idx < len(Object.shapes), f"shape_idx should be less than {len(Object.shapes)}, but got {self.shape_idx}"
        assert self.size_idx < len(Object.sizes), f"size_idx should be less than {len(Object.sizes)}, but got {self.size_idx}"
    
    def tensorize(self):
        ret = torch.tensor([self.shape_idx, self.size_idx, self.color_idx, self.material_idx])
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
    def str_with_break_line(self):
        return f"{Object.sizes[self.size_idx]}\n{Object.colors[self.color_idx]}\n{Object.materials[self.material_idx]}\n{Object.shapes[self.shape_idx]}"


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
    colours = [(255, 0, 0),     # Red, left
                (0, 255, 0),     # Green, right
                (0, 0, 255),     # Blue, front
                (255, 255, 0),   # Yellow, behind
                (0, 255, 255),   # Cyan, below
                (255, 0, 255),   # Magenta, above
                (0, 0, 0),       # Black, none
    ]
    def __init__(self, relation_idx, obj1, obj2, obj_indices, shift = 0):
        self.relation_idx = relation_idx
        self.relation = Relation.relations[relation_idx]
        self.obj1 = obj1
        self.obj2 = obj2
        self.obj_indices = obj_indices
        self.colour = Relation.colours[relation_idx]
        self.shift = shift
    
    def tensorize(self):
        ret = torch.tensor([*self.obj1.tensorize(), *self.obj2.tensorize(), self.relation_idx])
        assert ret.shape == (9,)
        return ret
    
    def indices(self):
        return torch.tensor(self.obj_indices)
    
    def __str__(self) -> str:
        # relation = Relation.relations[self.relation_idx]
        # print(f"in __str__ of Relation, relation_idx = {self.relation_idx}")
        # print(f"?? {self.obj1}")
        # print(f"?? {Relation.rel_to_description}")
        # print(f"?? {self.obj2}")
        return f"{self.obj1} {Relation.rel_to_description[self.relation]} {self.obj2}"
    
    def draw_relation(self, image, thickness=1):
        # calculate the center of the bounding boxes
        c1x, c1y = self.obj1.bbox.pos[0] + self.shift, self.obj1.bbox.pos[1] + self.shift
        c2x, c2y = self.obj2.bbox.pos[0] + self.shift, self.obj2.bbox.pos[1] + self.shift

        draw = ImageDraw.Draw(image)
        draw.line((c1x, c1y, c2x, c2y), fill=self.colour, width=thickness)


        # Calculate the angle of the line
        angle = math.atan2(c2y - c1y, c2x - c1x)

        # Calculate the coordinates for the arrow head
        arrow_size = 5
        cx = c2x - arrow_size * math.cos(angle + math.pi / 6)
        cy = c2y - arrow_size * math.sin(angle + math.pi / 6)
        dx = c2x - arrow_size * math.cos(angle - math.pi / 6)
        dy = c2y - arrow_size * math.sin(angle - math.pi / 6)

        # Draw the arrow head
        draw.polygon([c2x, c2y, cx, cy, dx, dy], fill=self.colour)

        return image
    
def annotate(image, relations, objects):
    image_with_annotation = image.copy()
    for relation in relations:
        image_with_annotation = relation.draw_relation(image_with_annotation)
    for obj in objects:
        image_with_annotation = obj.bbox.draw(image_with_annotation)
    return image_with_annotation

def prompt(objects = [], relations = []):
    return ", AND ".join([str(rel) for rel in relations if rel.relation_idx != 6]) + ", AND ".join([str(obj) for obj in objects])

def gen_rand_object():
    #in list form
    color_idx = np.random.randint(0, len(Object.colors) - 1)
    material_idx = np.random.randint(0, len(Object.materials) - 1)
    shape_idx = np.random.randint(0, len(Object.shapes) - 1)
    size_idx = np.random.randint(0, len(Object.sizes) - 1)
    return [shape_idx, size_idx, color_idx, material_idx]

def gen_rand_relations(generated_object_num, relation_threshold = 0.05):
    relations = np.zeros((generated_object_num, generated_object_num, 7))

    empty_graph = True
    while empty_graph:
        x_axis = np.random.rand(generated_object_num)
        y_axis = np.random.rand(generated_object_num)

        for a in range(generated_object_num):
            for b in range(generated_object_num):
                if a == b:
                    continue
                if x_axis[a] < x_axis[b] - relation_threshold or x_axis[b] < x_axis[a] - relation_threshold or y_axis[a] < y_axis[b] - relation_threshold or y_axis[b] < y_axis[a] - relation_threshold:
                    empty_graph = False
                relations[a][b][0] = x_axis[a] < x_axis[b] - relation_threshold
                relations[a][b][1] = x_axis[b] < x_axis[a] - relation_threshold
                relations[a][b][2] = y_axis[a] < y_axis[b] - relation_threshold
                relations[a][b][3] = y_axis[b] < y_axis[a] - relation_threshold
    return relations

def draw_scene_graph(objects, relations, relations_ids = None, resolution = 128):
    if isinstance(objects[0], torch.Tensor):
        objects = [Object(*obj) for obj in objects]
    if isinstance(relations[0], torch.Tensor):
        new_relations = []
        print("before:", relations, relations_ids)
        for i in range(len(relations)):
            a, b = relations_ids[i]
            if a >= b:
                continue 
            new_relations.append(Relation(relations[i][-1], objects[int(relations_ids[i][0])], objects[int(relations_ids[i][1])], relations_ids[i]))
        relations = new_relations
    import networkx as nx
    import matplotlib.pyplot as plt

    scene_graph = nx.DiGraph()
    # print("objects: ", objects)

    # matrix of string
    adj_matrix = {}
    # for object in objects:
    #     print(f"draw node {str(object)}")
    #     object_break_line = str(object).replace(' ', '\n')
    #     scene_graph.add_node(str(object))
    for (i, relation) in enumerate(relations):
        a = relation.obj1.str_with_break_line() + str(relation.obj_indices[0].item())
        b = relation.obj2.str_with_break_line() + str(relation.obj_indices[1].item())
        # test if (a, b) is in adj_matrix
        # print(f"adding potential edges: {a}, {b}")
        if (a, b) not in adj_matrix.keys():
            adj_matrix[(a, b)] = relation.relation
        else:
            adj_matrix[(a, b)] += f" & {relation.relation}"
    for (a, obj1) in enumerate(objects):
        for (b, obj2) in enumerate(objects):
            if a >= b:
                continue
            a_str = obj1.str_with_break_line() + str(a)
            b_str = obj2.str_with_break_line() + str(b)
            # print("checking edges for ", a_str, b_str)
            if (a_str, b_str) in adj_matrix:
                # print("real draw: ", a_str, b_str, adj_matrix[(a_str, b_str)])
                scene_graph.add_edge(a_str, b_str, label=adj_matrix[(a_str, b_str)])

    # small dpi
    plt.figure(figsize=(4, 4), dpi=100)
    pos = nx.spring_layout(scene_graph)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(scene_graph, pos, node_size=500)
    # edges
    nx.draw_networkx_edges(scene_graph, pos, width=3)
    # labels
    # node_colors = {str(obj): 'green' for obj in objects}
    nx.draw_networkx_labels(scene_graph, pos, font_size=10, font_family='sans-serif')
    nx.draw_networkx_edge_labels(scene_graph, pos, edge_labels=nx.get_edge_attributes(scene_graph, 'label'))

    # Save the current figure to a BytesIO object
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi = 100)
    # Closing the figure so that it won't be displayed in your environment
    plt.close()
    buf.seek(0)

    # Create a PIL image from the BytesIO object
    img = Image.open(buf)

    # Now img is a PIL image
    return img



class RelationalDataset(Dataset):
    def formula_to_image(self, raw_relations, raw_objects):
        """
            Picks some suitable relations from raw_relations
        """
        relations = []
        for i in range (len(raw_relations)):
            for j in range (len(raw_relations[i])):
                if i == j:
                    continue 
                ij_rels = np.nonzero(raw_relations[i][j])[0]
                if self.pick_one_relation:
                    idx = np.random.choice(ij_rels) if len(ij_rels) > 0 else 6
                    relations.append(Relation(idx, raw_objects[i], raw_objects[j], [i, j]))
                else:
                    for (id, k) in enumerate(ij_rels):
                        relations.append(Relation(k, raw_objects[i], raw_objects[j], [i, j], shift = (id - len(ij_rels) / 2) * 5))
        return relations
                
    def __init__(
        self,
        data_path,
        split = "train",
        test_size = 0.2,
        uncond_image_type="original",
        center_crop=True,
        pick_one_relation=False,
        image_path = None,
        num_upperbound = None,
        generated_object_num = 3,
        generated_data_num = 1000,
        pos_type = "lurb"
    ):
        self.center_crop = center_crop
        self.pick_one_relation = pick_one_relation
        self.uncod_image_type = uncond_image_type

        if data_path != "nothing":  
            self.data = np.load(data_path)
        else:
            self.data = {}
            self.data['objects'] = [] #shape (num_data, num_objects, 4)
            self.data['bboxes'] = [] #shape (num_data, num_objects, 4)
            self.data['relations'] = [] #shape (num_data, num_objects, num_objects, 7)


            for i in range(generated_data_num):
                objects = np.array([gen_rand_object() for _ in range(generated_object_num)])
                relations = gen_rand_relations(generated_object_num)
                bboxes = np.random.uniform(-1, 1, (generated_object_num, 4))
                #turn bboxes to Double type
                bboxes = bboxes.astype(np.double)
                
                self.data['objects'].append(objects)
                self.data['relations'].append(relations)
                self.data['bboxes'].append(bboxes)
                # print("Generating data: ", objects, relations)
            
            self.data['objects'] = np.array(self.data['objects'])
            self.data['relations'] = np.array(self.data['relations'])
            self.data['bboxes'] = np.array(self.data['bboxes'])
            pos_type = "cwh"

        train_indices, test_indices = train_test_split(
            range(len(self.data['objects'])), 
            test_size=test_size, 
            random_state=42
        )
        if split == 'train':
            indices = train_indices
        elif split == 'test':
            indices = test_indices
        else:
            raise ValueError(f"Invalid split argument: '{split}', expected 'train' or 'test'.")

        if num_upperbound is not None and num_upperbound < len(indices):
            # random shuffle and take the first num_upperbound
            # np.random.shuffle(indices)
            indices = indices[:num_upperbound]
        
        colours = [(166, 0, 0), (0, 166, 0), (0, 0, 166), (166, 166, 0), (166, 0, 166), (0, 166, 166), (166, 166, 166)]
        self.objects = [[Object(*obj) for obj in objects] for objects in self.data['objects'][indices]]
        self.bboxes = [[BoundingBox(bbox, pos_type=pos_type, color = colours[i % len(colours)]) for i, bbox in enumerate(bboxes) ] for bboxes in self.data['bboxes'][indices]]

        # print("boxes[0]", self.bboxes[0][0].tensorize(), self.data['bboxes'][indices][0][0])

        for objects, bboxes in zip(self.objects, self.bboxes):
            for obj, bbox in zip(objects, bboxes):
                obj.equip_bbox(bbox)
        
        if "images" in self.data.keys():
            print("Loading images from data")
            self.images = [Image.fromarray(image) for image in self.data['images'][indices]]
        elif image_path is not None:
            print("Loading images from image_path")
            self.image_data = np.load(image_path)
            self.images = [Image.fromarray(image) for image in self.image_data['images'][indices]]
        else:
            print("Empty images")
            self.images = [Image.new("RGB", (128, 128), (255, 255, 255)) for _ in range(len(indices))]

        self.all_raw_relations = self.data['relations'][indices]
        self.relations = [self.formula_to_image(raw_relations, raw_objects) for raw_relations, raw_objects in zip(self.data['relations'][indices], self.objects)]
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
        """
        Returns:
            clean_image: Tensor of shape (3, resolution, resolution)
            objects: Tensor of shape (num_objects, 4)
            relations: Tensor of shape (num_relations, 9)
            bboxes: Tensor of shape (num_objects, 4)
            generated_prompt: str
            annotated_image: Tensor of shape (3, resolution, resolution)
            relations_ids: Tensor of shape (num_relations, 2)
        """
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
        if len(self.relations[idx]) > 0:
            relations = torch.stack([relation.tensorize() for relation in self.relations[idx]])
            relations_ids = torch.stack([relation.indices() for relation in self.relations[idx]])
        else:
            relations = torch.zeros((0, 7))
            relations_ids = torch.zeros((0, 2))
        bboxes = torch.stack([bbox.tensorize() for bbox in self.bboxes[idx]])
        # print(f"in getitem, shapes are {clean_image.shape}, {objects.shape}, {relations.shape}, {bboxes.shape}")

        generated_prompt = self.prompts[idx]

        # put everythin in a dict

        annotated_image_tensor = pil_to_tensor(annotated_image)
        raw_image_tensor = pil_to_tensor(raw_image)
        
        return clean_image, objects, relations, bboxes, generated_prompt, raw_image, raw_image_tensor, relations_ids
    

# class RelationalDataset2O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_2objs_imgs/combined_file.npz'

#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=True):
#         super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)



# class RelationalDataset1O(RelationalDataset):
#     path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj.npz'
#     image_path = '/viscam/projects/ns-diffusion/dataset/clevr_rel_1obj_imgs/1.npz'
#     def __init__(self, uncond_image_type="original", center_crop=True, pick_one_relation=True):
#         super().__init__(self.path, uncond_image_type=uncond_image_type, center_crop=center_crop, pick_one_relation=pick_one_relation, image_path=self.image_path)
    
