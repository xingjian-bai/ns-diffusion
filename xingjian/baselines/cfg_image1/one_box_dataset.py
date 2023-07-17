import torch
from torch.utils.data import Dataset


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
        return clean_image, objects, relations, relations_ids, annotated_image_tensor

def collate_adapted(batch):
    clean_image_batch = []
    objects_batch = []
    relations_batch = []
    relations_ids_batch = []
    annotated_image_tensor_batch = []
    
    for (clean_image, objects, relations, relations_ids, annotated_image_tensor) in batch:
        clean_image_batch.append(clean_image)
        objects_batch.append(objects)
        relations_batch.append(relations)
        relations_ids_batch.append(relations_ids)
        annotated_image_tensor_batch.append(annotated_image_tensor)
        
    return clean_image_batch, objects_batch, relations_batch, relations_ids_batch, annotated_image_tensor_batch
