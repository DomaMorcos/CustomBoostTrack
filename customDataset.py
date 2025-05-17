import os
import pdb

import torch
import cv2
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from yolox.data import ValTransform
import dataset

def get_dataset(dataset_path, workers=4):
    dataset = CustomDataset(dataset_path)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "sampler": sampler,
        "batch_size": 1,
    }
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return dataloader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        self.imgs = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.imgs.sort()
        self.preproc=dataset.ValTransform(
            rgb_means=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        np_img = cv2.imread(img_path)
        if np_img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        # get size of image
        height, width, _ = np_img.shape
        tensor, target  = self.preproc(np_img, None, (height, width))
        return (tensor, np_img), target, (height, width, index+1, np.nan, "test"), index+1

