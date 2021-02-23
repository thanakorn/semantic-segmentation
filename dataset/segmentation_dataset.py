import torch
import random
import numpy as np
import albumentations as A

from torch import from_numpy
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from skimage.io import imread
from dataset.utils import normalize, dense_encode
from albumentations.pytorch import ToTensorV2

DEFAULT_TRANSFORM = A.Compose([ToTensorV2()])

class SegmentationDataset(Dataset):
    def __init__(self, dataset_root, transform=DEFAULT_TRANSFORM, use_cache=False):
        self.input_files = sorted(listdir(join(dataset_root, 'input')))
        self.input_files = [join(dataset_root, 'input', file) for file in self.input_files]
        self.target_files = sorted(listdir(join(dataset_root, 'target')))
        self.target_files = [join(dataset_root, 'target', file) for file in self.target_files]
        self.transform = transform
        self.use_cache = use_cache
        if self.use_cache:
            self.caches = [None for file in self.input_files]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        if self.use_cache and self.caches[index]:
            image, mask = self.caches[index]
        
        else:
            image = imread(self.input_files[index])
            mask = imread(self.target_files[index])
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = normalize(transformed['image'])
                mask = dense_encode(transformed['mask']).long()
            
            if self.use_cache:
                self.caches[index] = (image, mask)
            
        return image, mask