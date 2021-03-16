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
from dataset.cache import Cache, FixedSizeCache

DEFAULT_TRANSFORM = A.Compose([ToTensorV2()])

class CityScapesDataset(Dataset):
    def __init__(self, dataset_root, transform=DEFAULT_TRANSFORM, cache: Cache = FixedSizeCache()):
        self.files = sorted(listdir(dataset_root))
        self.files = [join(dataset_root, file) for file in self.files]
        self.transform = transform
        self.cache = cache

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.cache.contain(index):
            return self.cache.get(index)

        image = imread(self.files[index])
        input, label = image[:, :256, :], image[:, 256:, :]

        transformed = self.transform(image=input, mask=label)
        input = normalize(transformed['image'])
        label = dense_encode(transformed['mask']).long()

        self.cache.put(index, (input, label))

        return image, label
        
if __name__=='__main__':
    dataset = CityScapesDataset('./data/cityscapes')
    img, target = dataset.__getitem__(0)