import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from skimage.io import imread

DEFAULT_TRANSFORM = transforms.Compose([
            transforms.ToTensor()
        ])

class SegmentationDataset(Dataset):
    def __init__(self, dataset_root, transform=DEFAULT_TRANSFORM):
        self.input_files = sorted(listdir(join(dataset_root, 'input')))
        self.input_files = [join(dataset_root, 'input', file) for file in self.input_files]
        self.target_files = sorted(listdir(join(dataset_root, 'target')))
        self.target_files = [join(dataset_root, 'target', file) for file in self.target_files]
        self.transform = transform

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, index):
        x, y = imread(self.input_files[index]), imread(self.target_files[index])
        if self.transform:
            x = self.transform(x).float()
            y = self.transform(y).squeeze().long()
        return x, y