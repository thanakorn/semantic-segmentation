import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from skimage.io import imread
import torchvision.transforms as transforms

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
        x = self.transform(imread(self.input_files[index])).float()
        y = self.transform(imread(self.target_files[index])).squeeze().long()
        return x, y