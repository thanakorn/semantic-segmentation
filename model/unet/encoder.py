import torch
from torch.nn import Module, ModuleList, MaxPool2d
from model.unet import *
from model.unet.block import Block

class Encoder(Module):
    def __init__(self, block_channels=DEFAULT_BLOCK_CHANNELS, pool=MaxPool2d(2)):
        super().__init__()
        self.blocks = ModuleList([Block(block_channels[i], block_channels[i+1]) for i in range(len(block_channels) - 1)])
        self.pool = pool

    def forward(self, x):
        features = []
        out = x
        for block in self.blocks:
            out = block(out)
            features.append(out)
            out = self.pool(out)
        return features