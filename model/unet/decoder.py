import torch
import torchvision
from torch.nn import Module, ModuleList, ConvTranspose2d
from model.unet import *
from torchvision.transforms.functional import center_crop

class Decoder(Module):
    def __init__(self, block_channels=DEFAULT_BLOCK_CHANNELS[::-1]):
        super().__init__()
        self.upconvs = ModuleList([ConvTranspose2d(block_channels[i], block_channels[i+1], kernel_size=2, stride=2) for i in range(len(block_channels) - 1)])
        self.blocks = ModuleList([Block(block_channels[i], block_channels[i+1]) for i in range(len(block_channels) - 1)])

    def forward(self, x, encoded_features):
        out = x
        for upconv, block, encoded_feature in zip(self.upconvs, self.blocks, encoded_features):
            out = upconv(out)
            _, _, height, width = out.shape
            encoded_feature = center_crop(encoded_feature, [height, width])
            out = torch.cat([out, encoded_feature], dim=1)
            out = block(out)
        return out