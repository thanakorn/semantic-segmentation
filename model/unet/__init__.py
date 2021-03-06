import torch
from torch.nn import Module, Conv2d, ReLU

DEFAULT_BLOCK_CHANNELS = [3,64,128,256,512,1024]

class Block(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 activation=ReLU()):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride)
        self.activation = activation
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return out