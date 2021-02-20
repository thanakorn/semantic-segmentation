import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import interpolate
from model.unet import DEFAULT_BLOCK_CHANNELS
from model.unet.encoder import Encoder
from model.unet.decoder import Decoder

class UNet(Module):
    def __init__(self, num_classes=1, output_size=(572, 572), block_channels=DEFAULT_BLOCK_CHANNELS):
        super().__init__()
        encoder_channels = block_channels
        decoder_channels = block_channels[1:][::-1]
        self.encoder = Encoder(block_channels=encoder_channels)
        self.decoder = Decoder(block_channels=decoder_channels)
        self.head = Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.output_size = output_size

    def forward(self, x):
        encoded_features = self.encoder(x)
        out = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        out = self.head(out)
        if out.shape != self.output_size:
            out = interpolate(out, self.output_size)
        return out
        