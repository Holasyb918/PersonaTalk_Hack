# coding: utf-8

"""
Appearance extractor(F) defined in paper, which maps the source image s to a 2D appearance feature.
"""

import torch
from torch import nn

try:
    from .util import SameBlock2d, DownBlock2d, UpBlock2d
except:
    from util import SameBlock2d, DownBlock2d, UpBlock2d


class AppearanceFeatureExtractor2D(nn.Module):

    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features):
        super(AppearanceFeatureExtractor2D, self).__init__()
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        self.max_features = max_features

        self.first = SameBlock2d(
            image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1)
        )

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2**i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(
                    in_features, out_features, kernel_size=(3, 3), padding=(1, 1)
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, source_image):
        out = self.first(source_image)  # Bx3x256x256 -> Bx64x256x256

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out


class TexDecoder(nn.Module):
    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features):
        super(TexDecoder, self).__init__()
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        # self.up = nn.Upsample(scale_factor=2)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(
                max_features, block_expansion * (2 ** (num_down_blocks - i))
            )
            if i == 0:
                in_features += 3
            out_features = min(
                max_features, block_expansion * (2 ** (num_down_blocks - i - 1))
            )
            # print(i, in_features, out_features)
            up_blocks.append(
                UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1))
            )
        self.up_blocks = nn.ModuleList(up_blocks)
        self.last = nn.Conv2d(
            in_channels=out_features,
            out_channels=image_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.final_act = nn.Tanh()

    def forward(self, f_s, tgt_mesh):
        out = torch.cat((f_s, tgt_mesh), dim=1)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.last(out)
        out = self.final_act(out)
        return out


if __name__ == "__main__":
    encoder = AppearanceFeatureExtractor2D(3, 64, 2, 512)
    decoder = TexDecoder(3, 64, 2, 512)

    encoder_input = torch.randn(1, 3, 256, 256)
    encoder_output = encoder(encoder_input)
    print(encoder_output.shape)

    decoder_input_mesh = torch.randn(1, 3, 64, 64)
    decoder_output = decoder(encoder_output, decoder_input_mesh)
    print(decoder_output.shape)
