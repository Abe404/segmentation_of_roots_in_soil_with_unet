# pylint: disable=C0111, W0221, R0902
"""
U-Net architecture based on:
https://arxiv.org/abs/1505.04597
And modified to use Group Normalization
https://arxiv.org/abs/1803.08494


Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn
import torch


class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # need to keep track of output here for up phase.
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )

    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


def crop_tensor(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_height, tensor_width = tensor.size()
    _, _, crop_height, crop_width = target.size()
    left = (tensor_width - crop_height) // 2
    top = (tensor_height - crop_width) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_tensor = tensor[:, :, top: bottom, left: right]
    return cropped_tensor


class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        half_channels = in_channels // 2

        # Now a 2x2 convolution that halves the feature channels
        # this also up-samples
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, half_channels,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )
        # 2 layers of 3x3 conv + relu
        # still uses full channels as half channels
        # is added from down side output
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, half_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(half_channels, half_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, half_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = crop_tensor(down_out, out)
        out = torch.cat([cropped, out], dim=1)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class UNetGN(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        # input image is 572 by 572
        # 3x3 relu conv with 64 kernels
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            # now at 570 x 570 due to valid padding.
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        # need to keep track of output here for up phase.
        """ This down block makes the following transformations.
        to 284x284 with 64 channels
        to 282x282 with 128 channels
        to 280x280 with 128 channels
        """
        self.down1 = DownBlock(64)
        # need to keep track of output here for up phase.
        """ Makes following transformations
        to 140x140 with 128 channels
        to 138x138 with 256 channels
        to 136x136 with 256 channels
        """
        self.down2 = DownBlock(128)
        # need to keep track of output here for up phase

        """ Makes following transformations
        to 68x68 with 256 channels
        to 66x66 with 512 channels
        to 64x64 with 512 channels
        """
        self.down3 = DownBlock(256)
        # need to keep track of output here for up phase

        """ Makes following transformations
        to 32x32 with 512 channels
        to 30 x 30 with 1024 channels
        to 28x28 with 1024 channels
        """
        self.down4 = DownBlock(512)

        """ Makes following transformations
            to is 56x56 with 512 channels (up and conv)
            to is 56x56 with 1024 channels (concat)
            to is 54x54 with 512 channels (conv)
            to is 52x52 with 512 channels (conv)
        """
        self.up1 = UpBlock(1024)
        self.up2 = UpBlock(512)
        self.up3 = UpBlock(256)
        self.up4 = UpBlock(128)
        # output is now at 64x388x388
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.GroupNorm(2, 2)
        )
        # output is now at 2x388x388
        # each layer in the output represents a class 'probability'
        # but these aren't really probabilities as the model is not
        # calibrated.

    def forward(self, x):
        out1 = self.conv_in(x)
        # (1, 64, 568, 568)
        out2 = self.down1(out1)
        # (1, 128, 280, 280)
        out3 = self.down2(out2)
        # (1, 256, 136, 136)
        out4 = self.down3(out3)
        # (1, 512, 64, 64)
        out = self.down4(out4)
        # (1, 1024, 28, 28)
        out = self.up1(out, out4)
        # (1, 512, 52, 52)
        out = self.up2(out, out3)
        # (1, 256, 100, 100)
        out = self.up3(out, out2)
        # (1, 128, 196, 196)
        out = self.up4(out, out1)
        # (1, 64, 388, 388)
        out = self.conv_out(out)
        # each layer in the output represents a class 'probability'
        # but these aren't really probabilities as the model is not
        # calibrated.
        return out
