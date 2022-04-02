# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import torch
from torch import nn

__all__ = [
    "SpatialFeatureTransformLayer", "SpatialFeatureTransformResidualBlock", "UpsampleBlock",
    "SFTMD",
]


# References code from `https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/models/modules/sftmd_arch.py`
class SpatialFeatureTransformLayer(nn.Module):
    def __init__(self, channels: int, transform_channels: int) -> None:
        super(SpatialFeatureTransformLayer, self).__init__()
        self.add_unit = nn.Sequential(
            nn.Conv2d(channels + transform_channels, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, channels, (3, 3), (1, 1), (1, 1)),
        )
        self.mul_unit = nn.Sequential(
            nn.Conv2d(channels + transform_channels, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, channels, (3, 3), (1, 1), (1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, input_feature: torch.Tensor, kernel_feature: torch.Tensor) -> torch.Tensor:
        features = torch.cat([input_feature, kernel_feature], dim=1)
        add_unit = self.add_unit(features)
        mul_unit = self.mul_unit(features)

        out = torch.mul(mul_unit, features)
        out = torch.add(out, add_unit)

        return out


# References code from `https://github.com/yuanjunchai/IKC/blob/2a846cf1194cd9bace08973d55ecd8fd3179fe48/codes/models/modules/sftmd_arch.py`
class SpatialFeatureTransformResidualBlock(nn.Module):
    def __init__(self, channels: int, transform_channels: int) -> None:
        super(SpatialFeatureTransformResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.sft_layer1 = nn.Sequential(
            SpatialFeatureTransformLayer(channels, transform_channels),
            nn.ReLU(True),
        )
        self.sft_layer2 = nn.Sequential(
            SpatialFeatureTransformLayer(channels, transform_channels),
            nn.ReLU(True),
        )

    def forward(self, input_feature: torch.Tensor, kernel_feature: torch.Tensor) -> torch.Tensor:
        identity = input_feature

        out = self.sft_layer1(input_feature, kernel_feature)
        out = self.sft_layer2(self.conv1(out), kernel_feature)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class SFTMD(nn.Module):
    def __init__(self, upscale_factor: int, transform_channels: int = 10) -> None:
        super(SFTMD, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        )

        # Trunk
        trunk = []
        for _ in range(16):
            trunk.append(SpatialFeatureTransformResidualBlock(64, transform_channels))
        self.trunk = nn.Sequential(*trunk)

        # Spatial Feature Transform Layer
        self.sft = SpatialFeatureTransformLayer(64, transform_channels)

        # Second layer
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))

    def forward(self, x: torch.Tensor, batch_kernel: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.size()
        batch_size, kernel_length = batch_kernel.size()
        kernel = batch_kernel.view((batch_size, kernel_length, 1, 1)).expand((batch_size, kernel_length, height, width))

        out1 = self.conv1(x)
        out = self.trunk(out1, kernel)
        out = self.sft(out, kernel)
        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = out.clamp_(0.0, 1.0)

        return out
