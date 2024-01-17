# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Model construction utils

This module provides a convenient way to create different topologies
based around UNet.

"""
import tensorflow as tf
from model.layers import InputBlock, DownsampleBlock, BottleneckBlock, UpsampleBlock, OutputBlock


class Unet(tf.keras.Model):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Source:
        https://arxiv.org/pdf/1505.04597

    """
    def __init__(self):
        super().__init__(self)
        self.input_block = InputBlock(filters=32)
        self.bottleneck = [BottleneckBlock(filters,idx)
                            for idx, filters in enumerate([256])]#BottleneckBlock(1024)
        self.output_block = OutputBlock(filters=1, n_classes=2)

        self.down_blocks = [DownsampleBlock(filters, idx)
                            for idx, filters in enumerate([32, 64, 128])]

        self.up_blocks = [UpsampleBlock(filters, idx)
                          for idx, filters in enumerate([128, 64, 32])]

    def call(self, x, training=True):
        skip_connections = []
        #out, residual = self.input_block(x)
        #skip_connections.append(residual)
        out  = x
        for i, down_block in enumerate(self.down_blocks):
            #print(i)
            out, residual = down_block(out)
            #print('encoder:',i,out.shape, residual.shape)
            skip_connections.append(residual)

        for bn_block in self.bottleneck:
            out, residual = bn_block(out)
            #print('bottleneck:',i,out.shape, residual.shape)
            #skip_connections.append(residual)
        #out = self.bottleneck(out, training)
        #print('len of skip_connections',len(skip_connections))
        for up_block in self.up_blocks:
            out = up_block(out, skip_connections.pop())

        out = self.output_block(out, None)
        #print('unet output size:',out.shape)
        return out
