"""3D Sparse U-Net backbone using SpConv."""
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None, normalize_before=True):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


class SpConvUNet(nn.Module):
    """Recursive SpConv U-Net. num_planes=[32, 64, 96, 128, 160]."""

    def __init__(self, num_planes, norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2, indice_key_id=1, normalize_before=True, return_blocks=False):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        blocks = OrderedDict({
            f'block{i}': ResidualBlock(num_planes[0], num_planes[0], norm_fn,
                                       indice_key=f'subm{indice_key_id}',
                                       normalize_before=normalize_before)
            for i in range(block_reps)
        })
        self.blocks = spconv.SparseSequential(blocks)

        if len(num_planes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(num_planes[0]), nn.ReLU(),
                spconv.SparseConv3d(num_planes[0], num_planes[1], kernel_size=2, stride=2,
                                    bias=False, indice_key=f'spconv{indice_key_id}'))

            self.u = SpConvUNet(num_planes[1:], norm_fn, block_reps,
                                indice_key_id=indice_key_id + 1,
                                normalize_before=normalize_before,
                                return_blocks=return_blocks)

            self.deconv = spconv.SparseSequential(
                norm_fn(num_planes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(num_planes[1], num_planes[0], kernel_size=2,
                                           bias=False, indice_key=f'spconv{indice_key_id}'))

            blocks_tail = OrderedDict({
                f'block{i}': ResidualBlock(num_planes[0] * (2 - i), num_planes[0], norm_fn,
                                           indice_key=f'subm{indice_key_id}',
                                           normalize_before=normalize_before)
                for i in range(block_reps)
            })
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs=None):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        return output
