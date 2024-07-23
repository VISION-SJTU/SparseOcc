import numpy as np
import spconv.pytorch as spconv

from torch import nn
from mmdet3d.models.builder import BACKBONES
from mmcv.cnn import build_norm_layer

from projects.mmdet3d_plugin.utils import extract_nonzero_features

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 3, 1), stride=stride,
                             padding=(1, 1, 0), bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, 
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(ResContextBlock, self).__init__()

        self.conv1 = spconv.SparseSequential(
            conv3x3x1(in_filters, in_filters),
            build_norm_layer(norm_cfg, in_filters)[1],
            nn.LeakyReLU(),
            conv1x3(in_filters, out_filters),
            build_norm_layer(norm_cfg, out_filters)[1],
        )

        self.conv2 = spconv.SparseSequential(
            conv3x1(in_filters, in_filters),
            build_norm_layer(norm_cfg, in_filters)[1],
            nn.LeakyReLU(),
            conv3x3x1(in_filters, out_filters),
            build_norm_layer(norm_cfg, out_filters)[1],
        )

        self.init_weights()

    
    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)

        resA = self.conv2(x)
        
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class ContxAggreBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 pooling=True, 
                 indice_key=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),):
        super(ContxAggreBlock, self).__init__()
        self.pooling = pooling

        self.conv1 = spconv.SparseSequential(
            conv3x3x1(in_channels, in_channels),
            build_norm_layer(norm_cfg, in_channels)[1],
            nn.LeakyReLU(),
            conv3x1(in_channels, out_channels),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

        self.conv2 = spconv.SparseSequential(
            conv1x3(in_channels, in_channels),
            build_norm_layer(norm_cfg, in_channels)[1],
            nn.LeakyReLU(),
            conv3x3x1(in_channels, out_channels),
            build_norm_layer(norm_cfg, out_channels)[1]
        )

        if pooling:
            self.pool = spconv.SparseConv3d(out_channels, out_channels, kernel_size=3, stride=2, 
                                            padding=1, indice_key=indice_key, bias=False)
            
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)

        resA = self.conv2(x)

        resA = resA.replace_feature(resA.features + shortcut.features)


        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class ContxAggreBlock2D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=5, 
                 pooling=True, 
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),):
        super().__init__()

        self.pooling = pooling

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, in_channels, kernel_size=(kernel_size, kernel_size, 1), 
                              stride=1, padding=(kernel_size//2, kernel_size//2, 0)),
            build_norm_layer(norm_cfg, in_channels)[1],
            nn.LeakyReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, 1), 
                              stride=1, padding=(kernel_size//2, kernel_size//2, 0)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

        if pooling:
            self.pool = spconv.SparseConv3d(out_channels, out_channels, kernel_size=3, stride=2,
                                            padding=1, bias=False)
        else:
            self.pool = spconv.SubMConv3d(out_channels, out_channels, kernel_size=1, stride=1, 
                                          padding=0, bias=False)

        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv1(x)

        if self.pooling:
            down_out = self.pool(out)
            return down_out, out
        else:
            out = self.pool(out)
            return out

class CompletionLayer(spconv.SparseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    ):
        super(CompletionLayer, self).__init__()

        self.conv = spconv.SparseSequential(
            spconv.SparseConv3d(in_channels, out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, padding=padding, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            nn.LeakyReLU(),
        )

        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.conv(x)
        return out
    

class CompletionBlock(spconv.SparseModule):
    def __init__(self, 
                 in_filters, 
                 out_filters,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 only_xy=False,
                 num_layers=1):
        super(CompletionBlock, self).__init__()

        self.only_xy = only_xy

        self.layers1 = spconv.SparseSequential(
            *[CompletionLayer(in_filters, out_filters if i==num_layers-1 and only_xy else in_filters, 
                              kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), 
                              norm_cfg=norm_cfg) for i in range(num_layers)]
        )

        if not self.only_xy:
            self.layers2 = spconv.SparseSequential(
                CompletionLayer(in_filters, in_filters, kernel_size=(3, 1, 3), 
                                stride=1, padding=(1, 0, 1), norm_cfg=norm_cfg),
                CompletionLayer(in_filters, out_filters, kernel_size=(1, 3, 3), 
                                stride=1, padding=(0, 1, 1), norm_cfg=norm_cfg),
            )
        
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers1(x)
        if not self.only_xy:
            out = self.layers2(out)

        return out


@BACKBONES.register_module()
class SparseLatentDiffuser(nn.Module):
    def __init__(self,
                 output_shape,
                 in_channels=128,
                 norm_cfg=dict(type='BN1d', requires_grad=True),
                 num_layers=[1, 1, 1, 1]
                ):
        super(SparseLatentDiffuser, self).__init__()

        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape

        self.resBlock0 = SparseBasicBlock(in_channels, in_channels, 1, norm_cfg)
        
        self.complBlock1 = CompletionBlock(in_channels, in_channels, norm_cfg=norm_cfg, 
                                           only_xy=False, num_layers=num_layers[0])
        self.aggreBlock1 = ContxAggreBlock(in_channels, in_channels, norm_cfg=norm_cfg, 
                                           pooling=True)
        
        self.complBlock2 = CompletionBlock(in_channels, in_channels, norm_cfg=norm_cfg, 
                                           only_xy=False, num_layers=num_layers[1])
        self.aggreBlock2 = ContxAggreBlock(in_channels, 2 * in_channels, norm_cfg=norm_cfg, 
                                           pooling=True)
        
        self.complBlock3 = CompletionBlock(2 * in_channels, 2 * in_channels, norm_cfg=norm_cfg, 
                                           only_xy=True, num_layers=num_layers[2])
        self.aggreBlock3 = ContxAggreBlock2D(2 * in_channels, 4 * in_channels, kernel_size=5, norm_cfg=norm_cfg, 
                                             pooling=True)
        
        self.complBlock4 = CompletionBlock(4 * in_channels, 4 * in_channels, norm_cfg=norm_cfg, 
                                           only_xy=True, num_layers=num_layers[3])
        self.aggreBlock4 = ContxAggreBlock2D(4 * in_channels, 8 * in_channels, kernel_size=5, norm_cfg=norm_cfg,
                                             pooling=False)
        
        self.init_weights()
    

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, vox_feat):
        vox_coord, pts_feat = extract_nonzero_features(vox_feat)

        input = spconv.SparseConvTensor(pts_feat, vox_coord.int(), self.sparse_shape, 1)

        input = self.resBlock0(input)

        input = self.complBlock1(input)
        down1, skip1 = self.aggreBlock1(input)
        down1 = self.complBlock2(down1)
        down2, skip2 = self.aggreBlock2(down1)
        down2 = self.complBlock3(down2)
        down3, skip3 = self.aggreBlock3(down2)
        down3 = self.complBlock4(down3)
        skip4 = self.aggreBlock4(down3)

        output = [skip1, skip2, skip3, skip4]

        return output
    
