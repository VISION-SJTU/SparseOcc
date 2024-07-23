import torch
from torch import nn
from torch.nn import functional as F
from mmcv.cnn import Conv3d, ConvModule
from mmcv.cnn import build_norm_layer

from mmdet.models import NECKS
from mmcv.runner import BaseModule, ModuleList

import spconv.pytorch as spconv
import torch_scatter

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True


# designed for multi-scale feature aggregation: 
# each pixel within every level will access multi-scale features by interpolation
@NECKS.register_module()
class SparseFeaturePyramid(BaseModule):
    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            feat_channels=256,
            out_channels=256,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            init_cfg=None,
            up_kernel_size=(2, 2, 2),
            up_stride=(2, 2, 2),
            up_padding=(0, 0, 0),
            fix_interpolation=False):
        
        super().__init__(init_cfg=init_cfg)
        
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = 3
        self.feat_channels = feat_channels
        self.fix_interpolation = fix_interpolation
        self.up_kernel_size = up_kernel_size
        assert self.num_encoder_levels >= 1
        
        # build input conv for channel adapation
        # from top to down (low to high resolution)
        input_conv_list = []
        for i in range(self.num_input_levels - 1,
                self.num_input_levels - self.num_encoder_levels - 1, -1):
            input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels[i], 
                                  feat_channels, 
                                  kernel_size=1, 
                                  stride=1,
                                  padding=1,
                                  bias=False),
                build_norm_layer(norm_cfg, feat_channels)[1],
            )
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)
        
        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1, -1):
            lateral_conv = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels[i],
                                  feat_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=True),
                build_norm_layer(norm_cfg, feat_channels)[1],
            )
            
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv3d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        
        self.up_sample = ModuleList()
        for i in range(1, self.num_encoder_levels):
            self.up_sample.append(
                spconv.SparseSequential(
                    spconv.SparseConvTranspose3d(feat_channels, feat_channels, 
                                                 kernel_size=up_kernel_size, 
                                                 stride=up_stride, 
                                                 padding=up_padding, bias=False),
                    build_norm_layer(norm_cfg, feat_channels)[1],
                )
            )
        
        self.init_weights()
    
    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.fix_interpolation:
            for m in self.up_sample:
                for subm in m:
                    if isinstance(subm, spconv.SparseConvTranspose3d):
                        with torch.no_grad():
                            norm_value = 1 / (self.up_kernel_size[0]*
                                              self.up_kernel_size[1]*
                                              self.up_kernel_size[2])
                            subm.weight[:] = norm_value
                        for param in subm.parameters():
                            param.requires_grad = False
    
    def forward(self, feats):        
        outs_1 = []
        for i in range(self.num_encoder_levels):
            # from low to high resolution
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            outs_1.append(feat_projected)
        
        # Top to down aggregation&interpolation
        last_layer_feat = outs_1[-1].features
        last_layer_bcoord = outs_1[-1].indices
        for i in range(self.num_encoder_levels-2, -1, -1):
            # sparse_out = self.down_samples(outs_1[i+1])
            # down sample
            down_coord = last_layer_bcoord[:, 1:] // 2
            down_bcoord = torch.cat([last_layer_bcoord[:, 0:1], down_coord], dim=1)

            # fuse
            fused_bcoord = torch.cat([down_bcoord, outs_1[i].indices], dim=0)
            fused_feat = torch.cat([last_layer_feat, outs_1[i].features], dim=0)

            last_layer_feat = fused_feat
            last_layer_bcoord = fused_bcoord

        bcoord_new, inv = torch.unique(fused_bcoord, return_inverse=True, dim=0)
        feat_new = torch_scatter.scatter_mean(fused_feat, inv, dim=0)

        outs_2 = []
        up_feat = feat_new[inv]
        for i in range(self.num_encoder_levels-2, -1, -1):
            # up sample
            residual = outs_1[self.num_encoder_levels-2-i]
            cur_bcoord = residual.indices
            cur_feat = up_feat[-cur_bcoord.shape[0]:, :]

            # fuse
            residual = residual.replace_feature(residual.features + cur_feat)

            up_feat = up_feat[:-cur_bcoord.shape[0], :]
            outs_2.append(residual)
        
        residual = outs_1[-1]
        residual = residual.replace_feature(residual.features + up_feat)
        outs_2.append(residual)

        # Down to top interpolation
        outs_d2t = [{'feat': outs_2[0].features, 'coord': outs_2[0].indices.type(torch.int64)}]
        for i in range(1, self.num_encoder_levels):
            sparse_out = self.up_sample[i-1](outs_2[i-1])
            sparse_feats = [outs_2[i].features, sparse_out.features]
            sparse_coords = [outs_2[i].indices, sparse_out.indices]

            sparse_feats = torch.cat(sparse_feats, dim=0)
            sparse_coords = torch.cat(sparse_coords, dim=0)
            coord_new, unq_inv = torch.unique(sparse_coords, return_inverse=True, return_counts=False, dim=0)
            coord_new = coord_new.type(torch.int64)
            feats_new = torch_scatter.scatter_mean(sparse_feats, unq_inv, dim=0)

            out_dict = {
                'feat': feats_new,
                'coord': coord_new,
            }
            outs_d2t.append(out_dict)

        decoder_outs = []
        for i in range(self.num_encoder_levels):
            spatial_shape = [outs_1[i].batch_size] + outs_1[i].spatial_shape + [self.feat_channels]
            sparse_feat = outs_d2t[i]['feat']
            sparse_coord = outs_d2t[i]['coord']
            dense_out = torch.zeros(*spatial_shape, dtype=sparse_feat.dtype, device=sparse_feat.device)
            dense_out[sparse_coord[:, 0], sparse_coord[:, 1], sparse_coord[:, 2], sparse_coord[:, 3], :] = sparse_feat
            decoder_outs.append(dense_out.permute(0, 4, 1, 2, 3))

        # build FPN path
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x).dense()
            
            y = cur_feat + F.interpolate(
                decoder_outs[-1],
                size=cur_feat.shape[-3:],
                mode='trilinear',
                align_corners=False,
            )

            y = self.output_convs[i](y)
            decoder_outs.append(y)
        
        decoder_outs[-1] = self.mask_feature(decoder_outs[-1])

        return decoder_outs[::-1]
    
