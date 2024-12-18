import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY
from ..attention import SelfAttentionBlock, CrossAttentionBlock
from mvdnet.layers import Conv3d

@ROI_BOX_HEAD_REGISTRY.register()
class MVDNetBoxHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.history_on = cfg.INPUT.HISTORY_ON
        self.num_history = cfg.INPUT.NUM_HISTORY+1
        self.pooler_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        assert num_fc > 0

        for f in input_shape.keys():
            if f.startswith("radar"):
                self.radar_key = f
                self.radar_output_size = input_shape[f].channels * input_shape[f].height * input_shape[f].width
                self.radar_input_channels = input_shape[f].channels

        self.radar_self_attention = SelfAttentionBlock(self.radar_output_size)

        self.tnn = Conv2d(
            in_channels = self.radar_input_channels*2,
            out_channels = self.radar_input_channels,
            kernel_size = 3,
            padding = 1,
            bias=False,
            norm=nn.BatchNorm2d(self.radar_input_channels),
            activation=F.leaky_relu_
        )
        self._output_size = self.radar_output_size
        
        self.fcs = []
        for k in range(num_fc):
            fc = Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        if self.history_on:
            for layer in self.tnns:
                weight_init.c2_msra_fill(layer)
        else:
            weight_init.c2_msra_fill(self.tnn)

    def forward(self, x):
        radar_features = x[self.radar_key]

        radar_x = torch.flatten(radar_features, start_dim=1)
        radar_x = self.radar_self_attention(radar_x)
        feature_x = radar_x.unsqueeze(0).unsqueeze(0)
        feature_x = self.tnn(feature_x)
        fusion_feature = torch.flatten(feature_x, start_dim=1)
        
        for layer in self.fcs:
            fusion_feature = F.leaky_relu_(layer(fusion_feature))
        return fusion_feature

    @property
    def output_size(self):
        return self._output_size