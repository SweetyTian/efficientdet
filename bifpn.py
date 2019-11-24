import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
import torch

@NECKS.register_module
class BIFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(
                BiFPNModule(channels=out_channels, start_level=start_level, end_level=self.backbone_end_level))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        outs = self.stack_bifpn_convs(laterals)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.levels = levels
        self.bifpn_convs =nn.ModuleList()
        self.wtd = nn.Parameter(torch.Tensor(5,levels-1))

        for jj in range(2):
            for i in range(self.levels-1):  # 1,2,3
                fpn_conv = ConvModule(
                    channels,
                    channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.bifpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        # build laterals
        used_backbone_levels = self.levels #4
        pathtd = inputs
        # build top-down
        jj=0
        for i in range(used_backbone_levels - 1, 0, -1):
            pathtd[i - 1] = (self.wtd[0,i-1]*pathtd[i - 1] + self.wtd[1,i-1]*F.interpolate(
                pathtd[i], scale_factor=2, mode='nearest'))/(self.wtd[0,i-1]+self.wtd[1,i-1]+1e-5)
            pathtd[i - 1] = self.bifpn_convs[jj](pathtd[i - 1])
            jj=jj+1
        # build down-top
        for i in range(0, used_backbone_levels - 1, 1):
            pathtd[i + 1] = (self.wtd[2, i] * pathtd[i + 1] + self.wtd[3, i] * F.avg_pool2d(pathtd[i], kernel_size=2) +
                             self.wtd[4, i] * inputs[i + 1]) / (self.wtd[2, i] + self.wtd[3, i] + self.wtd[4, i] + 1e-5)
            pathtd[i + 1] = self.bifpn_convs[jj](pathtd[i - 1]) #3,4,5
            jj=jj+1
        return pathtd
