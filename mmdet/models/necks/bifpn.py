import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmcv.cnn import ConvModule
from ..builder import NECKS
import torch

eps = 0.0001 #启用加权feature fusion时用到的参数


@MODELS.register_module() # 注册器，不多说
class BIFPN(nn.Module):
    def __init__(self,
                 in_channels, # 每个尺度的输入通道数, 也是 backbone 的输出通道数.
                 out_channels,# fpn 的输出通道数, 所有尺度的输出通道数相同, 都是一个值.
                 num_outs, # 输出 特征 层的个数.(可以附加额外的层, num_outs 不一定等于 in_channels)
                 start_level=0, # 使用 backbone 的起始 stage 索引, 默认为 0.
                 end_level=-1, # 使用 backbone 的终止 stage 索引。默认为 -1, 代表到最后一层(包括)全使用.
                 stack=1,# BiFPN的
                 add_extra_convs=True,# 可以是 bool 或 str:bool 代表是否在原始的特征图上添加额外的卷积层.(默认值: False)如果为True，则在最顶层的feature map上添加额外的卷积层,具体的模式需要 extra_convs_on_inputs 指定.
                 extra_convs_on_inputs=False,# 若为True 则`add_extra_convs='on_input'    若为False 则 `add_extra_convs='on_output'   'on_output': 最高层的经过 conv 的 lateral 结果作为 extra 的输入
                 relu_before_extra_convs=False, # 是否在 extra conv 前使用 relu. (默认值: False)
                 no_norm_on_lateral=True, # 是否对 FPN的1×1卷积 使用 bn. 
                 conv_cfg=None, # 构建 conv 层的 config 字典.
                 norm_cfg=dict(type='BN', requires_grad=False) # 构建 bn 层的 config 字典.
                 ):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels) # 输入 stage 的个数
        self.num_outs = num_outs # 输出 stage 的个数
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1: # 如果是 -1 表示使用 backbone 最后一个 feature map 的索引作为最终的索引
            self.backbone_end_level = self.num_ins 
            assert num_outs >= self.num_ins - start_level # 因为还有 extra conv 所以存在 num_outs > num_ins - start_level 的情况
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels) # 如果 end_level < inputs, 说明不使用 backbone 全部的尺度, 并且不会提供额外的层
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList() # lateral_convs是一个list，存的就是1x1的卷积
        self.stack_bifpn_convs = nn.ModuleList()

        self.extra_levels = num_outs - self.backbone_end_level + self.start_level # 看除了backbone之后额外底层还有几层

        if self.add_extra_convs:
            self.extra_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
             # 构建1×1卷积模块，例子如下：l_conv=ConvModule((conv): Conv2d(3, 11, kernel_size=(1, 1), stride=(1, 1)))
            l_conv = ConvModule(in_channels[i],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,inplace=False)
            self.lateral_convs.append(l_conv)

        if self.extra_levels > 0:
            for i in range(self.extra_levels):
                in_channels = self.in_channels[self.backbone_end_level - 1]
                extra_l_conv = ConvModule(in_channels,out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,activation=None,inplace=False)
                self.lateral_convs.append(extra_l_conv)

                if self.add_extra_convs:
                    extra_conv = ConvModule(in_channels,in_channels,3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,activation=self.activation,inplace=False)
                    self.extra_convs.append(extra_conv)

        for ii in range(stack): # 创建stack个bifpn模块
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,levels=self.backbone_end_level - self.start_level + self.extra_levels,conv_cfg=conv_cfg,norm_cfg=norm_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = list(inputs)

        # add extra
        if self.extra_levels > 0:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.extra_levels):
                if self.add_extra_convs:
                    inputs.append(self.extra_convs[i](inputs[-1]))
                else:
                    inputs.append(F.max_pool2d(inputs[-1], 1, stride=2))

        laterals = [lateral_conv(inputs[i + self.start_level])for i, lateral_conv in enumerate(self.lateral_convs)] # 把指定的每个Stage输出的经过1x1卷积：
        # part 1: build top-down and down-top path with stack
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        return tuple(outs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # 给FPN卷积层初始化权重
                xavier_init(m, distribution='uniform')


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init)) # 构造一个可以训练的矩阵权重
        print(self.w1)
        self.relu1 = nn.ReLU(False)
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        print(self.w2)
        self.relu2 = nn.ReLU(False)
        for jj in range(2):
            for i in range(self.levels - 1):  # self.levels = 2
                fpn_conv = nn.Sequential(
                    ConvModule(channels,channels,3,padding=1,groups=channels,conv_cfg=conv_cfg,norm_cfg=None,inplace=False), # 构建3×3卷积
                    ConvModule(channels,channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,inplace=False) # 构建1×1卷积
                    )
                self.bifpn_convs.append(fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1_relu = self.relu1(self.w1)
        w1 = w1_relu / torch.sum(w1_relu, dim=0) + eps  # 归一化
        w2_relu = self.relu2(self.w2)
        w2 = w2_relu / torch.sum(w2_relu, dim=0) + eps
        # build top-down
        kk = 0
        # pathtd = inputs copy is wrong
        pathtd = [inputs[levels - 1]]
        #        for in_tensor in inputs:
        #            pathtd.append(in_tensor.clone().detach())
        for i in range(levels - 1, 0, -1):
            _t = w1[0, kk] * inputs[i - 1] + w1[1, kk] * F.interpolate(pathtd[-1], scale_factor=2, mode='nearest')
            pathtd.append(self.bifpn_convs[kk](_t))
            del (_t)
            kk = kk + 1
        jj = kk
        pathtd = pathtd[::-1]
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = w2[0, i] * inputs[i + 1] + w2[1, i] * nn.Upsample(scale_factor=0.5)(pathtd[i]) + w2[2, i] * pathtd[i + 1]
            pathtd[i + 1] = self.bifpn_convs[jj](pathtd[i + 1])
            jj = jj + 1

        pathtd[levels - 1] = w1[0, kk] * inputs[levels - 1] + w1[1, kk] * nn.Upsample(scale_factor=0.5)(pathtd[levels - 2])
        pathtd[levels - 1] = self.bifpn_convs[jj](pathtd[levels - 1])
        return pathtd

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')