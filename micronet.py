import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN


cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.DEVICE = "cuda"
cfg.MODEL.DEBUG = False
cfg.MODEL.WEIGHT = ""
cfg.MODEL.ACTIVATION = CN()
cfg.MODEL.ACTIVATION.MODULE = "MaxLUConv" # old for mbnetm2 "MaxLUConv"
cfg.MODEL.ACTIVATION.ACT_MAX = 1.0
cfg.MODEL.ACTIVATION.LAST_SE_OUP = False #use se-oup for the last 1x1 conv
cfg.MODEL.ACTIVATION.LINEARSE_BIAS = True
cfg.MODEL.ACTIVATION.INIT_A_BLOCK3 = [1.0, 0.0]
cfg.MODEL.ACTIVATION.INIT_A = [1.0, 0.0]
cfg.MODEL.ACTIVATION.INIT_B = [0.0, 0.0]
cfg.MODEL.ACTIVATION.REDUCTION = 4
cfg.MODEL.ACTIVATION.FC = False
cfg.MODEL.ACTIVATION.ACT = 'relu'

cfg.MODEL.MICRONETS = CN()
cfg.MODEL.MICRONETS.NET_CONFIG = "d12_3322_192_k5"
cfg.MODEL.MICRONETS.STEM_CH = 16
cfg.MODEL.MICRONETS.STEM_DILATION = 1
cfg.MODEL.MICRONETS.STEM_GROUPS = [4, 8]
cfg.MODEL.MICRONETS.STEM_MODE = "default" # defaut/max2
cfg.MODEL.MICRONETS.BLOCK = "MicroBlock1"
cfg.MODEL.MICRONETS.POINTWISE = 'group' #fft/1x1/shuffle
cfg.MODEL.MICRONETS.DEPTHSEP = True # YUNSHENG ADD FOR MUTUAL LEARNING
cfg.MODEL.MICRONETS.SHUFFLE = False
cfg.MODEL.MICRONETS.OUT_CH = 1024
cfg.MODEL.MICRONETS.DROPOUT = 0.0

cfg.OUTPUT_DIR = "."
preset = {
        'm0':
        ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock', 'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy6_exp4_4M_221', 'MODEL.MICRONETS.STEM_CH', '4', 'MODEL.MICRONETS.STEM_GROUPS', '2,2', 'MODEL.MICRONETS.STEM_DILATION', '1', 'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf', 'MODEL.MICRONETS.OUT_CH', '640', 'MODEL.MICRONETS.DEPTHSEP', 'True', 'MODEL.MICRONETS.POINTWISE', 'group', 'MODEL.MICRONETS.DROPOUT', '0.05', 'MODEL.ACTIVATION.MODULE', 'DYShiftMax', 'MODEL.ACTIVATION.ACT_MAX', '2.0', 'MODEL.ACTIVATION.LINEARSE_BIAS', 'False', 'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0', 'MODEL.ACTIVATION.INIT_A', '1.0,1.0', 'MODEL.ACTIVATION.INIT_B', '0.0,0.0', 'MODEL.ACTIVATION.REDUCTION', '8', 'MODEL.MICRONETS.SHUFFLE', 'True'],
        'm1':
        ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock', 'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy6_exp6_6M_221', 'MODEL.MICRONETS.STEM_CH', '6', 'MODEL.MICRONETS.STEM_GROUPS', '3,2', 'MODEL.MICRONETS.STEM_DILATION', '1', 'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf', 'MODEL.MICRONETS.OUT_CH', '960', 'MODEL.MICRONETS.DEPTHSEP', 'True', 'MODEL.MICRONETS.POINTWISE', 'group', 'MODEL.MICRONETS.DROPOUT', '0.05', 'MODEL.ACTIVATION.MODULE', 'DYShiftMax', 'MODEL.ACTIVATION.ACT_MAX', '2.0', 'MODEL.ACTIVATION.LINEARSE_BIAS', 'False', 'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0', 'MODEL.ACTIVATION.INIT_A', '1.0,1.0', 'MODEL.ACTIVATION.INIT_B', '0.0,0.0', 'MODEL.ACTIVATION.REDUCTION', '8', 'MODEL.MICRONETS.SHUFFLE', 'True'],
        'm2':
        ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock', 'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy9_exp6_12M_221', 'MODEL.MICRONETS.STEM_CH', '8', 'MODEL.MICRONETS.STEM_GROUPS', '4,2', 'MODEL.MICRONETS.STEM_DILATION', '1', 'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf', 'MODEL.MICRONETS.OUT_CH', '1024', 'MODEL.MICRONETS.DEPTHSEP', 'True', 'MODEL.MICRONETS.POINTWISE', 'group', 'MODEL.MICRONETS.DROPOUT', '0.1', 'MODEL.ACTIVATION.MODULE', 'DYShiftMax', 'MODEL.ACTIVATION.ACT_MAX', '2.0', 'MODEL.ACTIVATION.LINEARSE_BIAS', 'False', 'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0', 'MODEL.ACTIVATION.INIT_A', '1.0,1.0', 'MODEL.ACTIVATION.INIT_B', '0.0,0.0', 'MODEL.ACTIVATION.REDUCTION', '8', 'MODEL.MICRONETS.SHUFFLE', 'True'],
        'm3':
        ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock', 'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy12_exp6_20M_020', 'MODEL.MICRONETS.STEM_CH', '12', 'MODEL.MICRONETS.STEM_GROUPS', '4,3', 'MODEL.MICRONETS.STEM_DILATION', '1', 'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf', 'MODEL.MICRONETS.OUT_CH', '1024', 'MODEL.MICRONETS.DEPTHSEP', 'True', 'MODEL.MICRONETS.POINTWISE', 'group', 'MODEL.MICRONETS.DROPOUT', '0.1', 'MODEL.ACTIVATION.MODULE', 'DYShiftMax', 'MODEL.ACTIVATION.ACT_MAX', '2.0', 'MODEL.ACTIVATION.LINEARSE_BIAS', 'False', 'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0', 'MODEL.ACTIVATION.INIT_A', '1.0,0.5', 'MODEL.ACTIVATION.INIT_B', '0.0,0.5', 'MODEL.ACTIVATION.REDUCTION', '8', 'MODEL.MICRONETS.SHUFFLE', 'True']
        }


msnx_dy6_exp4_4M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
        #AABCCC
        [2, 1,   8, 3, 2, 2,  0,  4,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 4,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 4,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 4,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 384, 3, 1, 4, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy6_exp6_6M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
        #AABCCC
        [2, 1,   8, 3, 2, 2,  0,  6,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  16, 3, 2, 2,  0,  8,  16,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 16,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 6,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 576, 3, 1, 6, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy9_exp6_12M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
        #AABCCCCC
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
        [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
        [1, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
        [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
        [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
        [1, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
        [2, 1,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
        [1, 1, 128, 3, 1, 6, 12, 12, 128,  8,  8, 2, 2, 1, 2], #96->96(5,16)->576->128(16,8)->128
        [1, 1, 768, 3, 1, 6, 16, 16,   0,  0,  0, 2, 2, 1, 2], #128->128(4,32)->768
        ]
msnx_dy12_exp6_20M_020_cfgs = [
    #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r
    #AABCCCCCCCC
    [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 0, 2, 0, 1], #12->24(0, 0)->48  ->16(8,2)->16
    [2, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 0, 2, 0, 1], #16->32(0, 0)->64  ->24(8,3)->24
    [1, 1,  24, 3, 2, 2,  0, 24,  24,  4,  4, 0, 2, 0, 1], #24->48(0, 0)->96  ->24(8,3)->24
    [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 0, 2, 0, 1], #24->24(2,12)->144  ->32(16,2)->32
    [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 0, 2, 0, 2], #32->32(2,16)->192 ->32(16,2)->32
    [1, 1,  64, 5, 1, 6,  8,  8,  48,  8,  8, 0, 2, 0, 2], #32->32(2,16)->192 ->48(12,4)->48
    [1, 1,  80, 5, 1, 6,  8,  8,  80,  8,  8, 0, 2, 0, 2], #48->48(3,16)->288 ->80(16,5)->80
    [1, 1,  80, 5, 1, 6, 10, 10,  80,  8,  8, 0, 2, 0, 2], #80->80(4,20)->480->80(20,4)->80
    [2, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], #80->80(4,20)->480->128(16,8)->128
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], #120->128(4,32)->720->128(32,4)->120
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], #120->128(4,32)->720->160(32,5)->144
    [1, 1, 864, 3, 1, 6, 12, 12,   0,  0,  0, 0, 2, 0, 2], #144->144(5,32)->864
]

def get_micronet_config(mode):
   # print(mode)
    return eval(mode+'_cfgs')#, len(mode+'_cfgs')

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.) / 6.

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=h_sigmoid(), divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class SwishLinear(nn.Module):
    def __init__(self, inp, oup):
        super(SwishLinear, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

    def forward(self, x):
        return self.linear(x)

class GroupConv(nn.Module):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=self.groups[0]),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):

        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        out = x.view(b, -1, h, w)
        return out

class SpatialSepConvSF(nn.Module):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias=False, groups=1
            ),
            nn.BatchNorm2d(oup1),
            nn.ReLU(oup1),
            nn.Conv2d(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=oup1
            ),
            nn.BatchNorm2d(oup1*oup2),
            nn.ReLU(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Module):
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand
        hidden_dim = inp*exp1
        oup = inp*exp1*exp2
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp*exp1, 
                (kernel_size, 1), 
                (stride, 1), 
                (kernel_size//2, 0), 
                bias=False, groups=inp
            ),
            nn.BatchNorm2d(inp*exp1),
            nn.Conv2d(hidden_dim, oup,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=hidden_dim
            ),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DYShiftMax(nn.Module):
    def __init__(self, inp, oup, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=None, expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU(inplace=True) if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        self.gc = inp//self.g
        index=torch.Tensor(range(inp)).view(1,inp,1,1)
        index=index.view(1,self.g,self.gc,1,1)
        indexgs = torch.split(index, [1, self.g-1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc-1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(inp).type(torch.LongTensor)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)
        y = (y-0.5) * self.act_max

        n2, c2, h2, w2 = x_out.size()
        x2 = x_out[:,self.index,:,:]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out

def get_pointwise_conv(mode, inp, oup, hiddendim, groups):

    return GroupConv(inp, oup, groups)


def get_act_layer(inp, oup, mode='SE1', act_relu=True, act_max=2, act_bias=True, init_a=[1.0, 0.0], reduction=4, init_b=[0.0, 0.0], g=None, act='relu', expansion=True):
    
    layer = DYShiftMax(inp, oup, act_max=act_max, act_relu=act_relu, init_a=init_a, reduction=reduction, init_b=init_b, g=g, expansion=expansion)
    return layer

class MaxGroupPooling(nn.Module):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.size()

        # reshape
        y = x.view(b, c // self.channel_per_group, -1, h, w)
        out, _ = torch.max(y, dim=2)
        return out

class StemLayer(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, groups=(4,4)):
        super(StemLayer, self).__init__()
        
        g1, g2 = groups 
        self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.Sequential()
            )
           
    def forward(self, x):
        out = self.stem(x)    
        return out
 
class DYMicroBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, ch_exp=(2, 2), ch_per_group=4, groups_1x1=(1, 1), depthsep=True, shuffle=False, pointwise='fft', activation_cfg=None):
        super(DYMicroBlock, self).__init__()
        self.identity = stride == 1 and inp == oup
        
        y1, y2, y3 = activation_cfg.dy
        act = activation_cfg.MODULE
        act_max = activation_cfg.ACT_MAX
        act_bias = activation_cfg.LINEARSE_BIAS
        act_reduction = activation_cfg.REDUCTION * activation_cfg.ratio
        init_a = activation_cfg.INIT_A
        init_b = activation_cfg.INIT_B
        init_ab3 = activation_cfg.INIT_A_BLOCK3

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]
        self.se = SqueezeExcite(inp, se_ratio=0.25)

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = self.se(out)
            out = out + identity

        return out

class MicroNet(nn.Module):
    def __init__(self, cfg, input_size=224, num_classes=1000, teacher=False):
        super(MicroNet, self).__init__()

        mode = cfg.MODEL.MICRONETS.NET_CONFIG
        self.cfgs = get_micronet_config(mode)

        block = eval(cfg.MODEL.MICRONETS.BLOCK)
        stem_mode = cfg.MODEL.MICRONETS.STEM_MODE
        stem_ch = cfg.MODEL.MICRONETS.STEM_CH
        stem_dilation = cfg.MODEL.MICRONETS.STEM_DILATION
        stem_groups = cfg.MODEL.MICRONETS.STEM_GROUPS
        out_ch = cfg.MODEL.MICRONETS.OUT_CH
        depthsep = cfg.MODEL.MICRONETS.DEPTHSEP
        shuffle = cfg.MODEL.MICRONETS.SHUFFLE
        pointwise = cfg.MODEL.MICRONETS.POINTWISE
        dropout_rate = cfg.MODEL.MICRONETS.DROPOUT

        act_max = cfg.MODEL.ACTIVATION.ACT_MAX
        act_bias = cfg.MODEL.ACTIVATION.LINEARSE_BIAS
        activation_cfg= cfg.MODEL.ACTIVATION

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)


        self.avgpool = nn.Sequential(
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        ) 

        # building last several layers
        output_channel = out_ch
         
        self.classifier = nn.Sequential(
            SwishLinear(input_channel, output_channel),
            nn.Dropout(dropout_rate),
            SwishLinear(output_channel, num_classes)
        )
        self._initialize_weights()
           
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def micronet(name='m0'):
    cfg.merge_from_list(preset[name])
    return eval('MicroNet')(cfg)

if __name__ == '__main__':

    from torchsummaryX import summary

    model = micronet('m1')
    dummy_input = torch.rand(32,3,224,224)
    profile = summary(model, dummy_input)

