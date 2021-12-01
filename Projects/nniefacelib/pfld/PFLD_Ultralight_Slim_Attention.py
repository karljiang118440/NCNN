#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  Jiang chaoqing

# (1)、add attention module
#
######################################################

import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential

from torch.nn import Module,AvgPool2d,Linear

from pfld.base_module import Conv_Block
# from base_module import Conv_Block

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class SEModule(nn.Module):
    def __init__(self,channel,reduciton=16):
        super(SEModule,self).__init__()
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduciton,bias= False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduciton,channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c, _, _ = x.size()
        y  = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)



    


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def Conv_Block(in_channel,out_channel,kernel_size,stride,padding,group=1,has_bn=True,is_linear = False):
    return Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding=padding,groups=group,bias= False),
        BatchNorm2d(out_channel) if has_bn else Sequential(),
        nn.ReLU(inplace =True) if not is_linear else Sequential()
    )


class GhostModule(nn.Module):
    def __init__(self, in_channel, out_channel, is_linear=False):
        super(GhostModule, self).__init__()
        self.out_channel = out_channel
        init_channel = math.ceil(out_channel / 2)
        new_channel = init_channel

        self.primary_conv = Conv_Block(in_channel, init_channel, 1, 1, 0, is_linear=is_linear)
        self.cheap_operation = Conv_Block(init_channel, new_channel, 3, 1, 1, group=init_channel, is_linear=is_linear)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channel, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.ghost_conv = Sequential(
            # GhostModule
            GhostModule(in_channel, hidden_channel, is_linear=False),
            # DepthwiseConv-linear
            Conv_Block(hidden_channel, hidden_channel, 3, stride, 1, group=hidden_channel, is_linear=True) if stride == 2 else Sequential(),
            # GhostModule-linear
            GhostModule(hidden_channel, out_channel, is_linear=True)
        )

        if stride == 1 and in_channel == out_channel:
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                Conv_Block(in_channel, in_channel, 3, stride, 1, group=in_channel, is_linear=True),
                Conv_Block(in_channel, out_channel, 1, 1, 0, is_linear=True)
            )

    def forward(self, x):
        return self.ghost_conv(x) + self.shortcut(x)



class PFLDInference(nn.Module):
    def __init__(self,width_factor=0.5):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(
            3, int(64 * width_factor), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(64 * width_factor))
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            int(64 * width_factor), int(64 * width_factor), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(64 * width_factor))
        self.relu = nn.ReLU(inplace=True)


        self.conv3_1 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(64 * width_factor), stride=2)

        self.block3_2 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(64 * width_factor), stride=1)
        self.block3_3 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(64 * width_factor), stride=1)
        self.block3_4 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(64 * width_factor), stride=1)
        self.block3_5 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(64 * width_factor), stride=1)

        self.conv4_1 = GhostBottleneck(int(64 * width_factor), int(256 * width_factor), int(128 * width_factor), stride=2)

        self.conv5_1 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)
        self.block5_2 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)
        self.block5_3 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)
        self.block5_4 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)
        self.block5_5 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)
        self.block5_6 = GhostBottleneck(int(128 * width_factor), int(512 * width_factor), int(128 * width_factor), stride=1)



        # 增加 se 模块到模型中

        self.block5_SE1 = SEModule(int(128 * width_factor))
        self.block5_SE2 = SEModule(int(128 * width_factor))
        self.block5_SE3 = SEModule(int(128 * width_factor))
        self.block5_SE4 = SEModule(int(128 * width_factor))
        self.block5_SE5 = SEModule(int(128 * width_factor))
        self.block5_SE6 = SEModule(int(128 * width_factor))


        self.conv6_1 = GhostBottleneck(int(128 * width_factor), int(256 * width_factor), int(16 * width_factor), stride=1)


        # self.conv7 = Conv_Block(int(16 * 1), int(32 * 1), 3, 1, 1)
        # self.conv8 = Conv_Block(int(32 * 1), int(128 * 1), 112 // 16, 1, 0, has_bn=False)


        self.conv7 = conv_bn(int(16 * width_factor), int(32 * width_factor), 3, 2)
        self.conv8 = nn.Conv2d(int(32 * width_factor), int(128 * width_factor), 7, 1, 0)

        # self.avg_pool1 = AvgPool2d(128 // 2)
        # self.avg_pool2 = AvgPool2d(128 // 4)
        # self.avg_pool3 = AvgPool2d(128 // 8)
        # self.avg_pool4 = AvgPool2d(128 // 16)

        # self.fc = Linear(int(512 * 1), 98 * 2)

        self.bn8 = nn.BatchNorm2d(int(128 * width_factor) )



        self.avg_pool1 = nn.AvgPool2d(14* 1)
        self.avg_pool2 = nn.AvgPool2d(7* 1)
        self.fc = nn.Linear(int(176 * width_factor) , 196)
        self.fc_aux = nn.Linear(int(176 * width_factor), 3)

        self.conv1_aux = conv_bn(int(64 * width_factor) , int(128 * width_factor) , 3, 2)
        self.conv2_aux = conv_bn(int(128 * width_factor) , int(128 * width_factor), 3, 1)
        self.conv3_aux = conv_bn(int(128 * width_factor),int( 32 * width_factor), 3, 2)
        self.conv4_aux = conv_bn(int(32 * width_factor),int(128 * width_factor) , 7, 1)





        self.max_pool1_aux = nn.MaxPool2d(3)
        self.fc1_aux = nn.Linear(int(128 * width_factor), int(32* width_factor))
        self.fc2_aux = nn.Linear(int(32 * width_factor) + int(176 * width_factor), 3)



    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)

        x = self.conv5_1(x)
        x = self.block5_SE1(x)

        x = self.block5_2(x)
        x = self.block5_SE2(x)

        x = self.block5_3(x)
        x = self.block5_SE3(x)

        x = self.block5_4(x)
        x = self.block5_SE4(x)

        x = self.block5_5(x)
        x = self.block5_SE5(x)

        x = self.block5_6(x)
        x = self.block5_SE6(x)





        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)


        aux = self.conv1_aux(out1)
        aux = self.conv2_aux(aux)
        aux = self.conv3_aux(aux)
        aux = self.conv4_aux(aux)
        aux = self.max_pool1_aux(aux)
        aux = aux.view(aux.size(0), -1)
        aux = self.fc1_aux(aux)
        aux = torch.cat([aux, multi_scale], 1)
        pose = self.fc2_aux(aux)

        return pose, landmarks




if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    plfd_backbone = PFLDInference()
    angle, landmarks = plfd_backbone(input)
    print(plfd_backbone)
    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))
