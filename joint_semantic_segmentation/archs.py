import torch
import torch.nn as nn
from torch.utils import checkpoint as cp
from collections import OrderedDict

import torch.nn.functional as F
import numpy as np

__all__ = ['FRRNet','UNet','JointNetwork']

#######################################################################################
"""
@author: johnny, https://github.com/jcheunglin/Full-Resolution-Residual-Networks-with-PyTorch
@time: 2018/12/4 22:21
"""
# conv 3*3
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# efficient implementation of residual block
def residual_func(relu, norm1, conv1, norm2, conv2):
    def rbc_function(*inputs):
        outp_ = relu(norm1(conv1(*inputs)))
        outp = relu(torch.add(norm2(conv2(outp_)),*inputs))
        return outp
    return rbc_function


# efficient implementation of FRRU
def frr_func(relu, norm1, conv1, norm2, conv2):
    def rbc_func(*inputs):
        cat = torch.cat(inputs,1)
        outp_ = relu(norm1(conv1(cat)))
        outp = relu(norm2(conv2(outp_)))
        return outp
    return rbc_func


# ref torch vision model
# Residual base block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, efficient=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.efficent=efficient

    def forward(self, x):

        if self.efficent:
            resfunc = residual_func(self.relu,self.bn1,self.conv1,self.bn2,self.conv2)
            ret = cp.checkpoint(resfunc,x)
            return ret

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class FRRU(nn.Module):
    """
     full-resolution residual unit (FRRU)
    """
    def __init__(self,y_in_c, y_out_c, factor=2, z_c=32,efficient=True):
        super(FRRU,self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3,stride=factor,padding=1)  # for z
        self.conv1 = conv3x3(y_in_c+z_c,y_out_c)
        self.bn1 = nn.BatchNorm2d(y_out_c)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(y_out_c,y_out_c)
        self.bn2 = nn.BatchNorm2d(y_out_c)
        self.convz = nn.Conv2d(in_channels=y_out_c,out_channels=z_c,kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=factor)
        self.efficient=efficient

    def forward(self, y,z):
        z_ = self.pool(z)

        if self.efficient:
            frr = frr_func(self.relu,self.bn1,self.conv1,self.bn2,self.conv2)
            cat = cp.checkpoint(frr,y,z_)
            z_out = z + self.up(self.convz(cat))
            return cat,z_out

        cat = torch.cat((y,z_),1)
        cat = self.relu(self.bn1(self.conv1(cat)))
        y = self.relu(self.bn2(self.conv2(cat)))
        z_out =z + self.up(self.convz(y))
        return y,z_out


class FRRLayer(nn.Module):
    def __init__(self,in_channels,out_channels,factor,num_blocks,z_c=32):
        super(FRRLayer,self).__init__()
        self.frr1 = FRRU(in_channels,out_channels,z_c=z_c,factor=factor)
        self.nexts = nn.ModuleList([FRRU(out_channels,out_channels,factor=factor,efficient=False) for _ in range(1,num_blocks)])

    def forward(self, y,z):
        y,z = self.frr1(y,z)
        for m in self.nexts:
            y,z = m(y,z)
        return y,z


class FRRNet(nn.Module):
    """
    implementation table A of Full-Resolution Residual Networks
    """
    # Comment by Eli:
    # in_channels=1 -> Greyscale Images
    # out_channels=5 -> 5 Classes, Background + 4 Proteins
    def __init__(self,in_channels=1,out_channels=5,layer_blocks=(3,4,2,2)):
        super(FRRNet, self).__init__()

        # 5×5
        self.first = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels,out_channels=48,kernel_size=5,padding=2)),
                ('bn',nn.BatchNorm2d(48)),
                ('relu',nn.ReLU()),
                ]))
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()

        # 3×48 Residual Unit
        self.reslayers_in = nn.Sequential(*[BasicBlock(48,48,efficient=False) for _ in range(3)])

        # divide
        self.divide = nn.Conv2d(in_channels=48,out_channels=32,kernel_size=1)

        # frrlayer 1
        self.frrnlayer1 = FRRLayer(48,96,factor=2,num_blocks=layer_blocks[0])

        # frrlayer2
        self.frrnlayer2 = FRRLayer(96,192,factor=4,num_blocks=layer_blocks[1])

        # frrnlayer3
        self.frrnlayer3 = FRRLayer(192,384,factor=8,num_blocks=layer_blocks[2])

        # frrnlayer4
        self.frrnlayer4 = FRRLayer(384,384,factor=16,num_blocks=layer_blocks[3])

        # defrrnlayer1
        self.defrrnlayer1 = FRRLayer(384,192,factor=8,num_blocks=2)

        # defrrnlayer2
        self.defrrnlayer2 = FRRLayer(192,192,factor=4,num_blocks=2)

        # defrrnlayer3
        self.defrrnlayer3 = FRRLayer(192,96,factor=2,num_blocks=2)

        # join
        self.compress = nn.Conv2d(96+32,48,kernel_size=1)

        # 3×48 reslayer

        self.reslayers_out = nn.Sequential(*[BasicBlock(48,48,efficient=True) for _ in range(3)])

        self.out_conv = nn.Conv2d(48,out_channels,1)


    def forward(self, x):

        x = self.first(x)
        y = self.reslayers_in(x)

        z = self.divide(y)
        y = self.pool(y)

        y,z = self.frrnlayer1(y,z)

        y = self.pool(y)
        y,z = self.frrnlayer2(y,z)

        y = self.pool(y)
        y,z= self.frrnlayer3(y,z)

        y = self.pool(y)
        y,z =self.frrnlayer4(y,z)

        y = self.up(y)
        y,z = self.defrrnlayer1(y,z)

        y = self.up(y)
        y,z = self.defrrnlayer2(y,z)

        y = self.up(y)
        y,z = self.defrrnlayer3(y,z)


        y = self.up(y)
        refine = self.compress(torch.cat((y,z),1))

        out = self.reslayers_out(refine)
        out = self.out_conv(out)
        return out
#######################################################################################
#######################################################################################

# Taken from Eli's previous work

class VGGBlock(nn.Module):
    '''
        Consists of 2 Convolution Layers with ReLU and batch normalization between them. 
        params: Setting the filter size through the number of channels in input, middle and the final output.
    '''
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    '''
    Unet Architecture (Reference - https://arxiv.org/pdf/1505.04597.pdf)
    params: number of classes and input channels.
    '''	
    def __init__(self, num_classes=1, input_channels=1, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

#######################################################################################

class JointNetwork(nn.Module):
    def __init__(self, input_channels=1, num_classes=5, **kwargs):
        super().__init__()

        self.denoise = UNet(num_classes=input_channels, input_channels=input_channels)
        self.semantic_seg = FRRNet(in_channels=input_channels,out_channels=num_classes)
        self.final = nn.Softmax()

    def forward(self, input):
        denoised_imgs = self.denoise(input)
        raw_seg = self.semantic_seg(denoised_imgs)
        class_seg_prob = self.final(raw_seg)

        return denoised_imgs, class_seg_prob