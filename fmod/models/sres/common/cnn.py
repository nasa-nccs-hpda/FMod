import math

import torch
import torch.nn as nn
from ..util import *
import torch.nn.functional as F

def default_conv( in_channels: int, out_channels: int, kernel_size: Size2, bias: bool ):
    return nn.Conv2d( in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias )

class BasicBlock(nn.Sequential):
    def __init__( self,
            conv: Callable[[int,int,Size2,bool],nn.Module],
            in_channels: int,
            out_channels: int,
            kernel_size: Size2,
            stride: int = 1,
            bias: bool = False,
            bn: bool = True,
            act: nn.Module = nn.ReLU(True)
    ):
        m = [ conv(in_channels, out_channels, kernel_size, bias) ]
        if bn: m.append( nn.BatchNorm2d(out_channels) )
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


