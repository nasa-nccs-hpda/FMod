from ..common.residual import ResBlock
from ..common.upsample import SPUpsample
from fmod.model.sres.util import *
import torch, math, torch.nn as nn
from fmod.model.sres.common.common import FModule

def get_model( **config ) -> nn.Module:
	return EDSR(**config)
class EDSR(FModule):

    def __init__(self, **kwargs):
        super(EDSR, self).__init__({}, **kwargs)

        m_head: List[nn.Module] = [self.conv(self.nchannels_in, self.nfeatures, self.kernel_size, self.bias)]
        m_body: List[nn.Module] = [ ResBlock(self.conv, self.nfeatures, self.kernel_size, self.bias, self.batch_norm, self.act, self.res_scale) for _ in range(self.nlayers) ]
        m_body.append(self.conv(self.nfeatures, self.nfeatures, self.kernel_size, self.bias))

        m_tail: List[nn.Module] = [
            SPUpsample(self.conv, self.scale, self.nfeatures, False),
            self.conv(self.nfeatures, self.nchannels_out, self.kernel_size, self.bias)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x




