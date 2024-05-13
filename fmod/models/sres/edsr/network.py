from ..common.residual import ResBlock
from ..common.upsample import SPUpsample
from fmod.models.sres.util import *
import torch, math, torch.nn as nn
from fmod.models.sres.common.cnn import default_conv

class EDSR(nn.Module):
    def __init__( self,
		conv: Callable[[int,int,Size2,bool],nn.Module],
		scale: int,
        nchannels: int,
		nfeatures: int,
        kernel_size: Size2,
        n_resblocks: int,
		bn: bool = False,
		act: nn.Module = nn.ReLU(True),
		bias: bool = True,
        res_scale: float = 1.0
	 ):
        super(EDSR, self).__init__()

        m_head: List[nn.Module] = [ conv(nchannels, nfeatures, kernel_size, bias) ]

        m_body: List[nn.Module] = [ ResBlock( conv, nfeatures, kernel_size, bias, bn, act, res_scale ) for _ in range(n_resblocks) ]
        m_body.append( conv(nfeatures, nfeatures, kernel_size, bias) )

        m_tail: List[nn.Module] = [
            SPUpsample( conv, scale, nfeatures,False ),
            conv( nfeatures, nchannels, kernel_size, bias )
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

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')


def get_model( mconfig: Dict[str, Any] ) -> nn.Module:
    nchannels: int =            mconfig['nchannels']
    nfeatures: int =            mconfig['nfeatures']
    scale_factors: List[int] =  mconfig['upscale_factors']
    kernel_size: Size2 =        mconfig['kernel_size']
    nrlayers: int =             mconfig['nrlayers']
    res_scale: float =          mconfig['res_scale']
    conv: Callable[[int, int, Size2, bool], nn.Module] = default_conv
    scale: int = math.prod(scale_factors)
    bnorm: bool = False
    bias: bool = True
    act: nn.Module = nn.ReLU(True)
    return EDSR( conv, scale, nchannels, nfeatures, kernel_size, nrlayers, bnorm, act, bias,  res_scale)
