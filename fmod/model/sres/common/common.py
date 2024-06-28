import torch, math
import torch.nn as nn
from .cnn import default_conv
from fmod.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping
from fmod.base.util.logging import lgm, exception_handled, log_timing
from omegaconf import DictConfig, OmegaConf

common_parms = dict(
	nchannels_in=1,
	nchannels_out=1,
	nfeatures=64,
	kernel_size=3,
	nlayers=16,
	downscale_factors=[2, 2],
	bias=True,
	batch_norm=False,
	res_scale=1.0,
	ups_mode='bicubic'
)

class FModule(nn.Module):

	def __init__( self, mparms: Dict[str,Any], **custom_parms ):
		super(FModule, self).__init__()
		self.config: DictConfig = cfg().model
		self.conv = default_conv
		self.act: nn.Module = nn.ReLU(True)
		self.wn = lambda x: torch.nn.utils.weight_norm(x)
		self.parms = { pname: self.config.get( pname, dval ) for pname, dval in common_parms.items() }
		self.parms['scale'] = math.prod( self.parms['downscale_factors'] )
		for pdict in [ mparms, custom_parms]:
			for pname, dval in pdict.items():
				self.parms[pname] = self.config.get( pname, dval )

	def __setattr__(self, key: str, value: Any) -> None:
		self.parms[key] = value

	def __getattr__(self, key: str) -> Any:
		return self.parms.get(key)

	def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
		own_state = self.state_dict()
		for name, param in state_dict.items():
			if name in own_state:
				if isinstance(param, nn.Parameter):
					param = param.data
				try:
					own_state[name].copy_(param)
				except Exception:
					if name.find('tail') >= 0:
						print('Replace pre-trained upsampler to new one...')
					else:
						raise RuntimeError(f'While copying the parameter named {name}, whose dimensions in the model'
						                   f' are {own_state[name].size()} and whose dimensions in the checkpoint are {param.size()}.')
			elif strict:
				if name.find('tail') == -1:
					raise KeyError(f'unexpected key "{name}" in state_dict')

		if strict:
			missing = set(own_state.keys()) - set(state_dict.keys())
			if len(missing) > 0:
				raise KeyError(f'missing keys in state_dict: "{missing}"')


