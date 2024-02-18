from torch.utils.data.dataset import IterableDataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import configure, cfg
from enum import Enum
# from fmod.pipeline.merra2 import TensorRole

class ncFormat(Enum):
	Standard = 'standard'
	DALI = 'dali'
	SRES = "sres"

def path_suffix(vres: str="high") -> str:
	ncformat: ncFormat = ncFormat(cfg().task.nc_format)
	upscale_factor: int = cfg().task.upscale_factor
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{upscale_factor}"
	return res_suffix

def data_suffix(vres: str="high") -> str:
	ncformat: ncFormat = ncFormat(cfg().task.nc_format)
	format_suffix = ".dali" if ncformat == ncformat.DALI else ".nc"
	upscale_factor: int = cfg().task.upscale_factor
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{upscale_factor}"
	return res_suffix + format_suffix

class BaseDataset(IterableDataset):

	def __init__(self, lenght: int ):
		super(BaseDataset, self).__init__()
		self.length = lenght
		self.chanIds: Dict[str,List[str]] = {}

	def channel_ids(self, role: str) -> List[str]:
		return self.chanIds[role]

	def __len__(self):
		return self.length