from torch.utils.data.dataset import IterableDataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import configure, cfg
from enum import Enum
from datetime import date
# from fmod.pipeline.merra2 import TensorRole

class ncFormat(Enum):
	Standard = 'standard'
	DALI = 'dali'
	SRES = "sres"

def path_suffix(vres: str="high") -> str:
	ncformat: ncFormat = ncFormat(cfg().task.nc_format)
	upscale_factor: int = cfg().model.get('scale_factor',1)
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{upscale_factor}"
	return res_suffix

def data_suffix(vres: str="high") -> str:
	ncformat: ncFormat = ncFormat(cfg().task.nc_format)
	format_suffix = ".dali" if ncformat == ncformat.DALI else ".nc"
	upscale_factor: int = cfg().model.get('scale_factor',1)
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{upscale_factor}"
	return res_suffix + format_suffix

class BaseDataset(IterableDataset):

	def __init__(self, length: int ):
		super(BaseDataset, self).__init__()
		self.length = length
		self.chanIds: Dict[str,List[str]] = {}
		self.current_date = date(1, 1, 1)

	def __getitem__(self, idx: int):
		raise NotImplementedError()

	def randomize(self):
		raise NotImplementedError()

	def channel_ids(self, role: str) -> List[str]:
		return self.chanIds[role]

	def __len__(self):
		return self.length