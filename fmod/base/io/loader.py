import xarray, math
from torch.utils.data.dataset import IterableDataset
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import cfg
from enum import Enum
from datetime import date
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.dates import date_list, year_range, batches_range

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
	upscale_factors: List[int] = cfg().model.upscale_factors
	upscale_factor = math.prod(upscale_factors)
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{upscale_factor}"
	return res_suffix + format_suffix

class BaseDataset(IterableDataset):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(BaseDataset, self).__init__()
		self.task_config: DictConfig = task_config
		self.train_dates: List[date] = batches_range(task_config)
		self.days_per_batch: int = task_config.days_per_batch
		self.hours_per_step: int = task_config.hours_per_step
		self.steps_per_day = 24 // self.hours_per_step
		self.steps_per_batch: int = self.days_per_batch * self.steps_per_day
		self.chanIds: Dict[str,List[str]] = {}
		self.current_date: date = self.train_dates[0]

	def __getitem__(self, idx: int):
		raise NotImplementedError()

	def randomize(self):
		raise NotImplementedError()

	def channel_ids(self, role: str) -> List[str]:
		return self.chanIds[role]

	def __len__(self):
		return self.steps_per_batch

	def get_batch(self, start_date: date) -> Dict[str, xarray.DataArray]:
		raise NotImplementedError()

	def get_current_batch(self) -> Dict[str, xarray.DataArray]:
		return self.get_batch( self.current_date )