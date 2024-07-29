import xarray as xa, math
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import cfg
from enum import Enum
from datetime import date, datetime
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.config import start_date
from fmod.base.util.dates import date_list

class TSet(Enum):
	Train = 'train'
	Validation = 'valid'
	Test = 'test'
	Upsample = 'upsample'

class ncFormat(Enum):
	Standard = 'standard'
	DALI = 'dali'
	SRES = "sres"

class srRes(Enum):
	Low = 'lr'
	High = 'hr'
	Raw = 'raw'

	@classmethod
	def from_config(cls, sval: str ) -> 'srRes':
		if sval == "low": return cls.Low
		if sval == "high": return cls.High
		if sval == "raw": return cls.Raw

class batchDomain(Enum):
	Time = 'time'
	Tiles = 'tiles'

	@classmethod
	def from_config(cls, sval: str) -> 'batchDomain':
		if sval == "time": return cls.Time
		if sval == "tiles": return cls.Tiles

def nbatches( task_config ) -> int:
	nbs: Dict[str,int] = task_config.get('nbatches', None)
	if nbs is not None: return nbs[tset.value]
	return 0

def batches_date_range( task_config )-> List[datetime]:
	days_per_batch: int = task_config.get( 'days_per_batch', 0 )
	return date_list( start_date( task_config ), days_per_batch * nbatches( task_config ) )

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
	downscale_factors: List[int] = cfg().model.downscale_factors
	downscale_factor = math.prod(downscale_factors)
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{downscale_factor}"
	return res_suffix + format_suffix

class BaseDataset(object):

	def __init__(self, task_config: DictConfig, **kwargs ):
		super(BaseDataset, self).__init__()
		self.task_config: DictConfig = task_config
		self.train_dates: List[datetime] = batches_date_range(task_config)
		self.days_per_batch: int = task_config.days_per_batch
		self.hours_per_step: int = task_config.hours_per_step
		self.steps_per_day = 24 // self.hours_per_step
		self.steps_per_batch: int = self.days_per_batch * self.steps_per_day
		self.downscale_factors: List[int] = cfg().model.downscale_factors
		self.scalefactor = math.prod(self.downscale_factors)
		self.current_date: date = self.train_dates[0]
		self.current_origin: Dict[str, int] = task_config.origin

	def get_tile_locations(self) -> List[Dict[str,int]]:
		raise NotImplementedError()

	def randomize(self):
		raise NotImplementedError()

	def __len__(self):
		return self.steps_per_batch

	def get_batch(self, origin: Dict[str,int], batch_date: date ) -> Dict[str, xa.DataArray]:
		raise NotImplementedError()

	def get_batch_array(self, origin: Dict[str,int], batch_date: date ) -> xa.DataArray:
		raise NotImplementedError()

	def get_current_batch(self) -> Dict[str, xa.DataArray]:
		return self.get_batch(self.current_origin, self.current_date)

	def get_current_batch_array(self) -> xa.DataArray:
		raise NotImplementedError()

