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

def nbatches( task_config, tset: TSet ) -> int:
	nbs: Dict[str,int] = task_config.get('nbatches', None)
	if nbs is not None: return nbs[tset.value]
	return 0

def batches_date_range( task_config, tset: TSet )-> List[datetime]:
	days_per_batch: int = task_config.get( 'days_per_batch', 0 )
	return date_list( start_date( task_config ), days_per_batch * nbatches( task_config, tset ) )

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
