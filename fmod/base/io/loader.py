import xarray as xa, math
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.util.config import cfg
from enum import Enum
from datetime import date, datetime
from omegaconf import DictConfig, OmegaConf
from fmod.base.util.dates import date_list, year_range, batches_range

class TSet(Enum):
	Train = 'train'
	Validation = 'val'
	Test = 'test'
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
	downscale_factors: List[int] = cfg().model.downscale_factors
	downscale_factor = math.prod(downscale_factors)
	res_suffix = ""
	if (vres == "low") and (ncformat == ncformat.SRES):
		res_suffix = f".us{downscale_factor}"
	return res_suffix + format_suffix
