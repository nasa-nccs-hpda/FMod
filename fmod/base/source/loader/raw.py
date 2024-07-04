import math, os
from enum import Enum
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal
import time, numpy as np

import xarray as xa

from fmod.base.util.logging import lgm, log_timing
from fmod.base.util.config import cfg
from omegaconf import DictConfig, OmegaConf

class SRRawDataLoader(object):

	@classmethod
	def get_loader(cls, task_config: DictConfig, **kwargs) -> 'SRRawDataLoader':
		dset: str = task_config.dataset
		if dset.startswith("swot"):
			from fmod.base.source.swot.raw import SWOTRawDataLoader
			return SWOTRawDataLoader( task_config, **kwargs )

	def load_file(self, **kwargs) -> xa.DataArray:
		raise NotImplementedError("SRRawDataLoader:load")