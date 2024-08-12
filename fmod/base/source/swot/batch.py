import xarray as xa, math, os
from fmod.base.util.config import cfg
from omegaconf import DictConfig
from .raw import SWOTRawDataLoader
from typing import Any, Mapping, Sequence, Tuple, Union, List, Dict, Literal, Optional
from fmod.base.source.loader.batch import SRDataLoader


def filepath() -> str:
	return f"{cfg().dataset.dataset_root}/{cfg().dataset.dataset_files}"

CoordIdx = Union[ Dict[str,int], Tuple[int,int] ]

class SWOTDataLoader(SRDataLoader):

	def __init__(self, task_config: DictConfig, **kwargs):
		SRDataLoader.__init__(self, task_config)
		self.loader = SWOTRawDataLoader(task_config,  **kwargs)
		self.shape = None

	def load_timeslice(self, time_index: int, **kwargs) -> xa.DataArray:
		return self.loader.load_timeslice(time_index, **kwargs)

	def load_tile_batch(self, tile_range: Tuple[int,int] ) -> Optional[xa.DataArray]:
		tile_batch: xa.DataArray = self.loader.load_batch( tile_range )
		# print( f" ~~~~~~~~~~~~~~~~ load_tile_batch: attrs={list(tile_batch.attrs.keys())}")
		return tile_batch

	def get_batch_time_indices(self):
		return self.loader.get_batch_time_indices()






