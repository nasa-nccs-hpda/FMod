from fmod.base.io.loader import BaseDataset
from omegaconf import DictConfig

class Datasets(object):

	def __init__(self, dsname: str, task_config: DictConfig ):
		self.dsname = dsname
		self.task_config: DictConfig = task_config

	def get_dataset(self, vres, load_inputs, load_base, load_targets) -> BaseDataset:
		if self.dsname == "merra2":
			from fmod.pipeline.ncbatch import ncBatchDataset
			input_dataset = ncBatchDataset( self.task_config, vres=vres, load_inputs=load_inputs, load_base=load_base, load_targets=load_targets )
			return input_dataset
		else:
			raise Exception( f"Unsupported dataset: {self.dsname}")

