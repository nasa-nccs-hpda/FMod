import numpy as np, xarray as xa
import torch, time, random, os
from omegaconf import DictConfig, OmegaConf
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
import nvidia.dali as dali
from fmod.base.util.logging import lgm
from fmod.base.util.model  import normalize as dsnorm
from nvidia.dali.tensors import TensorCPU, TensorListCPU
from fmod.base.util.dates import date_list, year_range, batches_date_range
from fmod.base.util.ops import format_timedeltas
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
from modulus.datapipes.datapipe import Datapipe
from fmod.base.source.merra2.model import FMBatch, BatchType, SRBatch
from modulus.datapipes.meta import DatapipeMetaData
from fmod.base.util.model  import dataset_to_stacked
from fmod.base.io.loader import BaseDataset, data_suffix
from fmod.base.util.ops import nnan
from torch import FloatTensor
from fmod.base.util.ops import ArrayOrTensor
import pandas as pd
from fmod.base.util.config import cfg
from fmod.base.util.dates import drepr



class AWSExportDataset(BaseDataset):
	
	def __init__(self, task_config: DictConfig, **kwargs ):
		super(AWSExportDataset,self).__init__(task_config,**kwargs)
		self.task_config: DictConfig = task_config

	def cache_filepath(self, d: date = None, vres: str = "high") -> str:
		version = cfg().task.dataset_version
		assert d is not None, "cache_filepath: date arg is required for dynamic variables"
		fpath = f"{fmbdir('processed')}/{version}/{drepr(d)}{data_suffix(vres)}"
		os.makedirs(os.path.dirname(fpath), mode=0o777, exist_ok=True)
		return fpath

	def load_dataset(self, d: date, vres: str = "high") -> xa.Dataset:
		filepath = self.cache_filepath(VarType.Dynamic, d, vres)
		result: xa.Dataset = access_data_subset(filepath, vres)
		lgm().log(f" * load_dataset[{vres}]({d}) {bounds(result)} nts={result.coords['time'].size} {filepath}")
		return result