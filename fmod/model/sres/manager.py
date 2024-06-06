import logging, torch, math
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
import os, time, numpy as np
from fmod.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
import importlib, pandas as pd
from datetime import datetime
from fmod.base.io.loader import TSet
from fmod.base.source.loader import srRes
from fmod.data.batch import BatchDataset

def get_temporal_features( time: np.ndarray ) -> np.ndarray:
	sday, syear, t0, pi2 = [], [], time[0], 2 * np.pi
	for idx, t in enumerate(time):
		td: float = float((t - t0) / np.timedelta64(1, 'D'))
		sday.append((np.sin(td * pi2), np.cos(td * pi2)))
		ty: float = float((t - t0) / np.timedelta64(365, 'D'))
		syear.append([np.sin(ty * pi2), np.cos(ty * pi2)])
	# print( f"{idx}: {pd.Timestamp(t).to_pydatetime().strftime('%m/%d:%H/%Y')}: td[{td:.2f}]=[{sday[-1][0]:.2f},{sday[-1][1]:.2f}] ty[{ty:.2f}]=[{syear[-1][0]:.2f},{syear[-1][1]:.2f}]" )
	tfeats = np.concatenate([np.array(tf,dtype=np.float32) for tf in [sday, syear]], axis=1)
	return tfeats.reshape(list(tfeats.shape) + [1, 1])
class SRModels:

	def __init__(self,  device: torch.device):
		self.model_config = dict( cfg().model.items() )
		self.model_name = cfg().model.name
		self.device = device
		self.target_variables = cfg().task.target_variables
		self.datasets: Dict[Tuple[srRes,TSet],BatchDataset] = {}
		self.time: np.ndarray = self.sample_input().coords['time'].values
		self.cids: List[int] = self.get_channel_idxs( self.target_variables, srRes.High )
		self.model_config['nchannels'] = self.sample_input().sizes['channel']
		if self.model_config.get('use_temporal_features', False ):
			self.model_config['temporal_features'] = get_temporal_features(self.time)
		self.model_config['device'] = device

	def sample_input(self, tset: TSet = TSet.Train) -> xa.DataArray:
		return self.get_batch_array( srRes.Low, tset )

	def sample_target(self, tset: TSet = TSet.Train) -> xa.DataArray:
		return self.get_batch_array( srRes.High, tset )

	def get_batch_array(self, sres: srRes, tset: TSet) -> xa.DataArray:
		return self.get_dataset(sres, tset).get_current_batch_array()

	def get_dataset(self, sres: srRes, tset: TSet) -> BatchDataset:
		return self.datasets.setdefault( (sres, tset), BatchDataset(cfg().task, vres=sres, tset=tset) )

	def get_channel_idxs(self, channels: List[str], sres: srRes, tset: TSet = TSet.Train) -> List[int]:
		return self.get_dataset(sres, tset).get_channel_idxs(channels)

	def get_sample_target(self) -> xa.DataArray:
		result =  self.sample_target().isel(channel=self.cids) if (len(self.cids) < self.sample_target().sizes['channel']) else self.sample_target
		lgm().log(f" !!! Get Sample target !!! cids={self.cids}: sample_target{self.sample_target().dims}{self.sample_target().shape}, result{result.shape}")
		return result

	def get_sample_input(self, targets_only: bool = True) -> xa.DataArray:
		result = self.sample_input
		if targets_only and (len(self.cids) < self.sample_input().sizes['channel']):
			result =  self.sample_input().isel(channel=self.cids)
		lgm().log(f" !!! Get Sample input !!! cids={self.cids}: sample_input{self.sample_input().dims}{self.sample_input().shape}, result{result.shape}")
		return result

	def filter_targets(self, data_array: np.ndarray ) -> np.ndarray:
		return np.take( data_array, self.cids, axis=1 )

	def get_model(self) -> nn.Module:
		importpath = f"fmod.model.sres.{self.model_name}.network"
		model_package = importlib.import_module(importpath)
		return model_package.get_model( self.model_config ).to(self.device)





	#