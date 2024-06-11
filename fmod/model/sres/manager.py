import logging, torch, math
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
import os, time, yaml, numpy as np
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
		self.time: np.ndarray = self.sample_input(TSet.Train).coords['time'].values
		self.cids: List[int] = self.get_channel_idxs( self.target_variables, srRes.High, TSet.Train )
		self.model_config['nchannels'] = self.sample_input(TSet.Train).sizes['channel']
		if self.model_config.get('use_temporal_features', False ):
			self.model_config['temporal_features'] = get_temporal_features(self.time)
		self.model_config['device'] = device

	def sample_input( self, tset: TSet ) -> xa.DataArray:
		return self.get_batch_array( srRes.Low, tset )

	def sample_target( self, tset: TSet ) -> xa.DataArray:
		rv = self.get_batch_array( srRes.High, tset )
		return rv

	def get_batch_array(self, sres: srRes, tset: TSet) -> xa.DataArray:
		return self.get_dataset(sres, tset).get_current_batch_array()

	def get_dataset(self, sres: srRes, tset: TSet) -> BatchDataset:
		return self.datasets.setdefault( (sres, tset), BatchDataset(cfg().task, vres=sres, tset=tset) )

	def get_channel_idxs(self, channels: List[str], sres: srRes, tset: TSet ) -> List[int]:
		return self.get_dataset(sres, tset).get_channel_idxs(channels)

	def get_sample_target(self, tset: TSet ) -> xa.DataArray:
		result = self.sample_target(tset)
	#	result = ( result.isel(channel=self.cids)) if (len(self.cids) < self.sample_target().sizes['channel']) else self.sample_target
		return result

	def get_sample_input(self, tset: TSet, targets_only: bool = True) -> xa.DataArray:
		result: xa.DataArray = self.sample_input(tset)
		# if targets_only and (len(self.cids) < self.sample_input().sizes['channel']):
		#	result =  self.sample_input().isel(channel=self.cids)
		return result

	def filter_targets(self, data_array: np.ndarray ) -> np.ndarray:
		# return np.take( data_array, self.cids, axis=1 )
		return data_array

	def get_model(self) -> nn.Module:
		importpath = f"fmod.model.sres.{self.model_name}.network"
		model_package = importlib.import_module(importpath)
		return model_package.get_model( self.model_config ).to(self.device)

class ResultRecord(object):

	def __init__(self, model_loss: float, upsampled_loss: float, **kwargs ):
		self.model_loss = model_loss
		self.upsampled_loss = upsampled_loss
		self.epoch = kwargs.get('epoch', -1)

	def key(self, model: str, tset: TSet) -> str:
		epstr = f"-{self.epoch}" if self.epoch >= 0 else ''
		return f"{model}-{tset.value}{epstr}"

	def serialize(self) -> Tuple[float,float]:
		return self.model_loss, self.upsampled_loss

	def __str__(self):
		return f" --- Model: {self.model_loss:.4f}, Upsample: {self.upsampled_loss:.4f}, Alpha: {self.model_loss/self.upsampled_loss:.3f}"
class ResultsAccumulator(object):

	def __init__(self, task: str, dataset: str, scenario: str ):
		self.results: Dict[ str, ResultRecord ] = {}
		self.dataset: str = dataset
		self.scenario: str = scenario
		self.task = task

	def record_losses(self, model: str, tset: TSet, model_loss: float, upsampled_loss: float, **kwargs ):
		rr: ResultRecord = ResultRecord(model_loss, upsampled_loss, **kwargs)
		self.results[ rr.key( model, tset ) ] = rr

	def serialize(self)-> Dict[ str, Tuple[float,float] ]:
		sr =  { k: rr.serialize() for k, rr in self.results.items() }
		return sr

	@exception_handled
	def save(self, save_dir: str):
		results_save_dir =  f"{save_dir}/{self.task}_result_recs"
		os.makedirs( results_save_dir, exist_ok=True )
		file_path: str = f"{results_save_dir}/{self.dataset}_{self.scenario}_losses.yml"
		results = self.serialize()
		print(f"Saving results to file: '{file_path}'")
		with open(file_path, "w") as fh:
			yaml.dump(results, fh)

	def print(self):
		print( f"\n\n---------------------------- {self.task} Results --------------------------------------")
		print(f" * dataset: {self.dataset}")
		print(f" * scenario: {self.scenario}")
		for rid, result in self.results.items():
			print(f"{rid}: {result}")


