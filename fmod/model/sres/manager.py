import logging, torch, math, csv
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
from io import TextIOWrapper
import os, time, yaml, numpy as np
from fmod.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
import importlib, pandas as pd
from datetime import datetime
from fmod.base.io.loader import TSet
from fmod.base.source.loader import srRes
from fmod.data.batch import BatchDataset
from collections.abc import Iterable

def pkey( tset: TSet, ltype: str ): return '-'.join([tset.value,ltype])

def get_temporal_features( time: np.ndarray = None ) -> Optional[np.ndarray]:
	if time is None: return None
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
		self.cids: List[int] = self.get_channel_idxs( self.target_variables, srRes.High, TSet.Train )
		self.model_config['nchannels'] = len(cfg().task.input_variables)
		if self.model_config.get('use_temporal_features', False ):
			self.model_config['temporal_features'] = get_temporal_features()
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

def rrkey( tset: TSet, **kwargs ) -> str:
	epoch = kwargs.get('epoch', -1)
	epstr = f"-{epoch}" if epoch >= 0 else ''
	return f"{tset.value}{epstr}"

class ResultRecord(object):

	def __init__(self, tset: TSet, epoch: int, model_loss: float, interp_loss: float ):
		self.model_loss: float = model_loss
		self.upsampled_loss: float = interp_loss
		self.epoch: int = epoch
		self.tset: TSet = tset

	def serialize(self) -> List[str]:
		return [ self.tset.value, str(self.epoch), f"{self.model_loss:.4f}", f"{self.upsampled_loss:.4f}" ]

	def __str__(self):
		return f" --- TSet: {self.tset.value}, Epoch: {self.epoch},  Model: {self.model_loss:.4f}, Upsample: {self.upsampled_loss:.4f}"

class ResultFileWriter:

	def __init__(self, file_path: str):
		self.file_path = file_path
		self._csvfile: TextIOWrapper = None
		self._writer: csv.writer = None

	@property
	def csvfile(self) -> TextIOWrapper:
		if self._csvfile is None:
			self._csvfile = open(self.file_path, 'w', newline='\n')
		return self._csvfile

	@property
	def csvwriter(self) -> csv.writer:
		if self._writer is None:
			self._writer = csv.writer(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		return self._writer

	def write_entry(self, entry: List[str]):
		self.csvwriter.writerow(entry)

	def close(self):
		if self._csvfile is not None:
			self._writer = None
			self._csvfile.close()
			self._csvfile = None


class ResultFileReader:

	def __init__(self, file_path: str):
		self.file_path = file_path
		self._csvfile: TextIOWrapper = None
		self._reader = None

	@property
	def csvfile(self):
		if self._csvfile is None:
			self._csvfile = open(self.file_path, 'r', newline='')
		return self._csvfile

	@property
	def csvreader(self) -> csv.reader:
		if self._reader is None:
			self._reader =  csv.reader(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		return self._reader

	def close(self):
		if self._csvfile is not None:
			self._reader = None
			self._csvfile.close()
			self._csvfile = None

class ResultsAccumulator(object):

	def __init__(self, task: str, dataset: str, scenario: str, model: str, **kwargs):
		self.results: List[ResultRecord] = []
		self.refresh_state = kwargs.get('refresh_state', False)
		self.dataset: str = dataset
		self.scenario: str = scenario
		self.task = task
		self.model = model
		self.save_dir = kwargs.get('save_dir', cfg().platform.processed)
		self._writer: Optional[ResultFileWriter] = None
		self._reader: Optional[ResultFileReader] = None
		if kwargs.get('load',False): self.load_results()

	@property
	def reader(self) -> ResultFileReader:
		if self._reader is None:
			self._reader = ResultFileReader(self.result_file_path())
		return self._reader

	@property
	def writer(self) -> ResultFileWriter:
		if self._writer is None:
			self._writer = ResultFileWriter( self.result_file_path() )
		return self._writer

	def result_file_path(self) -> str:
		results_save_dir = f"{self.save_dir}/{self.task}_result_recs"
		os.makedirs(results_save_dir, exist_ok=True)
		return f"{results_save_dir}/{self.dataset}_{self.scenario}_{self.model}_losses.csv"

	def close(self):
		if self._reader is not None:
			self._reader.close()
			self._reader = None
		if self._writer is not None:
			self._writer.close()
			self._writer = None

	def load_results(self):
		results: List[Dict[str,Any]] = self.read_model_results( )
		for result in results: self.update_loss_record( result )

	def update_loss_record(self, lrec: Dict[str,Any] ):
		rr: ResultRecord = self.results.setdefault( rrkey( lrec['tset'], **lrec),  ResultRecord(**lrec) )
		rr.update(**lrec)

	def record_losses(self, tset: TSet, epoch, model_loss: float, upsampled_loss: float, **kwargs ):
		rr: ResultRecord = ResultRecord(tset, epoch, model_loss, upsampled_loss )
		self.results.append( rr )

	def serialize(self)-> Dict[ str, Tuple[float,float,int] ]:
		sr =  { k: rr.serialize() for k, rr in self.results }
		return sr

	@exception_handled
	def save(self):
		for result in self.results:
			self.writer.write_entry( result.serialize() )

	def read_model_results( self ) -> List[Dict[str,Any]]:
		results: List[Dict[str,Any]] = []
		for row in self.reader.csvreader:
			result = dict( tset=row[0], epoch=int(row[1]), model_loss=float(row[1]), interp_loss=float(row[2]) )
			results.append( result )
		return results

	def get_plot_data(self ) -> Tuple[Dict[TSet,np.ndarray],Dict[TSet,np.ndarray]]:
		results = self.read_model_results()
		plot_data, model_data = {}, {}
		for tset in [TSet.Validation, TSet.Test]:
			for ltype in ['model_loss', 'interp_loss']:
				result_data = model_data.setdefault(pkey(tset, ltype), [])
				for result in results:
					if result['tset'].strip() == tset.value:
						result_data.append( [ result['epoch'], result[ltype] ] )

		x, y = {}, {}
		for tset in [TSet.Validation, TSet.Test]:
			pdata = model_data[ pkey(tset, 'model_loss') ]
			x[tset] = np.array([pd[0] for pd in pdata])
			y[tset] = np.array([pd[1] for pd in pdata])
		return x, y

	def rprint(self):
		print( f"\n\n---------------------------- {self.task} Results --------------------------------------")
		print(f" * dataset: {self.dataset}")
		print(f" * scenario: {self.scenario}")
		print(f" * model: {self.model}")
		for rid, result in self.results.items():
			print(f"{rid}: {result}")


