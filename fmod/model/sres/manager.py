import logging, torch, math, csv
from fmod.base.util.logging import lgm, exception_handled, log_timing
import torch.nn as nn
import xarray as xa
from io import TextIOWrapper
from fmod.base.util.config import ConfigContext, cfg
import os, time, yaml, numpy as np
from fmod.base.util.config import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
import importlib, pandas as pd
from datetime import datetime
from fmod.base.io.loader import TSet, srRes
from fmod.base.util.array import xa_downsample
from fmod.data.batch import BatchDataset
from collections.abc import Iterable

def pkey( tset: TSet, ltype: str ): return '-'.join([tset.value,ltype])

def tidx() -> int:
	return int(time.time()/10)

def version_test( test: str ):
	try:
		tset = TSet(test)
		return 0
	except ValueError:
		return 1

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
		self.model_name = cfg().model.name
		self.device = device
		self._sample_input = None
		self._sample_target = None
		self.target_variables = cfg().task.target_variables
		self._dataset: BatchDataset = None
		self.cids: List[int] = self.get_channel_idxs( self.target_variables )
		self.model_config = dict( nchannels_in = len(cfg().task.input_variables), nchannels_out = len(cfg().task.target_variables), device = device )
		if cfg().model.get('use_temporal_features', False ):
			self.model_config['temporal_features'] = get_temporal_features()

	def sample_input( self ) -> xa.DataArray:
		if self._sample_input is None:
			batch: xa.DataArray = self.sample_target( )
			self._sample_input = xa_downsample( batch )
		return self._sample_input

	def sample_target( self ) -> xa.DataArray:
		if self._sample_target is None:
			self._sample_target = self.get_batch_array()
		return self._sample_target

	def get_batch_array(self) -> xa.DataArray:
		return self.get_dataset().get_current_batch_array()

	def get_dataset(self) -> BatchDataset:
		if self._dataset is None:
			self._dataset = BatchDataset(cfg().task)
		return self._dataset

	def get_channel_idxs(self, channels: List[str] ) -> List[int]:
		return self.get_dataset().get_channel_idxs(channels)

	def get_sample_target(self ) -> xa.DataArray:
		result = self.sample_target()
	#	result = ( result.isel(channel=self.cids)) if (len(self.cids) < self.sample_target().sizes['channel']) else self.sample_target
		return result

	def get_sample_input(self, targets_only=True ) -> xa.DataArray:
		result: xa.DataArray = self.sample_input()
		# if targets_only and (len(self.cids) < self.sample_input().sizes['channel']):
		#	result =  self.sample_input().isel(channel=self.cids)
		return result

	def filter_targets(self, data_array: np.ndarray ) -> np.ndarray:
		# return np.take( data_array, self.cids, axis=1 )
		return data_array

	def get_model(self) -> nn.Module:
		importpath = f"fmod.model.sres.{self.model_name}.network"
		model_package = importlib.import_module(importpath)
		return model_package.get_model( **self.model_config ).to(self.device)

def rrkey( tset: TSet, **kwargs ) -> str:
	epoch = kwargs.get('epoch', -1)
	epstr = f"-{epoch}" if epoch >= 0 else ''
	return f"{tset.value}{epstr}"

class ResultRecord(object):

	def __init__(self, tset: TSet, epoch: float, loss: float, ref_loss: float):
		self.loss: float = loss
		self.ref_loss: float = ref_loss
		self.epoch: float = epoch
		self.tset: TSet = tset

	def serialize(self) -> List[str]:
		return [ self.tset.value, f"{self.epoch:.3f}", f"{self.loss:.6f}", f"{self.ref_loss:.6f}" ]

	def __str__(self):
		return f" --- TSet: {self.tset.value}, Epoch: {self.epoch:.3f},  Loss: {self.loss:.6f},  Ref Loss: {self.ref_loss:.6f}"

class ResultFileWriter:

	def __init__(self, file_path: str):
		self.file_path = file_path
		self._csvfile: TextIOWrapper = None
		self._writer: csv.writer = None

	@property
	def csvfile(self) -> TextIOWrapper:
		if self._csvfile is None:
			self._csvfile = open(self.file_path, 'a', newline='\n')
		return self._csvfile

	def refresh(self):
		if self._csvfile is not None:
			os.rename(self.file_path, f"{self.file_path}.{tidx()}")
		self._csvfile = None

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

	def __init__(self, file_paths: List[str] ):
		self.file_paths = file_paths
		self._csvfiles: List[TextIOWrapper] = None
		self._readers: List[csv.reader] = None

	@property
	def csvfiles(self) -> List[TextIOWrapper]:
		if self._csvfiles is None:
			self._csvfiles = []
			for file_path in self.file_paths:
				try:
					self._csvfiles.append( open( file_path, 'r', newline='' ) )
					print( f"ResultFileReader reading from file: {file_path}")
				except FileNotFoundError:
					pass
		return self._csvfiles

	@property
	def csvreaders(self) -> List[csv.reader]:
		if self._readers is None:
			self._readers = []
			for csvfile in self.csvfiles:
				self._readers.append( csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL) )
		return self._readers

	def close(self):
		if self._csvfiles is not None:
			self._readers = None
			for csvfile in self._csvfiles:
				csvfile.close()
			self._csvfiles = None

class ResultsAccumulator(object):

	def __init__(self, cc: ConfigContext, **kwargs):
		self.results: List[ResultRecord] = []
		self.dataset: str = cc.dataset
		self.task = cc.task
		self.model = cc.model
		self.save_dir = kwargs.get( 'save_dir', cfg().platform.processed )
		self._writer: Optional[ResultFileWriter] = None
		self._reader: Optional[ResultFileReader] = None

	@property
	def reader(self) -> ResultFileReader:
		if self._reader is None:
			self._reader = ResultFileReader( [ self.result_file_path(model_specific=True) ] )
		return self._reader

	@property
	def writer(self) -> ResultFileWriter:
		if self._writer is None:
			self._writer = ResultFileWriter( self.result_file_path() )
		return self._writer

	def result_file_path(self, model_specific = True ) -> str:
		results_save_dir = f"{self.save_dir}/{self.task}_result_recs"
		os.makedirs(results_save_dir, exist_ok=True)
		model_id = f"_{self.model}" if model_specific else ""
		return f"{results_save_dir}/{self.dataset}_{self.task}{model_id}_losses.csv"

	def refresh_state(self):
		rfile =self.result_file_path()
		if os.path.exists( rfile ):
			os.remove( rfile )

	def close(self):
		if self._reader is not None:
			self._reader.close()
			self._reader = None
		if self._writer is not None:
			self._writer.close()
			self._writer = None
		self.results = []

	@classmethod
	def create_record( cls, rec: List[str] ) -> ResultRecord:
		ref_loss: float = float(rec[3]) if len(rec) > 3 else float('nan')
		return ResultRecord( TSet(rec[0]), float(rec[1]), float(rec[2]), ref_loss )

	def record_losses(self, tset: TSet, epoch: float, loss: float, ref_loss: float, flush=False):
		rr: ResultRecord = ResultRecord(tset, epoch, loss, ref_loss )
		self.results.append( rr )
		if flush: self.flush()

	def serialize(self)-> Dict[ str, Tuple[str,float,float] ]:
		sr =  { k: rr.serialize() for k, rr in self.results }
		return sr

	def flush(self):
		self.save()
		self.close()

	@exception_handled
	def save(self):
		print( f" ** Saving training stats to {self.result_file_path()}")
		for result in self.results:
			self.writer.write_entry( result.serialize() )

	def load_row(self, row: List[str]):
		rec = self.create_record(row)
		if rec is not None:
			self.results.append(rec)

	def load_results( self ):
		for reader in self.reader.csvreaders:
			for row in reader:
				self.load_row(row)
		print(f" ** Loading training stats ({len(self.results)} recs) from {self.result_file_path()}")

	def get_plot_data(self) -> Tuple[Dict[TSet,np.ndarray],Dict[TSet,np.ndarray]]:
		plot_data, model_data = {}, {}
		for tset in [TSet.Train, TSet.Validation]:
			result_data = model_data.setdefault(tset, {})
			print( f"get_plot_data({tset.value}): {len(self.results)} results")
			for result in self.results:
				if result.tset == tset:
					result_data[ result.epoch ] = result.loss

		x, y = {}, {}
		for tset in [TSet.Train, TSet.Validation]:
			result_data = model_data[ tset ]
			x[tset] = np.array(list(result_data.keys()))
			y[tset] = np.array(list(result_data.values()))
		return x, y

	def rprint(self):
		print( f"\n\n---------------------------- {self.task} Results --------------------------------------")
		print(f" * dataset: {self.dataset}")
		print(f" * model: {self.model}")
		for result in self.results:
			print(str(result))


