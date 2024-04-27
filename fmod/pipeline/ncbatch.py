import numpy as np, xarray as xa
import torch, time, random
from omegaconf import DictConfig, OmegaConf
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
import nvidia.dali as dali
from fmod.base.util.logging import lgm
from fmod.base.util.model import normalize as dsnorm
from nvidia.dali.tensors import TensorCPU, TensorListCPU
from fmod.base.util.dates import date_list, year_range, batches_range
from fmod.base.util.ops import format_timedeltas, fmbdir
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
from modulus.datapipes.datapipe import Datapipe
from fmod.base.source.merra2.model import FMBatch, BatchType, SRBatch
from modulus.datapipes.meta import DatapipeMetaData
from fmod.base.util.model import dataset_to_stacked
from fmod.base.io.loader import BaseDataset
from fmod.base.util.ops import nnan
from torch import FloatTensor
from fmod.base.util.ops import ArrayOrTensor
import pandas as pd

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

class TensorRole:
    INPUT = "input"
    TARGET = "target"
    BASE = "base"


TargetLeadTimes = Union[
    TimedeltaLike,
    Sequence[TimedeltaLike],
    slice  # with TimedeltaLike as its start and stop.
]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR

DAY_PROGRESS = "day_progress"
YEAR_PROGRESS = "year_progress"

def get_timedeltas( dset: xa.Dataset ):
    return format_timedeltas( dset.coords["time"] )
Tensor = torch.Tensor

def d2xa( dvals: Dict[str,float] ) -> xa.Dataset:
    return xa.Dataset( {vn: xa.DataArray( np.array(dval) ) for vn, dval in dvals.items()} )

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MERRA2NC"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True

class ncBatchDataset(BaseDataset):
    def __init__(self, task_config: DictConfig, **kwargs):
        self.task_config: DictConfig = task_config
        self.train_dates: List[date] = batches_range(task_config)
        self.batch_ndays: int = task_config.batch_ndays               # number of days per batch
        self.load_inputs: bool = kwargs.pop('load_inputs',True)
        self.load_targets: bool = kwargs.pop('load_targets', True)
        self.load_base: bool = kwargs.pop('load_base', True)
        self.dts: int =task_config.data_timestep                       # data timestep in hours
        self.n_day_offsets: int = 24//self.dts                         # number of timesteps per day
        self.batch_size: int = self.batch_ndays * self.dts             # number of timesteps per batch
        self.day_index: int = 0
        super(ncBatchDataset,self).__init__(len(self.train_dates) * self.n_day_offsets)
        self.train_steps: int = task_config.train_steps
        self.nsteps_input: int = task_config.nsteps_input
        self.fmbatch: SRBatch = SRBatch( **kwargs )
        self.norms: Dict[str, xa.Dataset] = self.fmbatch.norm_data
        self.mu: xa.Dataset  = self.norms['mean_by_level']
        self.sd: xa.Dataset  = self.norms['stddev_by_level']
        self.dsd: xa.Dataset = self.norms['diffs_stddev_by_level']
        self.batch_dates: List[date] = self.get_batch_start_dates()

    def randomize(self) -> List[date]:
        random.shuffle(self.batch_dates)
        return self.batch_dates

    def __getitem__(self, idx: int):
        self.day_index = idx
        return self.__next__()

    def normalize(self, vdata: xa.Dataset) -> xa.Dataset:
        return dsnorm( vdata, self.sd, self.mu )

    def get_date(self):
        return self.batch_dates[ self.day_index ]

    def get_batch_start_dates(self, randomize: bool = True ) -> List[date]:
        start_dates, ndates = [], len( self.train_dates )
        for dindex in range( 0, ndates, self.batch_ndays ):
            start_dates.append( self.train_dates[ dindex ] )
        if randomize: random.shuffle(start_dates)
        return start_dates

    def get_batch(self, batch_date: date ) -> Dict[str,xa.DataArray]:
        t0 = time.time()
        self.fmbatch.load( batch_date )
        batch_data: Dict[str, xa.DataArray] = self.extract_batch_inputs_targets(self.fmbatch.current_batch, **self.task_config)
        self.log(batch_data, t0)
        return batch_data

    def __next__(self) -> Dict[str,xa.DataArray]:
        if self.day_index >= len( self.batch_dates ):
            raise StopIteration()
        t0 = time.time()
        next_date = self.get_date()
        if self.current_date != next_date:
            self.fmbatch.load( next_date )
            self.current_date = next_date
        batch_inputs: Dict[str,xa.DataArray] = self.extract_batch_inputs_targets( self.fmbatch.current_batch, **self.task_config )
        self.log( batch_inputs, t0 )
        self.day_index = self.day_index + 1
        return batch_inputs

    def log(self, batch_inputs: Dict[str,xa.DataArray], start_time: float ):
        lgm().log(f" *** MERRA2Dataset.load_date[{self.day_index}]: {self.current_date}, device={self.task_config.device}, load time={time.time()-start_time:.2f} sec")
        for k,v in batch_inputs.items():
            lgm().log(f" --->> {k}{v.dims}: {v.shape}")

    def __iter__(self):
        self.day_index = 0
        return self

    def extract_input_target_times(self, dataset: xa.Dataset) -> Tuple[xa.Dataset, xa.Dataset]:
        """Extracts inputs and targets for prediction, from a Dataset with a time dim.

        The input period is assumed to be contiguous (specified by a duration), but
        the targets can be a list of arbitrary lead times.

        Returns:
          inputs:
          targets:
            Two datasets with the same shape as the input dataset except that a
            selection has been made from the time axis, and the origin of the
            time coordinate will be shifted to refer to lead times relative to the
            final input timestep. So for inputs the times will end at lead time 0,
            for targets the time coordinates will refer to the lead times requested.
        """

        (target_lead_times, target_duration) = self._process_target_lead_times_and_get_duration()

        # Shift the coordinates for the time axis so that a timedelta of zero
        # corresponds to the forecast reference time. That is, the final timestep
        # that's available as input to the forecast, with all following timesteps
        # forming the target period which needs to be predicted.
        # This means the time coordinates are now forecast lead times.
        time: xa.DataArray = dataset.coords["time"]
        dataset = dataset.assign_coords(time=time + target_duration - time[-1])
        lgm().debug(f"extract_input_target_times: initial input-times={dataset.coords['time'].values.tolist()}")
        targets: xa.Dataset = dataset.sel({"time": target_lead_times})
        zero_index = -1-self.train_steps[-1]
        input_bounds = [ zero_index-(self.nsteps_input-1), (zero_index+1 if zero_index<-2 else None) ]
        inputs: xa.Dataset = dataset.isel( {"time": slice(*input_bounds) } )
        lgm().debug(f" --> Input bounds={input_bounds}, input-sizes={inputs.sizes}, final input-times={inputs.coords['time'].values.tolist()}")
        return inputs, targets

    def _process_target_lead_times_and_get_duration( self ) -> TimedeltaLike:
        if not isinstance(self.target_lead_times, (list, tuple, set)):
            self.target_lead_times = [self.target_lead_times]
        target_lead_times = [pd.Timedelta(x) for x in self.target_lead_times]
        target_lead_times.sort()
        target_duration = target_lead_times[-1]
        return target_lead_times, target_duration

    def extract_inputs_targets(self,  idataset: xa.Dataset, *, input_variables: Tuple[str, ...], target_variables: Tuple[str, ...], forcing_variables: Tuple[str, ...],
                                      levels: Tuple[int, ...], **kwargs) -> Dict[str,xa.DataArray]:
        idataset = idataset.sel(level=list(levels))
        nptime: List[np.datetime64] = idataset.coords['time'].values.tolist()
        dvars = {}
        for vname, varray in idataset.data_vars.items():
            missing_batch = ("time" in varray.dims) and ("batch" not in varray.dims)
            dvars[vname] = varray.expand_dims("batch") if missing_batch else varray
        dataset = xa.Dataset(dvars, coords=idataset.coords, attrs=idataset.attrs)
        inputs, targets = self.extract_input_target_times(dataset)
        lgm().debug(f"Inputs & Targets: input times: {get_timedeltas(inputs)}, target times: {get_timedeltas(targets)}, base time: {pd.Timestamp(nptime[0])} (nt={len(nptime)})")

        if set(forcing_variables) & set(target_variables):
            raise ValueError(f"Forcing variables {forcing_variables} should not overlap with target variables {target_variables}.")
        results = {}

        if self.load_inputs:
            input_varlist: List[str] = list(input_variables)+list(forcing_variables)
            selected_inputs: xa.Dataset = inputs[input_varlist]
            lgm().debug(f" >> >> {len(inputs.data_vars.keys())} model variables: {input_varlist}")
            lgm().debug(f" >> >> dataset vars = {list(inputs.data_vars.keys())}")
            lgm().debug(f" >> >> {len(selected_inputs.data_vars.keys())} selected inputs: {list(selected_inputs.data_vars.keys())}")
            input_array: xa.DataArray = self.ds2array( self.normalize(selected_inputs) )
            channels = input_array.attrs.get('channels', [])
            lgm().debug(f" >> merged training array: {input_array.dims}: {input_array.shape}, coords={list(input_array.coords.keys())}" )
        #    print(f" >> merged training array: {input_array.dims}: {input_array.shape}, coords={list(input_array.coords.keys())}, #channel-values={len(channels)}")
            self.chanIds['input'] = channels
            results['input'] = input_array

        if self.load_base:
            base_inputs: xa.Dataset = inputs[list(target_variables)]
            base_input_array: xa.DataArray = self.ds2array( self.normalize(base_inputs.isel(time=-1)) )
            lgm().debug(f" >> merged base_input array: {base_input_array.dims}: {base_input_array.shape}, channels={base_input_array.attrs['channels']}")
            results['base'] = base_input_array

        if self.load_targets:
            lgm().debug(f" >> >> target variables: {target_variables}")
            target_array: xa.DataArray = self.ds2array( self.normalize(targets[list(target_variables)]) )
            lgm().debug(f" >> targets{target_array.dims}: {target_array.shape}, channels={target_array.attrs['channels']}")
            lgm().debug(f"Extract inputs: basetime= {pd.Timestamp(nptime[0])}, device={self.task_config.device}")
            self.chanIds['target'] = target_array.attrs['channels']
            results['target'] = target_array

        return results

    def extract_batch_inputs_targets(self,  idataset: xa.Dataset, *, input_variables: Tuple[str, ...], target_variables: Tuple[str, ...], forcing_variables: Tuple[str, ...], **kwargs) -> Dict[str,xa.DataArray]:
        nptime: List[np.datetime64] = idataset.coords['time'].values.tolist()
        dvars = {}
        for vname, varray in idataset.data_vars.items():
            missing_batch = ("time" in varray.dims) and ("batch" not in varray.dims)
            dvars[vname] = varray.expand_dims("batch") if missing_batch else varray
        dataset = xa.Dataset(dvars, coords=idataset.coords, attrs=idataset.attrs)

        if set(forcing_variables) & set(target_variables):
            raise ValueError(f"Forcing variables {forcing_variables} should not overlap with target variables {target_variables}.")
        results = {}

        if self.load_inputs:
            input_varlist: List[str] = list(input_variables)+list(forcing_variables)
            selected_inputs: xa.Dataset = dataset[input_varlist]
            lgm().debug(f" >> >> {len(dataset.data_vars.keys())} model variables: {input_varlist}")
            lgm().debug(f" >> >> dataset vars = {list(dataset.data_vars.keys())}")
            lgm().debug(f" >> >> {len(selected_inputs.data_vars.keys())} selected inputs: {list(selected_inputs.data_vars.keys())}")
            input_array: xa.DataArray = self.batch2array( self.normalize(selected_inputs) )
            channels = input_array.attrs.get('channels', [])
            lgm().debug(f" >> merged training array: {input_array.dims}: {input_array.shape}, coords={list(input_array.coords.keys())}" )
        #    print(f" >> merged training array: {input_array.dims}: {input_array.shape}, coords={list(input_array.coords.keys())}, #channel-values={len(channels)}")
            self.chanIds['input'] = channels
            results['input'] = input_array

        if self.load_base:
            base_inputs: xa.Dataset = dataset[list(target_variables)]
            base_input_array: xa.DataArray = self.batch2array( self.normalize(base_inputs.isel(time=-1)) )
            lgm().debug(f" >> merged base_input array: {base_input_array.dims}: {base_input_array.shape}, channels={base_input_array.attrs['channels']}")
            results['base'] = base_input_array

        if self.load_targets:
            lgm().debug(f" >> >> target variables: {target_variables}")
            target_array: xa.DataArray = self.batch2array( self.normalize(dataset[list(target_variables)]) )
            lgm().debug(f" >> targets{target_array.dims}: {target_array.shape}, channels={target_array.attrs['channels']}")
            lgm().debug(f"Extract inputs: basetime= {pd.Timestamp(nptime[0])}, device={self.task_config.device}")
            self.chanIds['target'] = target_array.attrs['channels']
            results['target'] = target_array

        return results
    def ds2array(self, dset: xa.Dataset, **kwargs) -> xa.DataArray:
        coords = self.task_config.coords
        merge_dims = kwargs.get('merge_dims', [coords['z'], coords['t']])
        sizes: Dict[str, int] = {}
        vnames = list(dset.data_vars.keys())
        vnames.sort()
        channels = []
        for vname in vnames:
            dvar: xa.DataArray = dset.data_vars[vname]
            levels: List[float] = dvar.coords[coords['z']].values.tolist() if coords['z'] in dvar.dims else []
            if levels:    channels.extend([f"{vname}{int(lval)}" for lval in levels])
            else:         channels.append(vname)
            for (cname, coord) in dvar.coords.items():
                if cname not in (merge_dims + list(sizes.keys())):
                    sizes[cname] = coord.size
        darray: xa.DataArray = dataset_to_stacked(dset, sizes=sizes, preserved_dims=tuple(sizes.keys()))
        darray.attrs['channels'] = channels
    #    print( f"ds2array{darray.dims}: shape = {darray.shape}" )
        return darray.transpose( "channels", coords['y'], coords['x'])

    def batch2array(self, dset: xa.Dataset, **kwargs) -> xa.DataArray:
        coords = self.task_config.coords
        merge_dims = kwargs.get('merge_dims', [coords['z']])
        sizes: Dict[str, int] = {}
        vnames = list(dset.data_vars.keys())
        vnames.sort()
        channels = []
        for vname in vnames:
            dvar: xa.DataArray = dset.data_vars[vname]
            levels: List[float] = dvar.coords[coords['z']].values.tolist() if coords['z'] in dvar.dims else []
            if levels:    channels.extend([f"{vname}{int(lval)}" for lval in levels])
            else:         channels.append(vname)
            for (cname, coord) in dvar.coords.items():
                if cname not in (merge_dims + list(sizes.keys())):
                    sizes[cname] = coord.size
        darray: xa.DataArray = dataset_to_stacked(dset, sizes=sizes, preserved_dims=tuple(sizes.keys()))
        darray.attrs['channels'] = channels
    #    print( f"ds2array{darray.dims}: shape = {darray.shape}" )
        return darray.transpose( "time", "channels", coords['y'], coords['x'] ).rename( time="batch" )

    def get_device(self):
        devname = self.task_config.device
        if devname == "gpu": devname = "cuda"
        device = torch.device(devname)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        return device

    def array2tensor(self, darray: xa.DataArray) -> Tensor:
        array_data: np.ndarray = np.ravel(darray.values).reshape(darray.shape)
        return torch.tensor(array_data, device=self.get_device(), requires_grad=True)