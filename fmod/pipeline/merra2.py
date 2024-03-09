import numpy as np, xarray as xa
import torch, dataclasses
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
import nvidia.dali as dali
from fmod.base.util.logging import lgm
from fmod.base.util.model import normalize as dsnorm
from nvidia.dali.tensors import TensorCPU, TensorListCPU
from fmod.base.util.dates import date_list, year_range
from fmod.base.util.config import cfg2meta, cfg
from fmod.base.util.ops import format_timedeltas, fmbdir
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
from modulus.datapipes.datapipe import Datapipe
from fmod.base.source.merra2.model import FMBatch, BatchType
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
    PREDICTION = "prediction"

class TensorType:
    DALI = "dali"
    TORCH = "torch"

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

def ds2array( dset: xa.Dataset, **kwargs ) -> xa.DataArray:
    coords = cfg().task.coords
    merge_dims = kwargs.get( 'merge_dims', [coords['z'], coords['t']] )
    sizes: Dict[str,int] = {}
    vnames = list(dset.data_vars.keys()); vnames.sort()
    channels = []
    for vname in vnames:
        dvar: xa.DataArray = dset.data_vars[vname]
        if coords['z'] in dvar.dims:    channels.extend([f"{vname}~{iL}" for iL in range(dvar.sizes[coords['z']])])
        else:                           channels.append(vname)
        for (cname, coord) in dvar.coords.items():
            if cname not in (merge_dims + list(sizes.keys())):
                sizes[ cname ] = coord.size
    darray: xa.DataArray = dataset_to_stacked( dset, sizes=sizes, preserved_dims=tuple(sizes.keys()) )
    darray.attrs['channels'] = channels
    return darray.transpose( "channels", coords['y'], coords['x'] )

def get_device():
    devname = cfg().task.device
    if devname == "gpu": devname = "cuda"
    device = torch.device(devname)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device

def array2tensor( darray: xa.DataArray ) -> Union[TensorCPU,FloatTensor]:
    tt = cfg().task.tensor_type.lower()
    array_data: np.ndarray = np.ravel(darray.values).reshape( darray.shape )
    if   tt == TensorType.DALI:   return TensorCPU( array_data )
    elif tt == TensorType.TORCH:  return FloatTensor( array_data, device=get_device() )
    else: raise Exception( f"Unsupported tensor type: {tt}")


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MERRA2NC"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True

class MERRA2Dataset(BaseDataset):
    def __init__(self, **kwargs):
        self.train_dates = kwargs.pop( 'train_dates', year_range(*cfg().task.year_range, randomize=True) )
        self.dts = cfg().task.data_timestep
        self.n_day_offsets = 24//self.dts
        super(MERRA2Dataset,self).__init__(len(self.train_dates) * self.n_day_offsets)

        self.train_steps = cfg().task.train_steps
        self.nsteps_input = cfg().task.nsteps_input
        self.input_duration = pd.Timedelta( self.dts*self.nsteps_input, unit="h" )
        self.target_lead_times = [f"{iS * self.dts}h" for iS in self.train_steps]
        self.fmbatch: FMBatch = FMBatch(BatchType.Training, **kwargs)
        self.norms: Dict[str, xa.Dataset] = self.fmbatch.norm_data
        self.current_date = date(1,1,1 )
        self.mu: xa.Dataset  = self.norms['mean_by_level']
        self.sd: xa.Dataset  = self.norms['stddev_by_level']
        self.dsd: xa.Dataset = self.norms['diffs_stddev_by_level']

    def normalize(self, vdata: xa.Dataset) -> xa.Dataset:
        return dsnorm( vdata, self.sd, self.mu )

    def get_date(self):
        return self.train_dates[ self.i // self.n_day_offsets ]

    def get_day_offset(self):
        return self.i % self.n_day_offsets

    def __next__(self) -> Tuple[ArrayOrTensor,ArrayOrTensor]:

        if self.i < self.length:
            next_date = self.get_date()
            if self.current_date != next_date:
                self.fmbatch.load( next_date )
                self.current_date = next_date
            lgm().log(f" *** MERRA2Dataset.load_date[{self.i}]: {self.current_date}, offset={self.get_day_offset()}, device={cfg().task.device}")
            train_data: xa.Dataset = self.fmbatch.get_train_data( self.get_day_offset() )
            lgm().log(f" *** >>> train_data: sizes={train_data.sizes}")
            inputs_targets: Tuple[ArrayOrTensor,ArrayOrTensor] = self.extract_inputs_targets(train_data, **cfg().task )
            self.i = self.i + 1
            return inputs_targets
        else:
            raise StopIteration

    def __iter__(self):
        self.i = 0
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
                                      levels: Tuple[int, ...], **kwargs) -> Tuple[ArrayOrTensor,ArrayOrTensor]:
        idataset = idataset.sel(level=list(levels))
        nptime: List[np.datetime64] = idataset.coords['time'].values.tolist()
        dvars = {}
        for vname, varray in idataset.data_vars.items():
            missing_batch = ("time" in varray.dims) and ("batch" not in varray.dims)
            dvars[vname] = varray.expand_dims("batch") if missing_batch else varray
        dataset = xa.Dataset(dvars, coords=idataset.coords, attrs=idataset.attrs)
        inputs, targets = self.extract_input_target_times(dataset)
        for vname, varray in inputs.data_vars.items():
            if 'time' in varray.dims:
                lgm().debug(f" ** INPUT {vname}: tcoord={varray.coords['time'].values.tolist()}")
        for vname, varray in targets.data_vars.items():
            if 'time' in varray.dims:
                lgm().debug(f" ** TARGET {vname}: tcoord={varray.coords['time'].values.tolist()}")
        lgm().debug(f"Inputs & Targets: input times: {get_timedeltas(inputs)}, target times: {get_timedeltas(targets)}, base time: {pd.Timestamp(nptime[0])} (nt={len(nptime)})")

        if set(forcing_variables) & set(target_variables):
            raise ValueError(f"Forcing variables {forcing_variables} should not overlap with target variables {target_variables}.")
        input_varlist: List[str] = list(input_variables)+list(forcing_variables)
        selected_inputs: xa.Dataset = inputs[input_varlist]

        lgm().debug(f" >> >> {len(inputs.data_vars.keys())} model variables: {input_varlist}")
        lgm().debug(f" >> >> dataset vars = {list(inputs.data_vars.keys())}")
        lgm().debug(f" >> >> {len(selected_inputs.data_vars.keys())} selected inputs: {list(selected_inputs.data_vars.keys())}")
        input_array: xa.DataArray = ds2array( self.normalize(selected_inputs) )
        lgm().debug(f" >> merged training array: {input_array.dims}: {input_array.shape}, nnan={nnan(input_array.values)}")

        lgm().debug(f" >> >> target variables: {target_variables}")
        target_array: xa.DataArray = ds2array( self.normalize(targets[list(target_variables)]) )
        lgm().debug(f" >> targets{target_array.dims}: {target_array.shape}, nnan={nnan(target_array.values)}")
        lgm().debug(f"Extract inputs: basetime= {pd.Timestamp(nptime[0])}, device={cfg().task.device}")
        self.chanIds['input']  = input_array.attrs['channels']
        self.chanIds['target'] = target_array.attrs['channels']

        if cfg().task.device == "gpu":
            return array2tensor(input_array), array2tensor(target_array)
        else:
            return input_array, target_array


class MERRA2NCDatapipe(Datapipe):
    """MERRA2 DALI data pipeline for NetCDF files"""


    def __init__(self,meta,**kwargs):
        super().__init__(meta=meta)
        self.batch_size: int = kwargs.get('batch_size', 1)
        self.paralle: bool = kwargs.get('parallel', False)
        self.batch: bool = kwargs.get('batch', False)
        self.num_workers: int = cfg().task.num_workers
        self.device: torch.device = self.get_device()
        self.pipe: dali.Pipeline = self._create_pipeline()
        self.chanIds: List[str] = None

    def build(self):
        return self.pipe.build()

    def run(self):
        return self.pipe.run()

    @classmethod
    def get_device(cls) -> torch.device:
        device = torch.device( cfg().task.device )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        return  device

    def _create_pipeline(self) -> dali.Pipeline:
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method = "spawn",
        )

        with pipe:
            source = MERRA2Dataset()
            self.length = source.length
            invar, outvar = dali.fn.external_source( source, num_outputs=2, parallel=self.parallel, batch=self.batch )
            if self.device.type == "cuda":
                invar = invar.gpu()
                outvar = outvar.gpu()
            pipe.set_outputs(invar, outvar)
        return pipe

    def __iter__(self):
        self.pipe.reset()
        return dali_pth.DALIGenericIterator([self.pipe], [ "invar", "outvar", "forcings"])

    def __len__(self):
        return self.length
