import numpy as np, xarray as xa
import torch, dataclasses
import nvidia.dali.plugin.pytorch as dali_pth
from dataclasses import dataclass
from datetime import date, timedelta
import nvidia.dali as dali
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
from torch.utils.data.dataset import IterableDataset
from fmod.base.source.merra2 import batch
from torch import FloatTensor
import pandas as pd

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.

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
    for vname in vnames:
        dvar: xa.DataArray = dset.data_vars[vname]
        for (cname, coord) in dvar.coords.items():
            if cname not in (merge_dims + list(sizes.keys())):
                sizes[ cname ] = coord.size
    darray: xa.DataArray = dataset_to_stacked( dset, sizes=sizes, preserved_dims=tuple(sizes.keys()) )
    return darray.transpose( "channels", coords['y'], coords['x'] )

def array2tensor( darray: xa.DataArray ) -> Union[TensorCPU,FloatTensor]:
    tt = cfg().task.tensor_type.lower()
    array_data: np.ndarray = np.ravel(darray.values).reshape( darray.shape )
    if   tt == TensorType.DALI:   return TensorCPU( array_data )
    elif tt == TensorType.TORCH:  return FloatTensor( array_data )
    else: raise Exception( f"Unsupported tensor type: {tt}")


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MERRA2NC"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True

class MERRA2InputIterator(IterableDataset):
    def __init__(self, **kwargs):
        self.train_steps = cfg().task.train_steps
        self.input_steps = cfg().task.input_steps
        self.dts = cfg().task.data_timestep
        self.n_day_offsets = 24//self.dts
        self.input_duration = f"{self.dts*self.input_steps}h"
        self.target_lead_times = [f"{iS * self.dts}h" for iS in range(1, self.train_steps + 1)]
        self.train_dates = kwargs.get( 'train_dates', year_range(*cfg().task.year_range, randomize=True) )
        self.nepochs = cfg().task.nepoch
        self.max_iter = cfg().task.max_iter
        self.fmbatch: FMBatch = FMBatch(BatchType.Training)
        self.norms: Dict[str, xa.Dataset] = self.fmbatch.norm_data
        self.current_date = date(1,1,1 )
        self.mu: xa.Dataset  = self.norms['mean_by_level']
        self.sd: xa.Dataset  = self.norms['stddev_by_level']
        self.dsd: xa.Dataset = self.norms['diffs_stddev_by_level']
        self.length = len(self.train_dates) * self.n_day_offsets

    def normalize(self, vdata: xa.Dataset) -> xa.Dataset:
        return dsnorm( vdata, self.sd, self.mu )

    def  __len__(self):
        return self.length

    def get_date(self):
        return self.train_dates[ self.i // self.n_day_offsets ]

    def get_day_offset(self):
        return self.i % self.n_day_offsets

    def __next__(self):
        if self.i < self.length:
            next_date = self.get_date()
            if self.current_date != next_date:
                self.fmbatch.load( next_date )
                self.current_date = next_date
            train_data: xa.Dataset = self.fmbatch.get_train_data( self.get_day_offset() )
            task_config = dict( target_lead_times=self.target_lead_times, input_duration=self.input_duration, **cfg().task )
            (inputs, targets) = self.extract_inputs_targets(train_data, **task_config )
            self.i = self.i + 1
            if (cfg().task.device == "gpu") and torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            return inputs, targets
        else:
            raise StopIteration

    def __iter__(self):
        self.i = 0
        return self

    def extract_input_target_times(self, dataset: xa.Dataset, input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes) -> Tuple[xa.Dataset, xa.Dataset]:
        """Extracts inputs and targets for prediction, from a Dataset with a time dim.

        The input period is assumed to be contiguous (specified by a duration), but
        the targets can be a list of arbitrary lead times.

        Examples:

          # Use 18 hours of data as inputs, and two specific lead times as targets:
          # 3 days and 5 days after the final input.
          extract_inputs_targets(
              dataset,
              input_duration='18h',
              target_lead_times=('3d', '5d')
          )

          # Use 1 day of data as input, and all lead times between 6 hours and
          # 24 hours inclusive as targets. Demonstrates a friendlier supported string
          # syntax.
          extract_inputs_targets(
              dataset,
              input_duration='1 day',
              target_lead_times=slice('6 hours', '24 hours')
          )

          # Just use a single target lead time of 3 days:
          extract_inputs_targets(
              dataset,
              input_duration='24h',
              target_lead_times='3d'
          )

        Args:
          dataset: An xa.Dataset with a 'time' dimension whose coordinates are
            timedeltas. It's assumed that the time coordinates have a fixed offset /
            time resolution, and that the input_duration and target_lead_times are
            multiples of this.
          input_duration: pandas.Timedelta or something convertible to it (e.g. a
            shorthand string like '6h' or '5d12h').
          target_lead_times: Either a single lead time, a slice with start and stop
            (inclusive) lead times, or a sequence of lead times. Lead times should be
            Timedeltas (or something convertible to). They are given relative to the
            final input timestep, and should be positive.

        Returns:
          inputs:
          targets:
            Two datasets with the same shape as the input dataset except that a
            selection has been made from the time axis, and the origin of the
            time coordinate will be shifted to refer to lead times relative to the
            final input timestep. So for inputs the times will end at lead time 0,
            for targets the time coordinates will refer to the lead times requested.
        """

        (target_lead_times, target_duration) = self._process_target_lead_times_and_get_duration(target_lead_times)

        # Shift the coordinates for the time axis so that a timedelta of zero
        # corresponds to the forecast reference time. That is, the final timestep
        # that's available as input to the forecast, with all following timesteps
        # forming the target period which needs to be predicted.
        # This means the time coordinates are now forecast lead times.
        time: xa.DataArray = dataset.coords["time"]
        dataset = dataset.assign_coords(time=time + target_duration - time[-1])
        targets: xa.Dataset = dataset.sel({"time": target_lead_times})
        input_duration = pd.Timedelta(input_duration)
        # Both endpoints are inclusive with label-based slicing, so we offset by a
        # small epsilon to make one of the endpoints non-inclusive:
        zero = pd.Timedelta(0)
        epsilon = pd.Timedelta(1, "ns")
        inputs: xa.Dataset = dataset.sel({"time": slice(-input_duration + epsilon, zero)})
        return inputs, targets

    @classmethod
    def _process_target_lead_times_and_get_duration( cls, target_lead_times: TargetLeadTimes) -> TimedeltaLike:
        """Returns the minimum duration for the target lead times."""
        # print( f"process_target_lead_times: {target_lead_times}")
        if isinstance(target_lead_times, slice):
            # A slice of lead times. xa already accepts timedelta-like values for
            # the begin/end/step of the slice.
            if target_lead_times.start is None:
                # If the start isn't specified, we assume it starts at the next timestep
                # after lead time 0 (lead time 0 is the final input timestep):
                target_lead_times = slice(pd.Timedelta(1, "ns"), target_lead_times.stop, target_lead_times.step)
            target_duration = pd.Timedelta(target_lead_times.stop)
        else:
            if not isinstance(target_lead_times, (list, tuple, set)):
                # A single lead time, which we wrap as a length-1 array to ensure there
                # still remains a time dimension (here of length 1) for consistency.
                target_lead_times = [target_lead_times]

            # A list of multiple (not necessarily contiguous) lead times:
            target_lead_times = [pd.Timedelta(x) for x in target_lead_times]
            target_lead_times.sort()
            target_duration = target_lead_times[-1]
        return target_lead_times, target_duration

    def extract_inputs_targets(self, idataset: xa.Dataset, *, input_variables: Tuple[str, ...], target_variables: Tuple[str, ...], forcing_variables: Tuple[str, ...],
        levels: Tuple[int, ...], input_duration: TimedeltaLike, target_lead_times: TargetLeadTimes, **kwargs) -> Tuple[TensorCPU, TensorCPU]:
        idataset = idataset.sel(level=list(levels))
        nptime: List[np.datetime64] = idataset.coords['time'].values.tolist()
        dvars, verbose = {}, False
        for vname, varray in idataset.data_vars.items():
            missing_batch = ("time" in varray.dims) and ("batch" not in varray.dims)
            dvars[vname] = varray.expand_dims("batch") if missing_batch else varray
        dataset = xa.Dataset(dvars, coords=idataset.coords, attrs=idataset.attrs)
        dataset = dataset.drop_vars("datetime")
        inputs, targets = self.extract_input_target_times(dataset, input_duration=input_duration, target_lead_times=target_lead_times)
        print(f"Inputs & Targets: input times: {get_timedeltas(inputs)}, target times: {get_timedeltas(targets)}, base time: {pd.Timestamp(nptime[0])} (nt={len(nptime)})")

        if set(forcing_variables) & set(target_variables):
            raise ValueError(f"Forcing variables {forcing_variables} should not overlap with target variables {target_variables}.")
        input_varlist: List[str] = list(input_variables)+list(forcing_variables)

        if verbose: print(f" >> >> input variables: {input_varlist}")
        input_array: xa.DataArray = ds2array( self.normalize(inputs[input_varlist]) )
        if verbose: print(f" >> inputs{input_array.dims}: {input_array.shape}")

        if verbose: print(f" >> >> target variables: {target_variables}")
        target_array: xa.DataArray = ds2array( self.normalize(targets[list(target_variables)]) )
        if verbose: print(f" >> targets{target_array.dims}: {target_array.shape}")

        return array2tensor(input_array), array2tensor(target_array)


class MERRA2NCDatapipe(Datapipe):
    """MERRA2 DALI data pipeline for NetCDF files"""


    def __init__(self,meta,**kwargs):
        super().__init__(meta=meta)
        self.batch_size = kwargs.get('batch_size', 1)
        self.parallel = kwargs.get('parallel', False)
        self.batch = kwargs.get('batch', False)
        self.num_workers: int = cfg().task.num_workers
        self.device = self.get_device()
        self.pipe = self._create_pipeline()

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
            source = MERRA2InputIterator()
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
