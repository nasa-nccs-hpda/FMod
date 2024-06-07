import numpy as np, xarray as xa
import torch, time, random, traceback, math
from omegaconf import DictConfig
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from fmod.base.source.batch import SRBatch
from fmod.base.util.logging import lgm
from fmod.base.util.model  import normalize as dsnorm
from fmod.base.util.ops import format_timedeltas
from typing import List, Tuple, Union, Dict, Any, Sequence
from modulus.datapipes.meta import DatapipeMetaData
from fmod.base.util.model import dataset_to_stacked
from fmod.base.io.loader import TSet, batches_range, nbatches
from fmod.base.source.loader import srRes
from fmod.base.util.config import cfg
from random import randint
import pandas as pd

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.
ArrayOrDataset = Union[xa.DataArray,xa.Dataset]

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

def norm( batch_data: xa.DataArray, batch=False):
    dims = [batch_data.dims[-1],batch_data.dims[-2]]
    if batch: dims.append('time')
    mean = batch_data.mean(dim=dims)
    std = batch_data.std(dim=dims)
    return (batch_data-mean)/std

def rshuffle(a: Dict[Tuple[int,int],Any] ) -> Dict[Tuple[int,int],Any]:
    a1: List[ Tuple[ Tuple[int,int],Any ] ] = list(a.items())
    random.shuffle(a1)
    return dict( a1 )

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "MERRA2NC"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True

class TileGrid(object):

    def __init__(self, tset: TSet = TSet.Train):
        self.tset: TSet = tset
        origins: Dict[str,Dict[str,int]] = cfg().task.get('origin',{})
        # print( f"TileGrid: origins={list(origins.keys())}, tset='{self.tset.value}'")
        self.origin: Dict[str,int] = origins[self.tset.value]
        self.tile_grid: Dict[str, int] = cfg().task.tile_grid[self.tset.value]
        self.tile_size: Dict[str,int] = cfg().task.tile_size
        self.tlocs: Dict[Tuple[int,int],Dict[str,int]] = {}
        downscale_factors: List[int] = cfg().model.downscale_factors
        self.downscale_factor = math.prod(downscale_factors)

    def get_tile_size(self, downscaled: bool = False ) -> Dict[str, int]:
        sf = self.downscale_factor if downscaled else 1
        return { d: self.tile_size[d] * sf for d in ['x', 'y'] }

    def get_tile_origin( self, ix: int, iy: int, downscaled: bool = False ) -> Dict[str, int]:
        sf = self.downscale_factor if downscaled else 1
        return { d: self.origin[d] + self.cdim(ix, iy, d) * self.tile_size[d] * sf for d in ['x', 'y'] }

    def get_tile_locations(self, randomize=False, downscaled: bool = False, selected_tile: Tuple[int,int] = None ) -> Dict[ Tuple[int,int], Dict[str,int] ]:
        if len(self.tlocs) == 0:
            for ix in range(self.tile_grid['x']):
                for iy in range(self.tile_grid['y']):
                    if (selected_tile is None) or ((ix,iy) == selected_tile):
                        self.tlocs[(ix,iy)] = self.get_tile_origin(ix,iy,downscaled)
        if randomize: rshuffle(self.tlocs)
        return self.tlocs

    @classmethod
    def cdim(cls, ix: int, iy: int, dim: str) -> int:
        if dim == 'x': return ix
        if dim == 'y': return iy


class BatchDataset(object):

    def __init__(self, task_config: DictConfig, vres: srRes, tset: TSet, **kwargs):
        self.vres: srRes = vres
        self.tset: TSet = tset
        self.srtype = 'input' if self.vres == srRes.High else 'target'
        self.task_config: DictConfig = task_config
        self.tile_grid = TileGrid(tset)
        self.load_inputs: bool = kwargs.pop('load_inputs', (vres=="low"))
        self.load_targets: bool = kwargs.pop('load_targets', (vres=="high"))
        self.load_base: bool = kwargs.pop('load_base', False)
        self.day_index: int = 0
        self.train_dates: List[datetime] = batches_range(task_config,tset)
        self.days_per_batch: int = task_config.days_per_batch
        self.hours_per_step: int = task_config.hours_per_step
        self.steps_per_day = 24 // self.hours_per_step
        self.steps_per_batch: int = self.days_per_batch * self.steps_per_day
        self.downscale_factors: List[int] = cfg().model.downscale_factors
        self.scalefactor = math.prod(self.downscale_factors)
        self.current_date: datetime = self.train_dates[0]
        self.current_origin: Dict[str, int] = self.tile_grid.origin
        self.train_steps: int = task_config.get('train_steps',1)
        self.nsteps_input: int = task_config.get('nsteps_input', 1)
        self.tile_size: Dict[str, int] = self.scale_coords(task_config.tile_size)

        self.srbatch: SRBatch = SRBatch( task_config, self.tile_size, vres, tset, **kwargs )
        self.norms: Dict[str, xa.Dataset] = self.srbatch.norm_data
        self.mu: xa.Dataset  = self.norms.get('mean_by_level')
        self.sd: xa.Dataset  = self.norms.get('stddev_by_level')
        self.dsd: xa.Dataset = self.norms.get('diffs_stddev_by_level')
        self.ntsteps = self.srbatch.batch_steps * self.ntbatches
        self.hours_per_step = task_config.hours_per_step
        self.hours_per_batch = self.days_per_batch * 24
        self.tcoords: List[datetime] = self.get_time_coords()
        self.current_batch_data = None

    def data_index_range(self) -> Tuple[int,int]:
        dindxs = self.srbatch.data_loader.dindxs
        return dindxs[0], dindxs[-1]

    @property
    def ntbatches(self):
        return  nbatches( self.task_config, self.tset )

    def __len__(self):
        return self.steps_per_batch

    def get_batch_array(self, origin: Dict[str,int], batch_date: datetime ) -> xa.DataArray:
        batch_data: xa.DataArray = self.srbatch.load( origin, batch_date)
        self.current_origin = origin
        self.current_date = batch_date
        self.current_batch_data = norm(batch_data)
        self.current_batch_data.attrs['didx-range'] = self.data_index_range()
        print( f" >> current_batch_data({origin}:{batch_date.strftime('%m/%d:%H/%Y')}) -> {self.current_batch_data.dims} {list(self.current_batch_data.shape)}")
        return self.current_batch_data

    def get_current_batch_array(self) -> xa.DataArray:
        return self.get_batch_array(self.current_origin, self.current_date)

    def in_batch(self, time_coord: datetime, batch_date: datetime) -> bool:
        if time_coord < batch_date: return False
        dt: timedelta = time_coord - batch_date
        hours: int = dt.seconds // 3600
        return hours < self.hours_per_batch

    def tile_index(self, origin: Dict[str,int] ):
        sgx = self.task_config.tile_grid['x']
        return origin['y']*sgx + origin['x']

    def load_global_timeslice(self, date_index: int = 0, **kwargs ) -> xa.DataArray:
        vid: str = kwargs.get( 'vid', self.task_config.target_variables[0] )
        global_timeslice: np.ndarray =  self.srbatch.load_global_timeslice( vid, self.train_dates[date_index] )
        return xa.DataArray( global_timeslice, dims=['y','x'] )

    @property
    def batch_dates(self)-> List[datetime]:
        return self.get_batch_dates()

    def scale_coords(self, c: Dict[str, int]) -> Dict[str, int]:
        if self.vres == srRes.Low:  return c
        else:                       return {k: v * self.scalefactor for k, v in c.items()}

    def get_channel_idxs(self, channels: List[str] ) -> List[int]:
        cidxs = None # [ self.srbatch.channels.index(cid) for cid in channels ]
        # lgm().log(f"get_channel_idxs: srtype={self.srtype}, channels={channels}, cidxs={cidxs}")
        return cidxs

    def normalize(self, vdata: xa.Dataset) -> xa.Dataset:
        return dsnorm( vdata, self.sd, self.mu )

    def get_batch_dates(self, randomize: bool = True, offset: bool = True, target_date: datetime = None ) -> List[datetime]:
        start_dates, ndates = [], len( self.train_dates )
        offset: int = randint(0, self.days_per_batch-1) if offset else 0
        for dindex in range( 0, ndates, self.days_per_batch):
            batch_date = self.train_dates[ dindex ] +  timedelta( days=offset )
            if (target_date is None) or self.in_batch( target_date, batch_date ):
                start_dates.append( batch_date )
        if randomize:   random.shuffle(start_dates)
        return start_dates

    def get_time_coord(self, tindex: int ) -> datetime:
        return self.tcoords[tindex]
    def get_time_coords(self) -> List[datetime]:
        bdates: List[datetime] = self.get_batch_dates(randomize=False, offset=False )
        time: List[datetime] = []
        for bdate in bdates:
            for bstep in range(self.srbatch.batch_steps):
                hours = bstep * self.hours_per_step
                time.append( bdate + timedelta(hours=hours) )
        return time

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
            lgm().log(f" >> >> {len(dataset.data_vars.keys())} model variables: {input_varlist}")
            lgm().log(f" >> >> dataset vars = {list(dataset.data_vars.keys())}")
            lgm().log(f" >> >> {len(selected_inputs.data_vars.keys())} selected inputs: {list(selected_inputs.data_vars.keys())}")
            input_array: xa.DataArray = self.batch2array( self.normalize(selected_inputs) )
            channels = input_array.attrs.get('channels', [])
            lgm().log(f" load_inputs-> merged training array{input_array.dims}{input_array.shape}" )
            # print(f" >> merged training array: {input_array.dims}: {input_array.shape}, coords={list(input_array.coords.keys())}, channels={list(channels)}")
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
        # print( f" @@@STACKED ARRAY: {darray.dims}{darray.shape}, coords={list(darray.coords.keys())}, channels={channels}", flush=True)
        darray.attrs['channels'] = channels
        result = darray.transpose( "time", "channels", darray.dims[1], darray.dims[2] )
        return result

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