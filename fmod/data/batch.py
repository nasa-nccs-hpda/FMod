import numpy as np, xarray as xa
import torch, time, random, math
from omegaconf import DictConfig
from dataclasses import dataclass
from datetime import datetime, timedelta
from fmod.base.util.dates import TimeType
from fmod.data.tiles import TileGrid
from fmod.base.util.logging import lgm
from fmod.base.util.model  import normalize as dsnorm
from fmod.base.util.ops import format_timedeltas
from typing import List, Tuple, Union, Dict, Any, Sequence, Optional
from modulus.datapipes.meta import DatapipeMetaData
from fmod.base.util.model import dataset_to_stacked
from fmod.base.io.loader import TSet, srRes, batches_date_range, nbatches, batchDomain
from fmod.base.source.batch import SRBatch
from fmod.base.util.config import cfg
import pandas as pd

TimedeltaLike = Any  # Something convertible to pd.Timedelta.
TimedeltaStr = str  # A string convertible to pd.Timedelta.
ArrayOrDataset = Union[xa.DataArray,xa.Dataset]
ArrayOrTensor = Union[xa.DataArray,torch.Tensor]

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

class BatchDataset(object):

    def __init__(self, task_config: DictConfig, **kwargs):
        self.srtype = 'target'
        self.task_config: DictConfig = task_config
        self.tile_grid = TileGrid()
        self.load_targets: bool = kwargs.pop('load_targets', True)
        self.load_base: bool = kwargs.pop('load_base', False)
        self.train_dates: List[datetime] = batches_date_range(task_config)
        self.days_per_batch: int = task_config.get('days_per_batch',0)
        self.hours_per_step: int = task_config.get('hours_per_step',0)
        self.batch_size: int = task_config.batch_size
        self.steps_per_day = (24 // self.hours_per_step) if (self.hours_per_step > 0) else 0
        self.steps_per_batch: int = self.days_per_batch * self.steps_per_day
        self.downscale_factors: List[int] = cfg().model.downscale_factors
        self.scalefactor = math.prod(self.downscale_factors)
        self.current_origin: Dict[str, int] = self.tile_grid.origin
        self.train_steps: int = task_config.get('train_steps',1)
        self.nsteps_input: int = task_config.get('nsteps_input', 1)
        self.tile_size: Dict[str, int] = self.scale_coords(task_config.tile_size)
        self.batch_domain: batchDomain = batchDomain.from_config(cfg().task.get('batch_domain', 'tiles'))

        self.srbatch: SRBatch = SRBatch( task_config, self.tile_size, **kwargs )
        self._norms: Dict[str, xa.Dataset] = None
        self._mu: xa.Dataset  = None
        self._sd: xa.Dataset  = None
        self._dsd: xa.Dataset = None
        self.ntsteps = self.srbatch.batch_steps * self.ntbatches
        self.hours_per_step = task_config.get('hours_per_step',0)
        self.hours_per_batch = self.days_per_batch * 24
        self.current_batch_data = None

    @property
    def norms(self)-> Dict[str, xa.Dataset]:
        if self._norms is None:
            self._norms = self.srbatch.norm_data
        return self._norms

    @property
    def mu(self)-> xa.Dataset:
        if self._mu is None:
            self._mu = self.norms.get('mean_by_level')
        return self._mu

    def sd(self)-> xa.Dataset:
        if self._sd is None:
            self._sd = self.norms.get('diffs_stddev_by_level')
        return self._sd

    def dsd(self)-> xa.Dataset:
        if self._dsd is None:
            self._dsd = self.norms.get('diffs_stddev_by_level')
        return self._dsd

    @property
    def tcoord(self):
        return range(*self.data_index_range())

    def data_index_range(self) -> Tuple[int,int]:
        dindxs = self.srbatch.data_loader.dindxs
        return dindxs[0], dindxs[-1]

    @property
    def ntbatches(self):
        return  nbatches( self.task_config )

    def __len__(self):
        return self.steps_per_batch

    def get_channel_idxs(self, channels):
        return [0]

    def get_batch_array(self, ctile: Dict[str,int], ctime: Union[datetime,int], **kwargs ) -> Optional[xa.DataArray]:
        if self.batch_domain == batchDomain.Time:
            rescale = kwargs.get( 'rescale', True )
            ctile = self.scale_coords(ctile) if rescale else ctile
        batch_data: xa.DataArray = self.srbatch.load( ctile, ctime)
        return batch_data

    def get_current_batch_array(self) -> xa.DataArray:
        return self.srbatch.current_batch

    def in_batch(self, time_coord: datetime, batch_date: datetime) -> bool:
        if time_coord < batch_date: return False
        dt: timedelta = time_coord - batch_date
        hours: int = (dt.seconds // 3600) + (dt.days * 24)
        return hours < self.hours_per_batch

    def tile_index(self, origin: Dict[str,int] ):
        sgx = self.task_config.tile_grid['x']
        return origin['y']*sgx + origin['x']

    def load_global_timeslice(self, **kwargs ) -> xa.DataArray:
        vid: str = kwargs.get( 'vid', self.task_config.target_variables[0] )
        global_timeslice: np.ndarray =  self.srbatch.load_global_timeslice( vid, **kwargs )
        return xa.DataArray( global_timeslice, dims=['y','x'] )

    def scale_coords(self, c: Dict[str, int]) -> Dict[str, int]:
        return {k: v * self.scalefactor for k, v in c.items()}

    def in_batch_idx(self, target_coord: int, dindex: int) -> bool:
        di = (target_coord - dindex)
        return (di>=0) and (di<self.batch_size)

    def get_batch_time_coords(self, target_coord: TimeType = -1) -> List[TimeType]:
        start_coords = []
        if self.batch_domain == batchDomain.Time:
            if self.days_per_batch > 0:
                ndates = len( self.train_dates )
                for dindex in range( 0, ndates, self.days_per_batch):
                    batch_date = self.train_dates[ dindex ]
                    if (target_coord is None) or self.in_batch( target_coord, batch_date ):
                        start_coords.append( batch_date )
            else:
                nidx = self.srbatch.get_dset_size()
                for dindex in range(0, nidx, self.batch_size):
                    if (target_coord < 0) or self.in_batch_idx(target_coord,dindex):
                        start_coords.append( dindex )
                lgm().log( f"  ------------- dataset size = {nidx}, target_coord={target_coord}, batch_size={self.batch_size}, start_coords={start_coords}  ------------- ")
        elif self.batch_domain == batchDomain.Tiles:
            start_coords = self.srbatch.get_batch_time_indices()
        random.shuffle(start_coords)
        return start_coords

    def log(self, batch_inputs: Dict[str,xa.DataArray], start_time: float ):
        lgm().log(f" *** MERRA2Dataset.load_date: device={self.task_config.device}, load time={time.time()-start_time:.2f} sec")
        for k,v in batch_inputs.items():
            lgm().log(f" --->> {k}{v.dims}: {v.shape}")

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