import numpy as np, xarray as xa
import torch, dataclasses
from nvidia.dali.tensors import TensorCPU, TensorListCPU
from fmod.base.util.dates import date_list, year_range
from fmod.base.util.config import cfg2meta, cfg
from fmod.base.util.ops import format_timedeltas, fmbdir
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
from fmod.base.util.model import dataset_to_stacked
from fmod.base.gpu import set_device, get_device

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
    levels: np.ndarray = dset.coords[coords['z']].values
    for vname in vnames:
        dvar: xa.DataArray = dset.data_vars[vname]
        if coords['z'] in dvar.dims:    channels.extend([f"{vname}{int(levels[iL])}" for iL in range(dvar.sizes[coords['z']])])
        else:                           channels.append(vname)
        for (cname, coord) in dvar.coords.items():
            if cname not in (merge_dims + list(sizes.keys())):
                sizes[ cname ] = coord.size
    darray: xa.DataArray = dataset_to_stacked( dset, sizes=sizes, preserved_dims=tuple(sizes.keys()) )
    darray.attrs['channels'] = channels
    return darray.transpose( "batch", "channels", coords['y'], coords['x'] )

def array2tensor( darray: xa.DataArray ) -> Tensor:
    array_data: np.ndarray = np.ravel(darray.values).reshape( darray.shape )
    return torch.tensor( array_data, device=get_device(), requires_grad=True, dtype=torch.float32 )