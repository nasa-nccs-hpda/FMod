import numpy as np, xarray as xa, torch
from typing import Iterable, List, Tuple, Union, Optional, Dict, Any, Sequence
import os, pandas as pd

def l2loss( prd: torch.Tensor, tar: torch.Tensor, squared=False) -> torch.Tensor:
    loss = ((prd - tar) ** 2).mean()
    if not squared: loss = torch.sqrt(loss)
    return loss
class StatsEntry:

    def __init__(self, varname: str ):
        self._stats: Dict[str,List[xa.DataArray]] = {}
        self._varname = varname

    def merge(self, entry: "StatsEntry"):
        for statname, mvars in entry._stats.items():
            for mvar in mvars:
                self.add( statname, mvar )

    def add(self, statname: str, mvar: xa.DataArray, weight: int = None ):
        if weight is not None: mvar.attrs['stat_weight'] = float(weight)
        elist = self._stats.setdefault(statname,[])
        elist.append( mvar )
#        print( f" SSS: Add stats entry[{self._varname}.{statname}]: dims={mvar.dims}, shape={mvar.shape}, size={mvar.size}, ndim={mvar.ndim}, weight={weight}")
#        if mvar.ndim > 0:  print( f"      --> sample: {mvar.values[0:8]}")
#        else:              print( f"      --> sample: {mvar.values}")

    def entries( self, statname: str ) -> Optional[List[xa.DataArray]]:
        return self._stats.get(statname)

class StatsAccumulator:
    statnames = ["mean", "std", "std_diff"]

    def __init__(self, vres: str ):
        self._entries: Dict[str, StatsEntry] = {}
        self.vres: str = vres

    @property
    def entries(self) -> Dict[str, StatsEntry]:
        return self._entries

    def entry(self, varname: str) -> StatsEntry:
        return self._entries.setdefault(varname,StatsEntry(varname))

    @property
    def varnames(self):
        return self._entries.keys()

    def add_entry(self, varname: str, mvar: xa.DataArray):
        istemporal = "tiles" in mvar.dims
        first_entry = varname not in self._entries
        dims = ['tiles', 'y', 'x'] if istemporal else ['y', 'x']
        weight =  mvar.shape[0] if istemporal else 1
        if istemporal or first_entry:
            mean: xa.DataArray = mvar.mean(dim=dims, skipna=True, keep_attrs=True)
            std: xa.DataArray = mvar.std(dim=dims, skipna=True, keep_attrs=True)
            entry: StatsEntry= self.entry( varname)
            entry.add( "mean", mean, weight )
            entry.add("std",  std, weight )
            if istemporal:
                mvar_diff: xa.DataArray = mvar.diff("tiles")
                weight = mvar.shape[0]
                mean_diff: xa.DataArray = mvar_diff.mean( dim=dims, skipna=True, keep_attrs=True )
                std_diff: xa.DataArray  = mvar_diff.std(  dim=dims, skipna=True, keep_attrs=True )
                entry: StatsEntry = self.entry( varname)
                entry.add("mean_diff", mean_diff, weight )
                entry.add("std_diff",  std_diff,  weight )
                times: List[str] = [str(pd.Timestamp(dt64)) for dt64 in mvar.coords['time'].values.tolist()]

    def accumulate(self, statname: str ) -> xa.Dataset:
        accum_stats = {}
        coords = {}
        for varname in self.varnames:
            varstats: StatsEntry = self._entries[varname]
            entries: Optional[List[xa.DataArray]] = varstats.entries( statname )
            squared = statname.startswith("std")
            if entries is not None:
                esum, wsum = None, 0
                for entry in entries:
                    w = entry.attrs['stat_weight']
                    eterm = w*entry*entry if squared else w*entry
                    esum = eterm if (esum is None) else esum + eterm
                    wsum = wsum + w
                astat = np.sqrt( esum/wsum ) if squared else esum/wsum
                accum_stats[varname] = astat
                coords.update( astat.coords )
        return xa.Dataset( accum_stats, coords )

    def save( self, statname: str, filepath: str ):
        os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
        accum_stats: xa.Dataset = self.accumulate(statname)
        accum_stats.to_netcdf( filepath )
        print(f" SSS: Save stats[{statname}] to {filepath}: {list(accum_stats.data_vars.keys())}")
        for vname, vstat in accum_stats.data_vars.items():
            print(f"   >> Entry[{statname}.{vname}]: dims={vstat.dims}, shape={vstat.shape}")
            if vstat.ndim > 0:  print(f"      --> sample: {vstat.values[0:8]}")
            else:               print(f"      --> sample: {vstat.values}")