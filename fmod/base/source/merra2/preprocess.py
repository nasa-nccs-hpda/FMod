import xarray as xa, pandas as pd
import numpy as np
from fmod.base.util.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type, Any, Sequence, Mapping, Literal, Hashable
import glob, sys, os, time, traceback
from fmod.base.util.ops import fmbdir
from fmod.base.util.dates import skw, dstr
from datetime import date
from fmod.pipeline.rescale import DataLoader, QType
from fmod.base.util.ops import get_levels_config, increasing, replace_nans
np.set_printoptions(precision=3, suppress=False, linewidth=150)
from numpy.lib.format import write_array
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.pipeline.stats import StatsAccumulator, StatsEntry
from fmod.base.io.loader import ncFormat
from .model import cache_filepath, VarType

_SEC_PER_HOUR =   3600
_HOUR_PER_DAY =   24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR
def nnan(varray: xa.DataArray) -> int: return np.count_nonzero(np.isnan(varray.values))

def sformat(aval: Any) -> str:
    list_method = getattr(aval, "list", None)
    result = str(aval)
    if callable(list_method): result = str(aval.list())
    return result.replace("=",":")

def nodata_test(vname: str, varray: xa.DataArray, d: date):
    num_nodata = nnan(varray)
    assert num_nodata == 0, f"ERROR: {num_nodata} Nodata values found in variable {vname} for date {d}"
def nmissing(varray: xa.DataArray) -> int:
    mval = varray.attrs.get('fmissing_value',-9999)
    return np.count_nonzero(varray.values == mval)
def pctnan(varray: xa.DataArray) -> str: return f"{nnan(varray) * 100.0 / varray.size:.2f}%"
def pctmissing(varray: xa.DataArray) -> str:
    return f"{nmissing(varray) * 100.0 / varray.size:.2f}%"

def dump_dset( name: str, dset: xa.Dataset ):
    print( f"\n ---- dump_dset {name}:")
    for vname, vdata in dset.data_vars.items():
        print( f"  ** {vname}{vdata.dims}-> {vdata.shape} ")

def get_day_from_filename( filename: str ) -> int:
    sdate = filename.split(".")[-2]
    return int(sdate[-2:])

class DailyFiles:

    def __init__(self, collection: str, variables: List[str], day: int, month: int, year: int ):
        self.collection = collection
        self.vars = variables
        self.day = day
        self.month = month
        self.year = year
        self.files = []

    def add(self, file: str ):
        self.files.append( file )

class MERRA2DataProcessor:

    def __init__(self):
        self.format = ncFormat(cfg().task.get('nc_format', 'standard'))
        self.month_range = cfg().preprocess.get('month_range',[0,12,1])
        self.vars: Dict[str, List[str]] = cfg().preprocess.vars
        self.var_file_template =  cfg().platform.dataset_files
        self.const_file_template =  cfg().platform.constant_file
        self.stats = { vres: StatsAccumulator(vres) for vres in ["high",'low'] }
        self.rescaler = DataLoader()

    @classmethod
    def get_qtype( cls, vname: str) -> QType:
        extensive_vars = cfg().preprocess.get('extensive',[])
        return QType.Extensive if vname in extensive_vars else QType.Intensive

    def merge_stats( self, vres: str, stats: List[StatsAccumulator] = None ):
        for stats_accum in ([] if stats is None else stats):
            for varname, new_entry in stats_accum.entries.items():
                entry: StatsEntry = self.stats[vres].entry(varname)
                entry.merge( new_entry )

    def save_stats(self,  proc_stats: List[Dict[str,StatsAccumulator]] ):
        from fmod.base.source.merra2.model import stats_filepath
        for vres in ["high","low"]:
            res_stats = [ pstat[vres] for pstat in proc_stats ]
            self.merge_stats( vres, res_stats )
            for statname in self.stats[vres].statnames:
                filepath = stats_filepath( cfg().preprocess.dataset_version, statname, vres )
                if os.path.exists( filepath ):
                    print( f" ---------------->> Skipping {vres}:{statname} write, stats file alread exists: {filepath}")
                else:
                    self.stats[vres].save( statname, filepath )

    def get_monthly_files(self, year: int, month: int) -> Dict[ str, Tuple[List[str],List[str]] ]:
        dsroot: str = fmbdir('dataset_root')
        assert "{year}" in self.var_file_template, "{year} field missing from platform.cov_files parameter"
        dset_files: Dict[str, Tuple[List[str],List[str]] ] = {}
        assert "{month}" in self.var_file_template, "{month} field missing from platform.cov_files parameter"
        for collection, vlist in self.vars.items():
            if collection.startswith("const"): dset_template: str = self.const_file_template.format( collection=collection )
            else:                              dset_template: str = self.var_file_template.format(   collection=collection, year=year, month=f"{month + 1:0>2}")
            dset_paths: str = f"{dsroot}/{dset_template}"
            gfiles: List[str] = glob.glob(dset_paths)
#            print( f" ** M{month}: Found {len(gfiles)} files for glob {dset_paths}, template={self.var_file_template}, root dir ={dsroot}")
            dset_files[collection] = (gfiles, vlist)
        return dset_files

    def get_daily_files(self, d: date) -> Tuple[ Dict[str, Tuple[str, List[str]]], Dict[str, Tuple[str, List[str]]]]:
        dsroot: str = fmbdir('dataset_root')
        dset_files:  Dict[str, Tuple[str, List[str]]] = {}
        const_files: Dict[str, Tuple[str, List[str]]] = {}
        for collection, vlist in self.vars.items():
            isconst = collection.startswith("const")
            if isconst : fpath: str = self.const_file_template.format(collection=collection)
            else:        fpath: str = self.var_file_template.format(collection=collection, **skw(d))
            file_path = f"{dsroot}/{fpath}"
            if os.path.exists( file_path ):
                dset_list = const_files if isconst else dset_files
                dset_list[collection] = (file_path, vlist)
        return dset_files, const_files

    def write_daily_files(self, filepath: str, collection_dsets: List[xa.Dataset], vres: str):
        merged_dset: xa.Dataset = xa.merge(collection_dsets)
        lgm().log(f"**** write_daily_files({self.format.value}): {filepath}", display=True )
        os.makedirs( os.path.dirname(filepath), exist_ok=True )
        if self.format == ncFormat.DALI:
            self.save_dali_dataset( filepath, merged_dset, vres )
        else:
            for vname, varray in merged_dset.data_vars.items():
                lgm().log(f"   --- {vname}{varray.dims}: shape={varray.shape}, pctnan={pctnan(varray)}" )
            merged_dset.to_netcdf(filepath, format="NETCDF4", mode="w", encoding=self.get_encoding(merged_dset) )
            lgm().log(f"   --- coords: { {c:cv.shape for c,cv in merged_dset.coords.items()} }")

    @classmethod
    def get_encoding(cls, dset: xa.Dataset) -> Dict[Hashable,Dict]:
        encoding = dict()
        for vid, var in dset.data_vars.items():
            chunksizes = list(var.shape).copy()
            try: chunksizes[ var.dims.index('time') ] = 1
            except ValueError: pass
            encoding[vid] = dict(zlib=True, chunksizes=chunksizes)
        return encoding

    @classmethod
    def save_dali_dataset(cls, filepath: str, merged_dset: xa.Dataset, vres: str ):
        os.makedirs(filepath, exist_ok=True)
        hattrs = dict(list(merged_dset.attrs.items()) + [('data_vars', list(merged_dset.data_vars.keys()))])
        for vid, var in merged_dset.data_vars.items():
            hattrs[vid] = [f"{k}={sformat(v)}" for k, v in var.attrs.items()] + [f"dims={','.join(var.dims)}"]
            vfpath = filepath + f"/{vid}.npy"
            with open(vfpath, 'w+b') as fp:
                write_array(fp, var.values, (1, 0), allow_pickle=False)
                lgm().log(f"  > Saving variable {vid} to: {vfpath}")
        hattrs['attrs'] = str(list(merged_dset.attrs.items()))
        header: xa.Dataset = xa.Dataset(merged_dset.coords, attrs=hattrs)
        hfpath = filepath + "/header.nc"
        header.to_netcdf(hfpath, format="NETCDF4", mode="w")
        lgm().log(f"  > Saving header to: {hfpath}")

    def needs_update(self, vtype: VarType, d: date, reprocess: bool ) -> bool:
        if reprocess: return True
        cache_fvpath: str = cache_filepath(vtype, "high", d )
        if not os.path.exists(cache_fvpath): return True
        if self.format == ncFormat.SRES:
            cache_fvpath: str = cache_filepath(vtype, "low", d)
            if not os.path.exists(cache_fvpath): return True
        lgm().log(f" ** Skipping date {d} due to existence of processed files",display=True)
        return False

    def process_day(self, d: date, **kwargs):
        reprocess: bool = kwargs.pop('reprocess', False)
        if self.needs_update( VarType.Dynamic, d, reprocess):
            dset_files, const_files = self.get_daily_files(d)
            ncollections = len(dset_files.keys())
            if ncollections == 0:
                lgm().log( f"No collections found for date {d}", display=True)
            else:
                vres_dsets: Dict[str,List[xa.Dataset]] = dict( high=[], low=[])
                for collection, (file_path, dvars) in dset_files.items():
                    lgm().log(f" >> Loading {collection} from {file_path}: dvvars= {dvars}", display=True)
                    daily_vres_dsets: Dict[str,xa.Dataset] = self.load_collection(  collection, file_path, dvars, d, **kwargs)
                    for vres, dsets in daily_vres_dsets.items(): vres_dsets[vres].append(dsets)
                for vres,collection_dsets in vres_dsets.items():
                    lgm().log(f" --------- Processing {vres} res variable data: {len(collection_dsets)} dsets --------- ")
                    cache_fvpath: str = cache_filepath(VarType.Dynamic, vres, d)
                    self.write_daily_files( cache_fvpath, collection_dsets, vres)
                    lgm().log(f" >> Saving {vres} res collection data for {d} to file '{cache_fvpath}'", display=True)

                if self.needs_update(VarType.Constant, d, reprocess):
                    const_vres_dsets: Dict[str,List[xa.Dataset]] = dict( high=[], low=[])
                    for collection, (file_path, dvars) in const_files.items():
                        lgm().log(f" >> Loading constants for {collection} from {file_path}: dvvars= {dvars}", display=True)
                        daily_vres_dsets: Dict[str,xa.Dataset] = self.load_collection(  collection, file_path, dvars, d, isconst=True, **kwargs)
                        for vres, dsets in daily_vres_dsets.items(): const_vres_dsets[vres].append(dsets)
                    for vres,const_dsets in const_vres_dsets.items():
                        lgm().log(f" --------- Processing {vres} res const data: {len(const_dsets)} dsets --------- ")
                        cache_fcpath: str = cache_filepath( VarType.Constant, vres )
                        if not os.path.exists( cache_fcpath ):
                            self.write_daily_files(cache_fcpath, const_dsets, vres)
                            lgm().log(f" >> Saving {vres} res const data to file '{cache_fcpath}'", display=True)
                    else:
                        lgm().log(f" >> No constant data found")

    def load_collection(self, collection: str, file_path: str, dvnames: List[str], d: date, **kwargs) -> Dict[str,xa.Dataset]:
        dset = xa.open_dataset(file_path)
        lgm().log(f" >> Loading collection '{collection}' from file {file_path}")
        isconst: bool = kwargs.pop( 'isconst', False )
        dset_attrs: Dict = dict(collection=collection, **dset.attrs, **kwargs)
        mvars: Dict[str,Dict[str,xa.DataArray]] = {}
        for vname in dvnames:
            darray: xa.DataArray = dset.data_vars[vname]
            qtype: QType = self.get_qtype(vname)
            ssvars: Dict[str,List[xa.DataArray]] = self.rescaler.rescale( darray, dset_attrs, qtype, isconst )
            lgm().log( f" >> Subsampling variable {vname}({d}): { {res:len(vlist) for (res,vlist) in ssvars.items()} } ")
            for vres, svars in ssvars.items():
                dvars = mvars.setdefault( vres, {} )
                for svar in svars:
                    self.stats[vres].add_entry(vname, svar)
                    nodata_test( vname, svar, d)
                    lgm().log(f" ** Processing {vres} res variable {vname}{svar.dims}: {svar.shape} for {d}")
                    dvars[vname] = svar
        dset.close()
        return { vres: self.create_dataset(dvars,isconst) for vres,dvars in mvars.items() }

    def create_dataset( self, mvars: Dict[str,xa.DataArray], isconst: bool ) -> xa.Dataset:
        result = xa.Dataset(mvars)
        if not isconst:
            self.add_derived_vars(result)
        return result

    @classmethod
    def get_year_progress(cls, seconds_since_epoch: np.ndarray) -> np.ndarray:
        years_since_epoch = (seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR))
        yp = np.mod(years_since_epoch, 1.0).astype(np.float32)
        return yp

    @classmethod
    def get_day_progress(cls, seconds_since_epoch: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        day_progress_greenwich = (np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY)
        longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
        day_progress = np.mod(day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0)
        return day_progress.astype(np.float32)

    @classmethod
    def featurize_progress(cls, name: str, dims: Sequence[str], progress: np.ndarray) -> Mapping[str, xa.Variable]:
        if len(dims) != progress.ndim:
            raise ValueError(f"Number of dimensions in feature {name}{dims} must be equal to the number of dimensions in progress{progress.shape}.")
        else: lgm().log(f"featurize_progress: {name}{dims} --> progress{progress.shape} ")
        progress_phase = progress * (2 * np.pi)
        return {name: xa.Variable(dims, progress), name + "_sin": xa.Variable(dims, np.sin(progress_phase)), name + "_cos": xa.Variable(dims, np.cos(progress_phase))}
    @classmethod
    def add_derived_vars(cls, data: xa.Dataset) -> None:
        if 'datetime' not in data.coords:
            data.coords['datetime'] = data.coords['time'].expand_dims("batch")
        seconds_since_epoch = (data.coords["datetime"].data.astype("datetime64[s]").astype(np.int64))
        batch_dim = ("batch",) if "batch" in data.dims else ()
        year_progress = cls.get_year_progress(seconds_since_epoch)
        data.update(cls.featurize_progress(name=cfg().preprocess.year_progress, dims=batch_dim + ("time",), progress=year_progress))
        longitude_coord = data.coords["x"]
        day_progress = cls.get_day_progress(seconds_since_epoch, longitude_coord.data)
        data.update(cls.featurize_progress(name=cfg().preprocess.day_progress, dims=batch_dim + ("time",) + longitude_coord.dims, progress=day_progress))

    @classmethod
    def get_varnames(cls, dset_file: str) -> List[str]:
        with xa.open_dataset(dset_file) as dset:
            return list(dset.data_vars.keys())


