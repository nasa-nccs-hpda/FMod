import xarray as xa, numpy as np
from omegaconf import DictConfig, OmegaConf
import linecache
from sparrow.base.util import vrange
from pathlib import Path
from sparrow.base.config import cfg
from typing import List, Union, Tuple, Optional, Dict, Type
import hydra, glob, sys, os, time

def year2date( year: Union[int,str] ) -> np.datetime64:
    return np.datetime64( int(year) - 1970, 'Y')

def is_float( string: str ) -> bool:
    try: float(string); return True
    except ValueError:  return False

def is_int( string: str ) -> bool:
    try: int(string);  return True
    except ValueError: return False

def str2num( string: str ) -> Union[float,int,str]:
    try: return int(string)
    except ValueError:
        try: return float(string)
        except ValueError:
            return string

def xmin( v: xa.DataArray ):
    return v.min(skipna=True).values.tolist()

def xmax( v: xa.DataArray ):
    return v.max(skipna=True).values.tolist()

def xrng( v: xa.DataArray ):
    return [ xmin(v), xmax(v) ]

def srng( v: xa.DataArray ):
    return f"[{xmin(v):.5f}, {xmax(v):.5f}]"

class CovariateDataProcessor:

    def __init__(self):
        self.xext, self.yext = cfg().scenario.get('xext'), cfg().scenario.get('yext')
        self.xres, self.yres = cfg().scenario.get('xres'), cfg().scenario.get('yres')
        self.xcDset, self.ycDset = cfg().covariates.xcoord, cfg().covariates.ycoord
        self.year_range = cfg().scenario.year_range
        self.month_range = cfg().scenario.get('month_range',[0,12,1])
        self.file_template = cfg().platform.cov_files
        self.cache_file_template = cfg().scenario.cache_file_template
        self.cfgId = cfg().scenario.id
        self.xcCache, self.ycCache = cfg().scenario.xc, cfg().scenario.yc
        if self.yext is None: self.yext, self.xci = None, None
        else:
            self.yci = np.arange( self.yext[0], self.yext[1]+self.yres/2, self.yres )
            self.xci = np.arange( self.xext[0], self.xext[1]+self.xres/2, self.xres )

    @property
    def data_dir(self):
        return cfg().dataset.dataset_root.format( **cfg().platform )

    @property
    def cache_dir(self):
        return cfg().platform.cache.format( **cfg().platform )

    def get_yearly_files(self, collection, year) -> List[str]:
        months = list(range(*self.month_range))
        assert "{year}" in self.file_template, "{year} field missing from platform.cov_files parameter"
        dset_template = self.file_template.format(collection=collection, year=year, month=0).replace("00*", "*")
        dset_paths = f"{self.data_dir}/{dset_template}"
        if len(months) == 12:
            dset_files = glob.glob(dset_paths)
        else:
            dset_files = []
            assert "{month}" in self.file_template, "{month} field missing from platform.cov_files parameter"
            for month in months:
                dset_template = self.file_template.format(collection=collection, year=year, month=month)
                dset_paths = f"{self.data_dir}/{dset_template}"
                dset_files.extend( glob.glob(dset_paths) )
        if len(dset_files) == 0: print( f"Unable to find any covariate data for glob: {dset_paths}" )
        return dset_files

    def load_asc(self, filepath: str, flipud=True ) -> xa.DataArray:
        raster_data: np.array = np.loadtxt( filepath, skiprows=6 )
        if flipud: raster_data = np.flipud( raster_data )
        header, varname = {}, Path(filepath).stem
        for hline in range(6):
            header_line = linecache.getline(filepath, hline).split(' ')
            if len(header_line) > 1:
                header[ header_line[0].strip() ] = str2num( header_line[1] )
        nodata: float = header.get( 'NODATA_VALUE', -9999.0 )
        raster_data[ raster_data==nodata ] = np.nan
        cs, xlc, ylc, nx, ny = header['CELLSIZE'], header['XLLCORNER'], header['YLLCORNER'], header['NCOLS'], header['NROWS']
        xc = np.array( [ xlc + cs*ix for ix in range(nx)] )
        yc = np.array([ylc + cs * iy for iy in range(ny)])
        header['_FillValue'] = np.nan
        header['long_name'] = varname
        header['varname'] = varname
        header['xres'] = cs
        header['yres'] = cs
        return xa.DataArray( raster_data, name=varname, dims=['lat','lon'], coords=dict(lat=yc,lon=xc), attrs=header )

    def process_merramax(self):
        years = list(range(*self.year_range))
        for year in years:
            dset_template = self.file_template.format(year=year)
            dset_paths = f"{self.data_dir}/{dset_template}"
            dset_files = glob.glob(dset_paths)
            for dset_file in dset_files:
                vardata: xa.DataArray = self.load_asc( dset_file ).expand_dims( dim=dict( time=[year2date(year)] ) )
                dset_attrs = dict( year=year, collection="MERRAMAX", varname=vardata.name )
                dset: xa.Dataset = self.create_cache_dset( vardata, dset_attrs )
                filepath = self.variable_cache_filepath( str(vardata.name), year )
                os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
                print( f"Writing cache file {filepath}, vrange={vrange(vardata)}")
                dset.to_netcdf(filepath)

    def process(self, collection: str = None, **kwargs):
        years = list(range( *self.year_range ))
        reprocess = kwargs.get( 'reprocess', False )
        print(f"\n --------------- Processing collection {collection}  --------------- ")
        for year in years:
            t0 = time.time()
            dset_files = self.get_yearly_files( collection, year )
            covars: List[str] = self.get_covnames( dset_files[0] )
            if len( covars ) == 0:
                print(f" ** No covariates in this collection")
                return
            if not reprocess and self.cache_files_exist( covars, year ):
                print(f" ** Skipping already processed year {year}")
            else:
                print(f" ** Loading dataset files for covariates {covars}, year={year}")
                agg_dataset: xa.Dataset =  self.open_collection( collection, dset_files, year=year )
                print(f" -- -- Processing {len(dset_files)} files, load time = {time.time()-t0:.2f} ")
                for covar in covars:
                    self.proccess_variable( covar, agg_dataset, **kwargs )
                agg_dataset.close()

    def cache_files_exist(self, varnames: List[str], year: int ) -> bool:
        for vname in varnames:
            filepath = self.variable_cache_filepath(vname, year )
            if not os.path.exists( filepath ): return False
        return True

    def get_covnames(self, dset_file: str ) -> List[str]:
        dset: xa.Dataset = xa.open_dataset(dset_file)
        covnames = [vname for vname in dset.data_vars.keys() if vname in cfg().scenario.vars]
        dset.close()
        return covnames

    def get_covariates(self, dset: xa.Dataset ) -> Dict[str,xa.DataArray]:
        covariates: Dict[str,xa.DataArray] = { vid: dvar for vid, dvar in dset.data_vars.items() if vid in cfg().scenario.vars }
        return { vid: dvar.where(dvar != dvar.attrs['fmissing_value'], np.nan) for vid, dvar in covariates.items()}

    def open_collection(self, collection, files: List[str], **kwargs) -> xa.Dataset:
        print( f" -----> open_collection[{collection}:{kwargs['year']}]>> {len(files)} files, Compute yearly averages: ", end="")
        t0 = time.time()
        dset: xa.Dataset = xa.open_mfdataset(files)
        dset_attrs = dict( collection=os.path.basename(collection), **dset.attrs, **kwargs )
        resampled_dset = xa.Dataset( self.get_covariates( dset ), dset.coords ).resample(time='AS').mean('time')
        resampled_dset.attrs.update(dset_attrs)
        print( f" Loaded {len(resampled_dset.data_vars)} vars in time = {time.time()-t0:.2f} sec")
        for vid, dvar in resampled_dset.data_vars.items(): dvar.attrs.update( dset.data_vars[vid].attrs )
        return resampled_dset

    def resample_variable(self, variable: xa.DataArray) -> xa.DataArray:
        scoords = {self.xcDset: slice(self.xext[0], self.xext[1]), self.ycDset: slice(self.yext[0], self.yext[1])}
        newvar: xa.DataArray = variable.sel(**scoords)
        newvar.attrs.update( variable.attrs )
        xc, yc = newvar.coords[self.xcDset].values, newvar.coords[self.ycDset].values
        newvar.attrs['xres'], newvar.attrs['yres'] = (xc[1]-xc[0]).tolist(), (yc[1]-yc[0]).tolist()
        newvar.attrs['fmissing_value'] = np.nan
        return newvar

    def variable_cache_filepath(self, vname: str, year: int ) -> str:
        filename = self.cache_file_template.format( varname=vname, year=year )
        return f"{self.cache_dir}/{self.cfgId}/{filename}"

    def proccess_variable(self, varname: str, agg_dataset: xa.Dataset, **kwargs ):
        t0 = time.time()
        reprocess = kwargs.get('reprocess',False)
        variable: xa.DataArray = agg_dataset.data_vars[varname]
        interp_var = self.resample_variable(variable)
        filepath = self.variable_cache_filepath( varname, agg_dataset.attrs['year'] )
        if reprocess or not os.path.exists(filepath):
            print(f" ** ** ** Processing variable {variable.name}, file= {filepath} ")
            dset: xa.Dataset = self.create_cache_dset(interp_var, agg_dataset.attrs )
            os.makedirs(os.path.dirname(filepath), mode=0o777, exist_ok=True)
            dset.to_netcdf(filepath)
            print(f" ** ** ** >> Writing cache data file: {filepath}, time= {time.time()-t0} sec.")
        else:
            print(f" ** ** ** >> Skipping existing variable {variable.name}, file= {filepath} ")

    def create_cache_dset(self, vdata: xa.DataArray, dset_attrs: Dict ) -> xa.Dataset:
        t0 = time.time()
        year, cname = dset_attrs['year'], "covariate"
        ccords = { 'time': vdata.coords['time'], self.xcCache: vdata.coords[self.xcDset], self.ycCache: vdata.coords[self.ycDset] }
        global_attrs = dict( **dset_attrs )
        global_attrs.update( varname=vdata.name, year=year )
        t1 = time.time()
        data_array = xa.DataArray( vdata.values, ccords, ['time',self.ycCache,self.xcCache], attrs=vdata.attrs, name=cname )
        print( f" ** ** ** >> Created cache dataset, shape={vdata.shape}: time = {t1-t0:.2f} {time.time()-t1:.2f} sec, vrange = {[vrange(vdata)]}")
        global_attrs['RangeStartingDate'] =  f"{year-1}-12-31"
        global_attrs['RangeStartingTime']  = "23:30:00.000000"
        global_attrs['RangeEndingDate'] =  f"{year}-12-31"
        global_attrs['RangeEndingTime']  = "23:30:00.000000"
        return xa.Dataset( {cname: data_array}, ccords, global_attrs )