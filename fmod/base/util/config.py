import logging
import xarray, warnings, torch
import xarray.core.coordinates
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
from dataclasses import dataclass
from fmod.base.util.logging import lgm, exception_handled, log_timing
from datetime import date, timedelta, datetime
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
import hydra, traceback, os
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
DataCoordinates = Union[DataArrayCoordinates,DatasetCoordinates]

def cfg() -> DictConfig:
    return ConfigContext.cfg

def config() -> Dict:
    return ConfigContext.configuration

def cid() -> str:
    return '-'.join([ cfg().model.name, cfg().task.dataset, cfg().task.name ])

def cfgdir() -> str:
    cdir = Path(__file__).parent.parent.parent / "config"
    print( f'cdir = {cdir}')
    return str(cdir)

class ConfigContext(initialize):
    cfg: Optional[DictConfig] = None
    defaults: Dict = {}
    configuration: Dict = {}

    def __init__(self, name: str, **kwargs ):
        assert self.cfg is None, "Only one ConfigContext instance is allowed at a time"
        self.name = name
        ConfigContext.configuration = dict(**self.defaults, **kwargs)
        self.model: str = self.get_config('model')
        self.pipeline: str = self.get_config('pipeline')
        self.platform: str = self.get_config('platform')
        self.task: str = self.get_config('task')
        self.dataset: str = self.get_config('dataset')
        self.config_path: str = self.get_config('config_path', "../../../config")
        self.cid = '-'.join( [self.name, self.model, self.dataset, self.task] )
        super(ConfigContext, self).__init__(version_base=None, config_path=self.config_path)

    def get_config(self,name: str, default: Any = None ):
        return self.configuration.get( name, self.defaults.get(name,default) )

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.defaults = kwargs

    @property
    def cfg_file( self ):
        currdir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath( os.path.join(currdir, self.config_path,  f"{self.name}.yaml") )

    @classmethod
    def deactivate(cls):
        cls.cfg = None

    @classmethod
    def activate_global(cls, name: str, **kwargs ) -> 'ConfigContext':
        cc = ConfigContext( name, **kwargs )
        cc.activate()
        return cc

    def activate(self):
        assert ConfigContext.cfg is None, "Context already activated"
        cfg = ConfigContext.cfg = self.load()
        gpu = self.configuration.get( 'gpu', int(os.getenv('FMOD_GPU',cfg.pipeline.gpu)) )
        cfg.pipeline.gpu = gpu
        print( f"Activating {self.name}: '{self.cfg_file}', gpu={gpu}, keys = {list(self.cfg.keys())}")
        cfg.task.name = self.task
        cfg.task.dataset = self.dataset
        cfg.task.training_version = self.cid

    def load(self) -> DictConfig:
        assert self.cfg is None, "Another Config context has already been activateed"
        if not GlobalHydra().is_initialized():
            hydra.initialize(version_base=None, config_path=self.config_path)
        print( f"load {self.name}: config = {self.configuration}")
        return hydra.compose(config_name=self.name, overrides=[f"{ov[0]}={ov[1]}" for ov in self.configuration.items()])

    def __enter__(self, *args: Any, **kwargs: Any):
       super(ConfigContext, self).__enter__(*args, **kwargs)
       self.activate()
       print( 'Entering cfg-context: ', self.name )
       return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
       super(ConfigContext, self).__exit__(exc_type, exc_val, exc_tb)
       print( 'Exiting cfg-context: ', self.name )
       self.deactivate()
       if exc_type is not None:
           traceback.print_exception( exc_type, value=exc_val, tb=exc_tb)

    def __del__ (self):
        self.deactivate()


def cfg2meta(csection: str, meta: object, on_missing: str = "ignore"):
    csections = csection.split(".")
    cmeta = cfg().get(csections[0])
    if (len(csections) > 1) and (cmeta is not None): cmeta = cmeta.get(csections[1])
    if cmeta is None:
        print( f"Warning: section '{csection}' does not exist in configuration" )
        return None
    for k,v in cmeta.items():
        valid = True
        if (getattr(meta, k, None) is None) and (on_missing != "ignore"):
            msg = f"Attribute '{k}' does not exist in metadata object"
            if on_missing.startswith("warn"): print("Warning: " + msg)
            elif on_missing == "skip": valid = False
            elif on_missing.startswith("excep"): raise Exception(msg)
            else: raise Exception(f"Unknown on_missing value in cfg2meta: {on_missing}")
        if valid: setattr(meta, k, v)
    return meta

def cfg2args( csection: str, pnames: List[str] ) -> Dict[str,Any]:
    csections = csection.split(".")
    cmeta = cfg().get(csections[0])
    if (len(csections) > 1) and (cmeta is not None): cmeta = cmeta.get(csections[1])
    args = {}
    if cmeta is None:
        print( f"Warning: section '{csection}' does not exist in configuration" )
    else:
        for pn in pnames:
            if pn in cmeta.keys():
                aval = cmeta.get(pn)
                if str(aval) == "None": aval = None
                args[pn] = aval
    return args

def cfg_date( csection: str ) -> date:
    dcfg = cfg().get(csection)
    return date( dcfg.year, dcfg.month, dcfg.day )

def start_date( task: DictConfig )-> Optional[datetime]:
    startdate = task.get('start_date', None)
    if startdate is None: return None
    toks = [ int(tok) for tok in startdate.split("/") ]
    return  datetime( month=toks[0], day=toks[1], year=toks[2] )

def dateindex(d: datetime, task: DictConfig) -> int:
    sd: date = start_date(task)
    dt: timedelta = d - sd
    hours: int = (dt.seconds // 3600) + (dt.days * 24)
    # print( f"dateindex: d[{d.strftime('%H:%d/%m/%Y')}], sd[{sd.strftime('%H:%d/%m/%Y')}], dts={dt.seconds}, hours={hours}")
    return hours + 1

def index_of_value( array: np.ndarray, target_value: float ) -> int:
    differences = np.abs(array - target_value)
    return differences.argmin()

def closest_value( array: np.ndarray, target_value: float ) -> float:
    differences = np.abs(array - target_value)
    print( f"Closest value: array{array.shape}, target={target_value}, differences type: {type(differences)}")
    return  float( array[ differences.argmin() ] )

def get_coord_bounds( coord: np.ndarray ) -> Tuple[float, float]:
    dc = coord[1] - coord[0]
    return  float(coord[0]), float(coord[-1]+dc)

def get_dims( coords: DataCoordinates, **kwargs ) -> List[str]:
    dims = kwargs.get( 'dims', ['x','y'] )
    dc: List[Hashable] = list(coords.keys())
    if 'x' in dc:
        return dims
    else:
        cmap: Dict[str, str] = cfg().task.coords
        vs: List[str] = list(cmap.values())
        if vs[0] in dc:
            return [ cmap[k] for k in dims ]
        else:
            raise Exception(f"Data Coordinates {dc} do not exist in configuration")

def get_roi( coords: DataCoordinates ) -> Dict:
    return { dim: get_coord_bounds( coords[ dim ].values ) for dim in get_dims(coords) }

def get_data_coords( data: xarray.DataArray, target_coords: Dict[str,float] ) -> Dict[str,float]:
    return { dim: closest_value( data.coords[ dim ].values, cval ) for dim, cval in target_coords.items() }

def cdelta(dset: xarray.DataArray):
    return { k: float(dset.coords[k][1]-dset.coords[k][0]) for k in dset.coords.keys() if dset.coords[k].size > 1 }

def cval( data: xarray.DataArray, dim: str, cindex ) -> float:
    coord : np.ndarray = data.coords[ cfg().task.coords[dim] ].values
    return float( coord[cindex] )

def get_data_indices( data: Union[xarray.DataArray,xarray.Dataset], target_coords: Dict[str,float] ) -> Dict[str,int]:
    return { dim: index_of_value( data.coords[ dim ].values, coord_value ) for dim, coord_value in target_coords.items() }


