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
    return Configuration.instance().cfg

def cid() -> str:
    return '-'.join([ cfg().task.name, cfg().model.name, cfg().task.dataset, cfg().task.scenario ])

def fmconfig( task: str, model: str, dataset: str, scenario: str, ccustom: Dict[str,Any], pipeline: str="sres", server: str="explore", log_level=logging.INFO ) -> DictConfig:
    taskname = f"{task}-{dataset}-{scenario}"
    overrides = {'task':taskname, 'model':model, 'platform':server, 'dataset':dataset, 'pipeline':pipeline}
    Configuration.init( pipeline, dict( **overrides, **ccustom) )
    cfg().task.name = taskname
    cfg().task.scenario = scenario
    cfg().task.dataset = dataset
    cfg().task.training_version = f"{model}-{dataset}-{scenario}"
    lgm().set_level( log_level )
    return Configuration.instance().cfg

def cfgdir() -> str:
    cdir = Path(__file__).parent.parent.parent / "config"
    print( f'cdir = {cdir}')
    return str(cdir)

class ConfigContext(initialize):

    def __init__(self, task: str, model: str, dataset: str, scenario: str, ccustom: Dict[str,Any], pipeline: str="sres", server: str="explore", log_level=logging.WARN, config_path="../../../config"):
        super(ConfigContext,self).__init__(config_path)
        task: str = task
        self.model: str = model
        self.task: str = task
        self.dataset: str = dataset
        self.scenario: str = scenario
        self.log_level: int = log_level
        self.cfg: DictConfig = None
        self.name: str = None
        self.ccustom: Dict[str,Any] = ccustom
        self.pipeline: str = pipeline
        self.server: str = server

    def __enter__(self, *args: Any, **kwargs: Any):
       super(ConfigContext, self).__enter__( *args, **kwargs)
       self.cfg = fmconfig( self.task, self.model, self.dataset, self.scenario, self.ccustom, self.pipeline, self.server, self.log_level)
       self.name = Configuration.instance().config_name
       print( 'Entering cfg-context: ', self.name )
       return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
       super(ConfigContext, self).__exit__(exc_type, exc_val, exc_tb )
       print( 'Exiting cfg-context: ', self.name )
       Configuration.clear()
       self.cfg = None
       self.name = None
       if exc_type is not None:
           traceback.print_exception( exc_type, value=exc_val, tb=exc_tb)

class Configuration(ABC):
    _instance = None
    _instantiated = None

    def __init__( self, config_name: str, overrides: Dict[str,Any] ):
        self.config_name = config_name
        if not GlobalHydra().is_initialized():
            hydra.initialize(version_base=None, config_path="../../../config")
        self.compose = hydra.compose(config_name=self.config_name, overrides=[f"{ov[0]}={ov[1]}" for ov in overrides.items()])
        self.cfg: DictConfig = self.compose

    @classmethod
    def init(cls, config_name: str, overrides: Dict[str,Any] ):
        if cls._instance is None:
            inst = cls( config_name, overrides )
            cls._instance = inst
            cls._instantiated = cls
            print(f' *** Configuration {config_name} initialized *** ')

    @classmethod
    def clear(cls):
        cls._instance = None


    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


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


