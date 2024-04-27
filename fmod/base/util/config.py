import xarray
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from dataclasses import dataclass
from fmod.base.util.logging import lgm, exception_handled, log_timing
from datetime import date, timedelta
import hydra, traceback, os
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str):
    Configuration.init( config_name )
 #   warnings.filterwarnings("error")
    lgm().log(f"Config loaded: {config_name}")

def cfgdir() -> str:
    cdir = Path(__file__).parent.parent.parent / "config"
    print( f'cdir = {cdir}')
    return str(cdir)

class ConfigBase(ABC):
    _instance = None
    _instantiated = None

    def __init__(self, config_name: str, **kwargs ):
        self.config_name = config_name
        self.cfg: DictConfig = self.get_parms(**kwargs)

    @abstractmethod
    def get_parms(self, **kwargs) -> DictConfig:
        return None

    @classmethod
    def init(cls, config_name: str ):
        if cls._instance is None:
            inst = cls( config_name )
            cls._instance = inst
            cls._instantiated = cls
            print(f' *** Configuration {config_name} initialized *** ')


    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


class Configuration(ConfigBase):
    def get_parms(self, **kwargs) -> DictConfig:
        return hydra.compose(self.config_name, return_hydra_config=True)

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

def start_date( task_config )-> date:
    toks = [ int(tok) for tok in reversed(task_config.start_date.split("/")) ]
    print( f"Task start date: {task_config.start_date}: {toks}")
    return  date( *toks )

def get_coord_bounds( coord: np.ndarray ) -> Tuple[float, float]:
    dc = coord[1] - coord[0]
    return  float(coord[0]), float(coord[-1]+dc)

def get_roi( coords: Dict[str,xarray.DataArray] ) -> Dict:
    cmap: Dict = cfg().task.coords
    return { dim: get_coord_bounds( coords[ cmap[dim] ].values ) for dim in ['x','y'] }
