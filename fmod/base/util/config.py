from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from dataclasses import dataclass
from datetime import date, timedelta
import hydra, traceback, os
os.environ["DGLBACKEND"] = "pytorch"

def cfg() -> DictConfig:
    return Configuration.instance().cfg

def configure(config_name: str):
    Configuration.init( config_name )

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

    @classmethod
    def instance(cls) -> "Configuration":
        return cls._instance


class Configuration(ConfigBase):
    def get_parms(self, **kwargs) -> DictConfig:
        return hydra.compose(self.config_name, return_hydra_config=True)

def cfg2meta(csection: str, meta: object, on_missing: str = "ignore"):
    cmeta = cfg().get(csection)
    assert cmeta is not None, f"Section '{csection}' not found in hydra configuratrion"
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
