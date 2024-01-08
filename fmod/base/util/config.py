from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from dataclasses import dataclass
from datetime import date, timedelta
import hydra

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
