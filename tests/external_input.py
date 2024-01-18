from fmod.pipeline.merra2 import MERRA2InputIterator
from fmod.base.util.config import configure, cfg
import hydra, os, time, numpy as np

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )

source = MERRA2InputIterator()

for result in source:
	print(result)
	break