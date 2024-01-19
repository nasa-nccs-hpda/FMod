from fmod.pipeline.merra2 import MERRA2InputIterator, MetaData
from fmod.base.util.config import configure, cfg2meta
import hydra, os, time, numpy as np

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
pmeta: MetaData =cfg2meta('pipeline', MetaData(), on_missing="skip" )

source = MERRA2InputIterator()

for result in source:
	break