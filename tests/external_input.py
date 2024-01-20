from fmod.pipeline.merra2 import MetaData, MERRA2NCDatapipe
from fmod.base.util.config import configure, cfg2meta
import hydra, os, time, numpy as np

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
pmeta: MetaData =cfg2meta('pipeline', MetaData(), on_missing="skip" )

pipe = MERRA2NCDatapipe(pmeta)
pipe.build()
pipe_out = pipe.run()
inputs = pipe_out[0][0]
targets = pipe_out[1][0]

print( f"TEST COMPLETE: {type(inputs)} {type(targets)}")