from fmod.base.util.config import configure, cfg
from typing import List, Tuple
from datetime import date
from multiprocessing import Pool, cpu_count
import hydra, os, time, numpy as np
from fmod.base.source.merra2.pipeline import load_batch
from nvidia.dali.pipeline import pipeline_def

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
nproc = cpu_count()-2
start: date = date(1990,4,1)
batch_size = 1

@pipeline_def(batch_size=batch_size, num_threads=nproc, device_id=0)
def get_dali_pipeline():
	batch = load_batch( start )
	return batch

p = get_dali_pipeline()
t0 = time.time()

p.build()

t1 = time.time()

result = p.run()

print( f"Completed run in {time.time()-t1} sec (build: {t1-t0} sec), result = {type(result)}")






