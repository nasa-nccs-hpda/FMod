from fmod.base.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmod.base.util.config import configure, cfg
from typing import List, Tuple
from datetime import date
from fmod.base.util.dates import date_range
from fmod.base.source.merra2.model import clear_const_file
from multiprocessing import Pool, cpu_count
from fmod.base.source.merra2.preprocess import ncFormat
import hydra, os

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
reprocess=True
nc_format = ncFormat( cfg().preprocess.get('nc_format','standard') )
nproc = cpu_count()-2
start: date = date(1990,4,1)
end: date = date(1990,5,1)

def process( d: date ) -> StatsAccumulator:
	reader = MERRA2DataProcessor(nc_format)
	reader.process_day( d, reprocess=reprocess)
	return reader.stats

if __name__ == '__main__':
	dates: List[date] = date_range( start, end )
	print( f"Multiprocessing {len(dates)} days with {nproc} procs")
	if reprocess: clear_const_file()
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, dates )
		MERRA2DataProcessor().save_stats(proc_stats)



