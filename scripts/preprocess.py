from fmbase.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmbase.util.config import configure, cfg
from typing import List, Tuple
from datetime import date
from fmbase.util.dates import year_range
from fmbase.source.merra2.model import clear_const_file
from multiprocessing import Pool, cpu_count
import hydra, os

hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-finetuning' )
reprocess=False
nproc = cpu_count()-2
yrange: Tuple[int,int] = cfg().preprocess.year_range

def process( d: date ) -> StatsAccumulator:
	reader = MERRA2DataProcessor()
	reader.process_day( d, reprocess=reprocess)
	return reader.stats

if __name__ == '__main__':
	dates: List[date] = year_range( *yrange )
	print( f"Multiprocessing {len(dates)} days with {nproc} procs")
	if reprocess: clear_const_file()
	with Pool(processes=nproc) as pool:
		proc_stats: List[StatsAccumulator] = pool.map( process, dates )
		MERRA2DataProcessor().save_stats(proc_stats)



