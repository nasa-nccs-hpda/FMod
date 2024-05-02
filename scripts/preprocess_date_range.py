from fmod.base.source.merra2.preprocess import MERRA2DataProcessor, StatsAccumulator
from fmod.base.util.config import configure, cfg
from typing import List, Tuple, Dict
from datetime import date
from fmod.base.util.dates import date_range
from fmod.base.source.merra2.model import clear_const_file
import hydra, os  
from multiprocessing import Pool, cpu_count


hydra.initialize( version_base=None, config_path="../config" )
configure( 'merra2-srdn-s1' )
reprocess=True
nproc = cpu_count()
start: date = date(1995,1,17 )
end: date = date(1996,1,1 )

def process( d: date ) -> Dict[str,StatsAccumulator]:
	reader = MERRA2DataProcessor()
	reader.process_day( d, reprocess=reprocess)
	return reader.stats

if __name__ == '__main__':
	dates: List[date] = date_range( start, end )
	print( f"Multiprocessing {len(dates)} days with {nproc} procs")
	if reprocess: clear_const_file()
	if nproc > 1:
		with Pool(processes=nproc) as pool:
			proc_stats: List[Dict[str,StatsAccumulator]] = pool.map( process, dates )
			MERRA2DataProcessor().save_stats(proc_stats)
	else:
		for d in  dates:
			proc_stats: Dict[str,StatsAccumulator] = process(d)
			MERRA2DataProcessor().save_stats( [proc_stats] )



