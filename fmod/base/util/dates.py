from typing import Any, Dict, List, Tuple, Type, Optional, Union
from datetime import date, timedelta
from fmod.base.util.config import start_date
import random

def kw(d: date) -> Dict[str,int]:
	return dict( day=d.day, month=d.month, year=d.year )

def skw(d: date) -> Dict[str,str]:
	return dict( year = syear(d), month = smonth(d) , day = sday(d) )

def smonth(d: date) -> str:
	return f"{d.month:0>2}"

def sday(d: date) -> str:
	return f"{d.day:0>2}"

def syear(d: date) -> str:
	return str(d.year)

def dstr(d: date) -> str:
	return syear(d) + smonth(d) + sday(d)

def drepr(d: date) -> str:
	return f'{d.year}-{d.month}-{d.day}'

def next(d: date) -> date:
	return d + timedelta(days=1)

def date_list( start: date, num_days: int )-> List[date]:
	d0: date = start
	dates: List[date] = []
	for iday in range(0,num_days):
		dates.append(d0)
		d0 = next(d0)
	return dates

def cfg_date_range( task_config )-> List[date]:
	start = date( task_config.start_date )
	end = date(task_config.end_date)
	return date_range( start, end )
def date_range( start: date, end: date )-> List[date]:
	d0: date = start
	dates: List[date] = []
	while d0 < end:
		dates.append( d0 )
		d0 = next(d0)
	return dates

def year_range( y0: int, y1: int, **kwargs )-> List[date]:
	randomize: bool = kwargs.get( 'randomize', False )
	rlist = date_range( date(y0,1,1), date(y1,1,1) )
	if randomize: random.shuffle(rlist)
	return rlist

def batches_range( task_config )-> List[date]:
	return date_list( start_date( task_config ), task_config.batch_ndays*task_config.nbatches )


