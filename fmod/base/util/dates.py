from typing import Any, Dict, List, Tuple, Type, Optional, Union
from datetime import date, datetime, timedelta
import random
TimeType = Union[datetime, int]

def kw(d: datetime) -> Dict[str,int]:
	return dict( day=d.day, month=d.month, year=d.year )

def skw(d: datetime) -> Dict[str,str]:
	return dict( year = syear(d), month = smonth(d) , day = sday(d) )

def smonth(d: datetime) -> str:
	return f"{d.month:0>2}"

def sday(d: datetime) -> str:
	return f"{d.day:0>2}"

def syear(d: datetime) -> str:
	return str(d.year)

def dstr(d: datetime) -> str:
	return syear(d) + smonth(d) + sday(d)

def drepr(d: datetime) -> str:
	return f'{d.year}-{d.month}-{d.day}'

def next(d: datetime) -> datetime:
	return d + timedelta(days=1)

def date_list( start: Optional[datetime], num_days: int )-> List[datetime]:
	dates: List[datetime] = []
	if (start is not None) and (num_days > 0):
		d0: datetime = start
		for iday in range(0,num_days):
			dates.append(d0)
			d0 = next(d0)
	return dates

def date_bounds( start: datetime, num_days: int )-> Tuple[datetime,datetime]:
	return start, start+timedelta(days=num_days)

def cfg_date_range( task_config )-> List[datetime]:
	start = date( str(task_config['start_date']) )
	end = date( str(task_config['end_date']) )
	return date_range( start, end )

def date_range( start: datetime, end: datetime )-> List[datetime]:
	d0: datetime = start
	dates: List[datetime] = []
	while d0 < end:
		dates.append( d0 )
		d0 = next(d0)
	return dates

def year_range( y0: int, y1: int, **kwargs )-> List[datetime]:
	randomize: bool = kwargs.get( 'randomize', False )
	rlist = date_range( datetime(y0,1,1), datetime(y1,1,1) )
	if randomize: random.shuffle(rlist)
	return rlist



