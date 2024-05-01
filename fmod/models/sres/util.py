from typing import Any, Dict, List, Tuple, Type, Optional, Union, TypeAlias, Callable, Mapping
Size2: TypeAlias = Union[int,Tuple[int,int]]

def highest_powerof2(n: int) -> int:
	res = 0
	for i in range(n, 0, -1):
		# If i is a power of 2
		if (i & (i - 1)) == 0:
			res = i
			break

	return res