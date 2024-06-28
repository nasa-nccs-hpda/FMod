from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'edsr' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 5, 'task.lr': 5e-5 }

configuration = dict(
	task = "cape_basin_3x3",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )






