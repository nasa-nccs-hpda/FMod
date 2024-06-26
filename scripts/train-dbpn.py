from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'dbpn' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 3, 'pipeline.gpu': 0 }

configuration = dict(
	task = "cape_basin_3x3",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )



