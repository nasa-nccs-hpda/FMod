from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'edsr' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 200, 'task.lr': 5e-5 }
refresh = False

configuration = dict(
	task = "cape_basin_1x1",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, refresh_state=refresh )
controller.train( models, **ccustom )






