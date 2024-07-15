from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'rcan-10-10-64' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 100, 'task.lr': 1e-4 }
refresh =  False

configuration = dict(
	task = "swot",
	dataset = "swot",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, refresh_state=refresh )
controller.train( models, **ccustom )







