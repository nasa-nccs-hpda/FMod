from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'rcan-10-20-64' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 1, 'task.lr': 1e-4 }
refresh =  False

configuration = dict(
	task = "swot",
	dataset = "swot_gt",
	pipeline = "sres",
	platform = "explore_gt"
)

controller = WorkflowController( cname, configuration, refresh_state=refresh, interp_loss=True )
controller.train( models, **ccustom )







