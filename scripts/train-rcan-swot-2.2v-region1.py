from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'rcan-10-20-64' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 100, 'task.lr': 1e-4 }
refresh =  True

configuration = dict(
	task = "swot-2.2v",
	dataset = "swot_southpacific_1200",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration, refresh_state=refresh, interp_loss=True )
controller.train( models, **ccustom )







