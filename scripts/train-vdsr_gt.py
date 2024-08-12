from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'vdsr' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 30, 'task.lr': 5e-4, 'pipeline.gpu': 3 }

configuration = dict(
	task = "cape_basin",
	dataset = "LLC4320",
	pipeline = "sres",
	platform = "explore_gt"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )







