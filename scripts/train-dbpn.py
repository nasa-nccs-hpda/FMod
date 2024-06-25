from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.controller.workflow import WorkflowController

cname: str = "sres"
models: List[str] = [ 'dbpn' ] # [ 'dbpn', 'edsr', 'srdn', 'unet', 'vdsr', 'mscnn', 'lapsrn' ]
ccustom: Dict[str,Any] = { 'task.nepochs': 10, 'pipeline.gpu': 0 }

configuration = dict(
	task = "sres",
	dataset = "LLC4320",
	scenario = "s4",
	pipeline = "sres",
	platform = "explore"
)

controller = WorkflowController( cname, configuration )
controller.train( models, **ccustom )



