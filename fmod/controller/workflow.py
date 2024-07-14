import time
from fmod.base.util.config import ConfigContext
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.util.logging import lgm, exception_handled, log_timing
from fmod.base.gpu import save_memory_snapshot
from fmod.base.io.loader import TSet, srRes
from typing import Any, Dict, List
from fmod.view.plot.results import ResultPlot
from fmod.view.plot.training import TrainingPlot
from fmod.view.plot.base import Plot

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.refresh_state = kwargs.get('refresh_state', False )
		self.config: ConfigContext = None
		self.trainer: ModelTrainer = None
		self.plot: Plot = None
		self.model = None
		ConfigContext.set_defaults( **configuration )

	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				try:
					self.config = cc
					self.trainer = ModelTrainer(cc)
					self.trainer.train(refresh_state=self.refresh_state, seed=self.seed)
				except Exception as e:
					lgm().exception( "Exception while training model: %s" % str(e) )
					save_memory_snapshot()

				lgm().log(f"Completed training model: {model}")

	def init_plotting(self, cname, model, **kwargs ):
		self.model = model
		self.config = ConfigContext.activate_global( cname, model=model, **kwargs )
		self.trainer = ModelTrainer( self.config )

	def get_result_view(self, tset: TSet, **kwargs ):
		self.plot = ResultPlot(self.trainer, tset, **kwargs)
		return self.plot.plot()

	def get_training_view(self, **kwargs):
		self.plot = TrainingPlot(self.trainer, **kwargs)
		return self.plot.plot()

	def test(self, model: str, test_name: str, **kwargs):
		with ConfigContext(self.cname, model=model, **kwargs) as cc:
			self.config = cc
			self.trainer = ModelTrainer(cc)
			if test_name == "load_raw_dataset":
				self.trainer.model_manager.get_raw_dataset()

