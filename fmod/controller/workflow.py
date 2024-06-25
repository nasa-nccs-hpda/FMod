import torch, time
from fmod.base.util.config import ConfigContext, cfg
from fmod.controller.dual_trainer import ModelTrainer
from fmod.base.io.loader import ncFormat, TSet
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from fmod.base.plot.sres import SRPlot

class WorkflowController(object):

	def __init__(self, cname: str, configuration: Dict[str,Any], **kwargs):
		self.cname = cname
		self.seed = kwargs.get('seed', int( time.time()/60 ) )
		self.refresh_state = kwargs.get('refresh_state', False )
		self.config: ConfigContext = None
		self.trainer: ModelTrainer = None
		self.plot: Any = None
		self.model = None
		ConfigContext.set_defaults( **configuration )

	def train(self, models: List[str], **kwargs):
		for model in models:
			with ConfigContext(self.cname, model=model, **kwargs) as cc:
				self.config = cc
				self.trainer = ModelTrainer(cc)
				self.trainer.train(refresh_state=self.refresh_state, seed=self.seed)

	def init_plotting(self, cname, model, **kwargs ):
		self.model = model
		self.config = ConfigContext.activate_global( cname, model=model, **kwargs )
		self.trainer = ModelTrainer( self.config )

	def get_result_view(self, tset: TSet, **kwargs ):
		self.plot = SRPlot(self.trainer, tset, **kwargs )
		return self.plot.plot()

	def get_training_view(self, **kwargs):
		fsize = kwargs.get( 'fsize', 8.0 )
		yscale = kwargs.get('yscale', 'log')
		with plt.ioff():
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[fsize * 2, fsize], layout="tight")
		fig.suptitle('SRes Loss Over Training Epochs', fontsize=14, va="top", y=1.0)
		fmt = {TSet.Train: 'b', TSet.Validation: 'g'}
		self.trainer.results_accum.load_results()
		(x, y), min_loss = self.trainer.results_accum.get_plot_data(), {}
		for tset in [TSet.Train, TSet.Validation]:
			xp, yp = x[tset], y[tset]
			min_loss[tset] = yp.min() if (yp.size > 0) else 0.0
			ax.plot(xp, yp, fmt[tset], label=tset.name)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss")
		ax.set_yscale(yscale)
		ax.set_title(f"Model '{self.model}':  Validation Loss = {min_loss[TSet.Validation]:.4f} ")
		ax.legend()
		fig.canvas.draw_idle()
		return fig.canvas

