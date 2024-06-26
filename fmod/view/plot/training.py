import matplotlib.pyplot as plt
from fmod.view.plot.base import Plot
import ipywidgets as ipw
from fmod.base.io.loader import ncFormat, TSet
from fmod.controller.dual_trainer import ModelTrainer

class TrainingPlot(Plot):

	def __init__(self, trainer: ModelTrainer, **kwargs ):
		super(TrainingPlot,self).__init__(trainer, **kwargs)
		self.fmt = {TSet.Train: 'b', TSet.Validation: 'g'}
		self.trainer.results_accum.load_results()
		self.min_loss = {}
		self.create_figure( title='Training Loss' )

	def plot(self) -> ipw.Box:
		(x, y) = self.trainer.results_accum.get_plot_data()
		for tset in [TSet.Train, TSet.Validation]:
			xp, yp = x[tset], y[tset]
			self.min_loss[tset] = yp.min() if (yp.size > 0) else 0.0
			self.axs.plot(xp, yp, self.fmt[tset], label=tset.name)
		self.axs.set_xlabel("Epoch")
		self.axs.set_ylabel("Loss")
		self.axs.set_yscale(self.yscale)
		self.axs.set_title(f"Model '{self.model}':  Validation Loss = {self.min_loss[TSet.Validation]:.4f} ")
		self.axs.legend()
		# self.fig.canvas.draw_idle()
		return ipw.VBox( [self.fig.canvas] )