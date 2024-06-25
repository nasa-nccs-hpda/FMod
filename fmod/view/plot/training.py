import matplotlib.pyplot as plt
from fmod.view.plot.base import Plot
from fmod.base.io.loader import ncFormat, TSet
from fmod.controller.dual_trainer import ModelTrainer


class TrainingPlot(Plot):


	def __init__(self, trainer: ModelTrainer, **kwargs ):
		super(TrainingPlot,self).__init__(trainer, **kwargs)
		self.fmt = {TSet.Train: 'b', TSet.Validation: 'g'}
		self.trainer.results_accum.load_results()
		self.min_loss = {}


	def plot(self):
		(x, y) = self.trainer.results_accum.get_plot_data()
		for tset in [TSet.Train, TSet.Validation]:
			xp, yp = x[tset], y[tset]
			self.min_loss[tset] = yp.min() if (yp.size > 0) else 0.0
			ax.plot(xp, yp, fmt[tset], label=tset.name)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss")
		ax.set_yscale(self.yscale)
		ax.set_title(f"Model '{self.model}':  Validation Loss = {min_loss[TSet.Validation]:.4f} ")
		ax.legend()
		fig.canvas.draw_idle()
		return fig.canvas