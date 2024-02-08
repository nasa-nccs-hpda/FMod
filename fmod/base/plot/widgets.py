from typing  import List, Tuple, Union, Optional, Dict, Callable
import ipywidgets as ipw, traitlets as tl
from fmod.base.util.logging import lgm, exception_handled, log_timing
from ipywidgets import GridspecLayout

class Counter(ipw.DOMWidget):
	value = tl.CInt(0, sync=True)

	def __init__(self, nval: int ):
		super(Counter,self).__init__()
		self.nval = nval

	def plus(self, *args):
		self.value = (self.value + 1) % self.nval

	def minus(self, *args):
		self.value = (self.value - 1) % self.nval

class StepSlider:

	def __init__(self, label: str, nval: int, callback: Callable[[int], int], **kwargs):
		self.bsize = kwargs.get('bsize','30px')
		self.ssize = kwargs.get('ssize', '920px')
		self.executable: Callable[[int], int] = callback
		self.counter = Counter(nval)
		self.slider: ipw.IntSlider = ipw.IntSlider(value=0, min=0, max=nval-1, description=label, layout=ipw.Layout(width=self.ssize, height=self.bsize) )
		self.slider.observe(self.update, names='value')
		self.button_cback    = ipw.Button(description='<', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cforward = ipw.Button(description='>', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		tl.link((self.slider, 'value'), (self.counter, 'value'))
		self.button_cback.on_click(self.counter.minus)
		self.button_cforward.on_click(self.counter.plus)

	@exception_handled
	def update(self, change):
		self.executable( change['new'] )

	def gui(self):
		return ipw.HBox([self.slider, self.button_cback, self.button_cforward])
