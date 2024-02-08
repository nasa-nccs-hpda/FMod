from typing  import List, Tuple, Union, Optional, Dict, Callable
import ipywidgets as ipw, traitlets as tl
from fmod.base.util.logging import lgm, exception_handled, log_timing
from ipywidgets import GridspecLayout

class Counter(ipw.DOMWidget):
    value = tl.CInt(0, sync=True)

class StepSlider:

	def __init__(self, label: str, maxval: int, callback: Callable[[int], int], **kwargs):
		self.bsize = kwargs.get('bsize','30px')
		self.ssize = kwargs.get('ssize', '920px')
		self.maxval = maxval
		self.executable: Callable[[int], int] = callback
		self.counter = Counter()
		self.slider: ipw.IntSlider = ipw.IntSlider(value=0, min=0, max=maxval, description=label, layout=ipw.Layout(width=self.ssize, height=self.bsize) )
		self.slider.observe(self.update, names='value')
		self.button_cback    = ipw.Button(description='<', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cforward = ipw.Button(description='>', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		tl.link((self.slider, 'value'), (self.counter, 'value'))
		self.button_cback.on_click(self.bminus)
		self.button_cforward.on_click(self.bplus)

	@exception_handled
	def bplus(self,b):
		self.counter.value = ( self.counter.value + 1 ) % (self.maxval+1)

	@exception_handled
	def bminus(self,b):
		self.counter.value = ( self.counter.value - 1 ) % (self.maxval+1)

	@exception_handled
	def update(self, change):
		self.executable( change['new'] )

	def gui(self):
		return ipw.HBox([self.slider, self.button_cback, self.button_cforward])
