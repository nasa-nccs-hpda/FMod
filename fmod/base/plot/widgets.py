from typing  import List, Tuple, Union, Optional, Dict, Callable
import ipywidgets as ipw, traitlets as tl
from fmod.base.util.logging import lgm, exception_handled, log_timing
from ipywidgets import GridspecLayout

class Counter(ipw.DOMWidget):
	value = tl.CInt(0, sync=True)

	def __init__(self, nval: int, **kwags ):
		super(Counter,self).__init__()
		self.nval = nval
		for source in kwags.get("links", [] ):
			self.link(source)

	def increment(self, *args):
		self.value = (self.value + 1) % self.nval

	def decrement(self, *args):
		self.value = (self.value - 1) % self.nval

	def link(self, source: ipw.Widget, attr: str = 'value') -> ipw.Widget:
		tl.link( (source, attr), (self, 'value'), )
		return self

class StepSlider(ipw.HBox):

	def __init__(self, label: str, nval: int, callback: Callable = None, **kwargs):
		self.bsize = kwargs.get('bsize','30px')
		self.ssize = kwargs.get('ssize', '920px')
		self.executable: Callable[[int], int] = callback
		self.slider: ipw.IntSlider = ipw.IntSlider(value=0, min=0, max=nval-1, description=label, layout=ipw.Layout(width=self.ssize, height=self.bsize) )
		self.slider.observe( self.update, names='value' )
		self.counter = Counter( nval, links=[self.slider] )
		self.button_cback    = ipw.Button(description='<', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cforward = ipw.Button(description='>', button_style='info', layout=ipw.Layout(width=self.bsize, height=self.bsize) )
		self.button_cback.on_click(    self.counter.decrement )
		self.button_cforward.on_click( self.counter.increment )
		super(StepSlider,self).__init__( [self.slider, self.button_cback, self.button_cforward] )

	def set_callback(self, callback: Callable):
		self.executable = callback

	@property
	def value(self):
		return self.slider.value

	@exception_handled
	def update(self, change):
		self.executable( change['new'] )

