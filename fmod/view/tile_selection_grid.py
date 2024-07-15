import matplotlib.pyplot as plt
import torch, math
import xarray, traceback, random
from datetime import datetime
from torch import Tensor
from typing import Any, Dict, List, Tuple, Union, Sequence, Callable, Optional
from fmod.base.util.config import cdelta, cfg, cval, get_data_coords
from fmod.base.util.array import array2tensor
from fmod.data.tiles import TileGrid
from fmod.base.util.logging import lgm, exception_handled
from fmod.base.util.ops import pctnan, pctnant
from omegaconf import DictConfig, OmegaConf
from enum import Enum
import numpy as np, xarray as xa
import torch.nn as nn
import time
from fmod.base.io.loader import TSet, srRes
from matplotlib.collections import PatchCollection
from matplotlib.patches import  Rectangle, Patch

def default_selection_callabck( tilerec: Dict[str,float]):
	print( f" **** Actor-based Tile selection: {tilerec} ****")

def r2str( r: Rectangle ) -> str:
	return f"({r.get_x()},{r.get_y()})x({r.get_width()},{r.get_height()})"

def onpick_test(event):
	lgm().log( f" **** Actor-based Tile selection: {event} ****", display=True)

class TileSelectionGrid(object):

	def __init__(self, sample_data: Optional[xa.DataArray] = None ):
		print( f"TileSelectionGrid: {sample_data} ****")
		self.sample_data: Optional[xa.DataArray] = sample_data
		self.tile_grid: TileGrid = TileGrid()
		self.tiles: Dict[Tuple[int, int], Rectangle] = None
		self._selection_callback = default_selection_callabck

	def get_tile_coords(self, tile_index: int) -> Tuple[int, int]:
		tile_keys = list(self.tiles.keys())
		print( f" **** get_tile_coords: {tile_index}, tile_keys: {tile_keys} ****")
		return tile_keys[tile_index]

	@property
	def ntiles(self):
		return len(self.tiles)

	def get_selected(self, x: float, y: float ) -> Optional[Tuple[int,int]]:
		for xyi, r in self.tiles.items():
			if r.contains_point( (x,y) ):
				return xyi
	def create_tile_recs(self, **kwargs):
		refresh = kwargs.get('refresh', False)
		highres = kwargs.get('highres', True)
		ts: Dict[str, int] = self.tile_grid.get_tile_size(highres)
		if (self.tiles is None) or refresh:
			self.tiles = {}
			ishape: Dict[str, int]  = { cn: self.sample_data.sizes[cn] for cn in ['x','y'] }
			tile_locs: Dict[Tuple[int, int], Dict[str, int]] = self.tile_grid.get_tile_locations( image_shape=ishape, highres=highres )
			for xyi, tloc in tile_locs.items():
				xy = (tloc['x'], tloc['y'])
				r = Rectangle(xy, ts['x'], ts['y'], fill=False, picker=True, linewidth=kwargs.get('lw', 1), edgecolor=kwargs.get('color', 'white'))
				self.tiles[xyi] = r
		print( f" **** create_tile_recs: tiles={list(self.tiles.keys())} ****")

	def set_selection_callabck(self, selection_callabck: Callable):
		self._selection_callback = selection_callabck

	def onpick(self,event):
		lgm().log( f" **** Tile selection: {event} ****")
		rect: Rectangle = event.artist
		coords = dict(x=rect.get_x(), y=rect.get_y())
		lgm().log(f" ----> Coords: {coords}", display=True)
		self._selection_callback( coords )

	def overlay_grid(self, ax: plt.Axes, **kwargs):
		self.create_tile_recs(**kwargs)
		p = PatchCollection( self.tiles.values(), match_original=True )
		ax.add_collection(p)
		ax.figure.canvas.mpl_connect('pick_event', self.onpick )