import math, random, numpy as np
from typing import Dict, Tuple, List, Optional
from fmod.base.io.loader import TSet, batchDomain
from fmod.base.util.config import cfg
from fmod.base.util.logging import lgm, log_timing

class TileIterator(object):

    def __init__(self, **kwargs ):
        self.randomize: bool = kwargs.get('randomize', False)
        self._batch_losses = {}
        self.index: int = 0
        self.next_index = 0

    def batch_losses(self, ltype) -> List[float]:
        return self._batch_losses.setdefault(ltype, [])

    def register_loss(self, ltype: str, loss: float ):
        self.batch_losses(ltype).append( loss )

    def epoch_loss(self, ltype: str):
        epoch_loss = np.array(self.batch_losses(ltype)).mean()
        return epoch_loss

    def __iter__(self):
        raise NotImplementedError("TileIterator:__iter__")

    @property
    def active(self):
        raise NotImplementedError("TileIterator:active")


    def __next__(self) ->  Dict[str,int]:
        raise NotImplementedError("TileIterator:__next__")

    @classmethod
    def get_iterator(cls, **kwargs ):
        domain: batchDomain = batchDomain.from_config(cfg().task.get('batch_domain', 'tiles'))
        if domain == batchDomain.Tiles: return TileBatchIterator(**kwargs)
        if domain == batchDomain.Time:  return TileGridIterator(**kwargs)


class TileBatchIterator(TileIterator):

    def __init__(self, **kwargs ):
        super(TileBatchIterator, self).__init__(**kwargs)
        self.batch_size: int = cfg().task.batch_size
        self.ntiles: int = kwargs.get('ntiles',0)
        assert self.ntiles > 0, "Must provide ntiles for TileBatchIterator"
        self.batch_start_idxs: List[int] = list(range(0,self.ntiles,self.batch_size))
        if self.randomize: random.shuffle( self.batch_start_idxs )


    def __iter__(self):
        self.next_index = 0
        return self

    @property
    def active(self):
        return (self.ntiles == 0) or (self.next_index < len(self.batch_start_idxs))


    def __next__(self) ->  Dict[str,int]:
        if not self.active: raise StopIteration()
        self.index = self.next_index
        bstart = self.batch_start_idxs[self.index]
        result = dict( start=bstart, end=bstart + self.batch_size )
        self.next_index = self.index + 1
        return result

class TileGridIterator(TileIterator):

    def __init__(self,  **kwargs ):
        super(TileGridIterator, self).__init__(**kwargs)
        self.grid = TileGrid()
        self.regular_grid: List[  Dict[str,int]  ] = list( self.grid.get_tile_locations(**kwargs).values() )


    def __iter__(self):
        if self.randomize: random.shuffle( self.regular_grid )
        self.next_index = 0
        return self

    @property
    def active(self):
        return self.next_index < len(self.regular_grid)

    def __next__(self) ->  Dict[str,int]:
        if not self.active: raise StopIteration()
        self.index = self.next_index
        result = self.regular_grid[self.index]
        self.next_index = self.index + 1
        return result

class TileGrid(object):

    def __init__(self):
        self.origin: Dict[str,int] = cfg().task.get('origin',{})
        self.tile_grid: Dict[str, int] = None
        self.tile_size: Dict[str,int] = cfg().task.tile_size
        self.tlocs: Dict[Tuple[int,int],Dict[str,int]] = {}
        upsample_factors: List[int] = cfg().model.downscale_factors
        self.upsample_factor = math.prod(upsample_factors)

    def get_global_grid_shape(self, **kwargs ) -> Dict[str,int]:
        image_shape: Optional[Dict[str, int]] = kwargs.get( 'image_shape', cfg().task.get('image_shape', None ) )
        if image_shape is None: return dict(x=1,y=1)
        ts = self.get_full_tile_size()
        global_shape = {dim: image_shape[dim] // ts[dim] for dim in ['x', 'y']}
        return global_shape

    def get_grid_shape(self, **kwargs) -> Dict[str, int]:
        global_grid_shape = self.get_global_grid_shape(**kwargs)
        cfg_grid_shape = cfg().task.tile_grid
        self.tile_grid = { dim: (cfg_grid_shape[dim] if (cfg_grid_shape[dim]>=0) else global_grid_shape[dim]) for dim in ['x', 'y'] }
        return self.tile_grid

    def get_active_region(self, **kwargs ) -> Dict[str, Tuple[int,int]]:
        ts = self.get_full_tile_size()
        gs = self.get_grid_shape( **kwargs )
        region = { d: (self.origin[d],self.origin[d]+ts[d]*gs[d]) for d in ['x', 'y'] }
        return region

    def get_tile_size(self, highres: bool = False ) -> Dict[str, int]:
        sf = self.upsample_factor if highres else 1
        rv = { d: self.tile_size[d] * sf for d in ['x', 'y'] }
        return  rv

    def get_full_tile_size(self) -> Dict[str, int]:
        return { d: self.tile_size[d] * self.upsample_factor for d in ['x', 'y']}

    def get_tile_origin( self, ix: int, iy: int, highres: bool = False ) -> Dict[str, int]:
        sf = self.upsample_factor if highres else 1
        return { d: self.origin[d] + self.cdim(ix, iy, d) * self.tile_size[d] * sf for d in ['x', 'y'] }

    def get_tile_locations(self, **kwargs ) -> Dict[ Tuple[int,int], Dict[str,int] ]:
        highres: bool = kwargs.get('highres', False)
        selected_tile: Optional[Tuple[int,int]] = kwargs.get('selected_tile', None)
        if len(self.tlocs) == 0:
            if self.tile_grid is None:
                self.get_grid_shape( **kwargs )
            for ix in range(self.tile_grid['x']):
                for iy in range(self.tile_grid['y']):
                    if (selected_tile is None) or ((ix,iy) == selected_tile):
                        self.tlocs[(ix,iy)] = self.get_tile_origin(ix,iy,highres)
        return self.tlocs



    @classmethod
    def cdim(cls, ix: int, iy: int, dim: str) -> int:
        if dim == 'x': return ix
        if dim == 'y': return iy
