import xarray as xa, numpy as np
from typing import List, Union, Tuple, Optional, Dict, Type
from matplotlib.image import AxesImage
from matplotlib.widgets import Slider, Button
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

class ImageBrowser:

    def __init__(self, label: str, ax: Axes, images: List[xa.DataArray], plot_args: Dict, **kwargs ):
        self.ax: Axes = ax
        self.label = label
        self.name = plot_args.pop('name','')
        self.images: List[xa.DataArray] = images
        self.overlay_plots: List[Tuple[str,AxesImage]] = []
        self.overlay_index = 0
        self.build( plot_args )

    def norm(self, x: np.ndarray ) -> np.ndarray:
        xmin, xmax =  np.nanmin(x), np.nanmax(x)
        return (x-xmin)/(xmax-xmin)

    def build(self, plot_args: Dict):
        cmap = plot_args.pop('cmap', "jet")
        overlay_args = dict( overlays=       plot_args.pop( 'overlays', {} ),
                             overlay_alpha = plot_args.pop( 'overlay_alpha', 0.5 ) )
        idata = self.image_data(0)
        self.plot: AxesImage = self.ax.imshow( idata, cmap=cmap, origin="lower", **plot_args )
        self.build_slider()
        self.build_overlay( **overlay_args )

    def build_slider(self):
        self.ax.figure.subplots_adjust( bottom=0.25 )
        sax = self.ax.figure.add_axes([0.1, 0.1, 0.65, 0.03])
        svm = len( self.images )-1
        self.slider = Slider( ax=sax, label=self.label, valmin=0, valmax=svm, valinit=0, valstep=1, dragging=True )
        self.slider.on_changed(self.update)

    def build_overlay(self, **kwargs ):
        self.overlays: Dict[str,xa.DataArray] = kwargs.get('overlays', {})
        bax = self.ax.figure.add_axes([0.8, 0.1, 0.15, 0.03])
        self.overlay_alpha = kwargs.get('overlay_alpha',0.7)
        for oname,overlay in self.overlays.items():
            overlay_data = self.norm( overlay[0].values )
            plot = self.ax.imshow(overlay_data, cmap='binary', origin="lower", alpha=0.0)
            self.overlay_plots.append( (oname,plot) )
        self.ax.set_title(self.name)
        self.overlay_button = Button( bax, 'Overlay', hovercolor='0.975' )
        self.overlay_button.on_clicked(self.toggle_overlay)

    def toggle_overlay( self, *args ):
        omod = len(self.overlay_plots) + 1
        print(f'toggle_overlay {omod}')
        self.overlay_index = (self.overlay_index + 1) % omod
        for idx, ( oname, overlay_plot ) in enumerate(self.overlay_plots):
            overlay_plot.set_alpha( self.overlay_alpha if (idx+1==self.overlay_index) else 0.0 )
        print(f'set_alpha: {self.overlay_index}')
        self.ax.set_title( self.name if self.overlay_index == 0 else self.overlay_plots[self.overlay_index-1][0] )
        print('set_title')
        self.ax.figure.canvas.draw_idle()

    def image_data(self, step: int ) -> np.ndarray:
        return self.norm( self.images[step].values )

    def update(self, step ):
        try:
            self.plot.set_data( self.image_data(step) )
            # for ixd,cmap,overlays in enumerate(self.overlays.items()):
            #     if len(overlays) > 1:
            #         self.overlay_plot[1].set_data( overlays[step] )
            self.ax.figure.canvas.draw_idle()
        except Exception as err:
            print( err )


