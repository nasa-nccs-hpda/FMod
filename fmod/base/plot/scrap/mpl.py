import math, xarray, matplotlib, datetime
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def plot_data( data: dict[str, xarray.Dataset], fig_title: str, **kwargs ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
  plot_size: float = kwargs.get('plot_size',5)
  robust: bool = kwargs.get('robust',False)
  cols: int = kwargs.get('cols',4)

  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols,
                               plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()

  images = []
  for i, (title, (pdata, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    extend = ("both" if robust else "neither")
    im = ax.imshow( pdata.isel(time=0, missing_dims="ignore"), norm=norm, origin="lower", cmap=cmap)
    plt.colorbar( mappable=im, ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.75, cmap=cmap, extend=extend)
    images.append(im)

  def update(frame):
    if "time" in first_data.dims:
      td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for image, (idata, norm, cmap) in zip(images, data.values()):
      image.set_data(idata.isel(time=frame, missing_dims="ignore"))

  ani = animation.FuncAnimation( fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())