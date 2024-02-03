import torch
import torch_harmonics as harmonics
import numpy as np

class GridOps:

	def __init__(self, nlat, nlon, grid='equiangular', radius=6.37122E6 ):
		super().__init__()
		self.nlat = nlat
		self.nlon = nlon
		self.grid = grid
		self.radius = radius

		# compute gridpoints
		if self.grid == "legendre-gauss":
			cost, quad_weights = harmonics.quadrature.legendre_gauss_weights(self.nlat, -1, 1)
		elif self.grid == "lobatto":
			cost, quad_weights = harmonics.quadrature.lobatto_weights(self.nlat, -1, 1)
		elif self.grid == "equiangular":
			cost, quad_weights = harmonics.quadrature.clenshaw_curtiss_weights(self.nlat, -1, 1)
		else:
			raise Exception( f"Unknown grid: {self.grid}")

		self.quad_weights = torch.as_tensor(quad_weights).reshape(-1, 1).cuda()

		self.lats = -torch.as_tensor(np.arcsin(cost))
		self.lons = torch.linspace(0, 2 * np.pi, self.nlon + 1, dtype=torch.float64)[:nlon]


	def integrate_grid(self, ugrid, dimensionless=False, polar_opt=0):
		dlon = 2 * torch.pi / self.nlon
		radius = 1 if dimensionless else self.radius
		if polar_opt > 0:
			out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * self.quad_weights[polar_opt:-polar_opt] * dlon * radius ** 2, dim=(-2, -1))
		else:
			out = torch.sum(ugrid * self.quad_weights * dlon * radius ** 2, dim=(-2, -1))
		return out

	def plot_griddata(self, data, fig, cmap='twilight_shifted', vmax=None, vmin=None, projection='3d', title=None, antialiased=False):
		"""
		plotting routine for data on the grid. Requires cartopy for 3d plots.
		"""
		import matplotlib.pyplot as plt

		lons = self.lons.squeeze() - torch.pi
		lats = self.lats.squeeze()

		if data.is_cuda:
			data = data.cpu()
			lons = lons.cpu()
			lats = lats.cpu()

		Lons, Lats = np.meshgrid(lons, lats)

		if projection == 'mollweide':
			# ax = plt.gca(projection=projection)
			ax = fig.add_subplot(projection=projection)
			im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, vmax=vmax, vmin=vmin)
			# ax.set_title("Elevation map of mars")
			ax.grid(True)
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			plt.colorbar(im, orientation='horizontal')
			plt.title(title)

		elif projection == '3d':
			import cartopy.crs as ccrs

			proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)

			# ax = plt.gca(projection=proj, frameon=True)
			ax = fig.add_subplot(projection=proj)
			Lons = Lons * 180 / np.pi
			Lats = Lats * 180 / np.pi

			# contour data over the map.
			im = ax.pcolormesh(Lons, Lats, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=antialiased, vmax=vmax, vmin=vmin)
			plt.title(title, y=1.05)

		else:
			raise NotImplementedError

		return im

