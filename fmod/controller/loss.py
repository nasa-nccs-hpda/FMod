import torch, math
from fmod.base.util.config import cfg
from fmod.base.util.grid import GridOps
import torch_harmonics as harmonics
from fmod.base.io.loader import BaseDataset
from controller.SCRAP.trainer import TaskType

class SphericalLoss(object):

	def __init__(self, dataset: BaseDataset, device: torch.device):
		self.dataset = dataset
		self.device = device
		self.scale_factor = cfg().model.get('scale_factor', 1)
		self.task_type: TaskType = TaskType(cfg().task.task_type)
		inp, tar = next(iter(dataset))
		self.data_iter = iter(dataset)
		if self.task_type == TaskType.Downscale:
			self.grid_shape = tar.shape[-2:]
			lmax = inp.shape[-2]
		else:
			self.grid_shape = inp.shape[-2:]
			lmax = math.ceil(self.grid_shape[0] / cfg().model.get('scale_factor', 1))
		self.gridops = GridOps(*self.grid_shape, self.device)
		self.sht = harmonics.RealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)
		self.isht = harmonics.InverseRealSHT( *self.grid_shape, lmax=lmax, mmax=lmax, grid='equiangular', csphase=False)


	def l2loss_sphere(self, prd, tar, relative=False, squared=True):
		loss = self.gridops.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
		if relative:
			loss = loss / self.gridops.integrate_grid(tar ** 2, dimensionless=True).sum(dim=-1)

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()

		return loss

	def spectral_l2loss_sphere(self, prd, tar, relative=False, squared=True):
		# compute coefficients
		coeffs = torch.view_as_real(self.sht(prd - tar))
		coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
		norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
		loss = torch.sum(norm2, dim=(-1, -2))

		if relative:
			tar_coeffs = torch.view_as_real(self.sht(tar))
			tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
			tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
			tar_norm2 = torch.sum(tar_norm2, dim=(-1, -2))
			loss = loss / tar_norm2

		if not squared:
			loss = torch.sqrt(loss)
		loss = loss.mean()
		return loss