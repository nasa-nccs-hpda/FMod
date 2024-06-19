from fmod.base.util.config import cfg
import torch

def set_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
	if torch.cuda.is_available():   torch.cuda.set_device(device.index)
	else:                           assert gpu_index == 0, "Can't run on multiple GPUs: No GPUs available"
	return device


def get_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
	return device