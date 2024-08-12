from fmod.base.util.config import cfg
from fmod.base.util.logging import lgm, exception_handled, log_timing
from torch import cuda
import torch, os, time

def set_device() -> torch.device:
        gpu_index = cfg().pipeline.gpu
        print('torch.cuda.is_available(?): ' + str(torch.cuda.is_available()))
        device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
        print('<device>:<index>:<device.index> ' + str(device) + ':' + str(gpu_index) + ':' + str(device.index))
        # if torch.cuda.is_available():   torch.cuda.set_device(device.index)
        # else:                           assert gpu_index == 0, "Can't run on multiple GPUs: No GPUs available"
        if torch.cuda.is_available():  device = "cuda:0" 
        print('<device>:<index>:<device.index> ' + str(device) + ':' + str(gpu_index) + ':' + str(device.index))
        return device

def _set_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if cuda.is_available() else 'cpu')
	if cuda.is_available():
		cuda.set_device(device.index)
		if cfg().pipeline.memory_debug:
			cuda.memory._record_memory_history()
	else:
		assert gpu_index == 0, "Can't run on multiple GPUs: No GPUs available"
	return device


def get_device() -> torch.device:
	gpu_index = cfg().pipeline.gpu
	device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
	return device

def memory_snapshot_path( ) -> str:
	cpath = f"{cfg().platform.results}/memory/snapshot.{cfg().task.training_version}.pkl"
	os.makedirs(os.path.dirname(cpath), 0o777, exist_ok=True)
	return cpath

def save_memory_snapshot():
	if cfg().pipeline.memory_debug:
		t0 = time.time()
		mspath = memory_snapshot_path()
		cuda.memory._dump_snapshot( mspath )
		lgm().log(f" *** SAVE memory snapshot to {mspath}, dt={time.time()-t0:.4f} sec", display=True)
