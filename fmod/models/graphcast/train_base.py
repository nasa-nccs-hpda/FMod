import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast
from fmod.base.util.config import cfg
from fmod.models.graphcast.constants import get_constants
from fmod.models.graphcast.constants import Constants

class BaseTrainer:

    def __init__(self):
        self.self.C: Constants = get_constants()

    def rollout(self, grid_nfeat, y):
        with autocast(enabled=self.self.C.amp, dtype=self.amp_dtype):
            total_loss = 0
            pred_prev = grid_nfeat
            for i in range(y.size(dim=1)):
                # Shape of y is [N, M, self.C, H, W]. M is the number of steps
                pred = self.model(pred_prev)
                loss = self.criterion(pred, y[:, i])
                total_loss += loss
                pred_prev = pred
            return total_loss

    def forward(self, grid_nfeat, y):
        torch.cuda.nvtx.range_push("Loss computation")
        if self.C.pyt_profiler:
            with profile( activities=[ProfilerActivity.CUDA], record_shapes=True ) as prof:
                with record_function("training_step"):
                    loss = self.rollout(grid_nfeat, y)
            print(  prof.key_averages(group_by_input_shape=True).table( sort_by="cuda_time_total", row_limit=10 ) )
            exit(0)
        else:
            loss = self.rollout(grid_nfeat, y)
        torch.cuda.nvtx.range_pop()
        return loss

    def backward(self, loss):
        # backward pass
        torch.cuda.nvtx.range_push("Weight gradients")
        if self.C.amp:
            self.scaler.scale(loss).backward()
            torch.cuda.nvtx.range_pop()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.C.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.C.grad_clip_norm)
            torch.cuda.nvtx.range_pop()
            self.optimizer.step()

    def train(self, grid_nfeat, y):
        self.optimizer.zero_grad()
        loss = self.forward(grid_nfeat, y)
        self.backward(loss)
        self.scheduler.step()
        return loss