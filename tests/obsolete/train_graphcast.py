import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import time, hydra
import wandb as wb
import torch.cuda.profiler as profiler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
import os
from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.utils.graphcast.loss import CellAreaWeightedLossFunction
from modulus.launch.logging import PythonLogger, initialize_wandb, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from data.merra2 import MERRA2NCDatapipe
from fmod.model.graphcast.train_utils import count_trainable_params
from fmod.model.graphcast.loss.utils import grid_cell_area
from fmod.model.graphcast.train_base import BaseTrainer
from fmod.model.graphcast.validation import Validation
from modulus.distributed import DistributedManager
try: import apex
except: pass

class GraphCastTrainer(BaseTrainer):

    def __init__(self, wb, dist, rank_zero_logger):
        super().__init__()
        self.dist = dist
        self.dtype = torch.bfloat16 if self.C.full_bf16 else torch.float32
        self.enable_scaler = False
        self.amp_dtype = None

        if self.C.full_bf16:
            assert torch.cuda.is_bf16_supported()
            rank_zero_logger.info(f"Using {str(self.dtype)} dtype")
            if self.C.amp:
                raise ValueError("Full bfloat16 training is enabled, switch off self.C.amp")

        if self.C.amp:
            rank_zero_logger.info(f"Using self.C.amp with dtype {self.C.amp_dtype}")
            if self.C.amp_dtype == "float16" or self.C.amp_dtype == "fp16":
                self.self.C.amp_dtype = torch.float16
                self.enable_scaler = True
            elif self.C.amp_dtype == "bfloat16" or self.C.amp_dtype == "bf16":
                self.self.C.amp_dtype = torch.bfloat16
            else:
                raise ValueError("Invalid dtype for self.C.amp")

        # instantiate the model
        self.model = GraphCastNet(
            meshgraph_path=self.C.icospheres_path,
            static_dataset_path=self.C.static_dataset_path,
            input_dim_grid_nodes=self.C.num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=self.C.num_channels,
            processor_layers=self.C.processor_layers,
            hidden_dim=self.C.hidden_dim,
            do_concat_trick=self.C.concat_trick,
            use_cugraphops_encoder=self.C.cugraphops_encoder,
            use_cugraphops_processor=self.C.cugraphops_processor,
            use_cugraphops_decoder=self.C.cugraphops_decoder,
            recompute_activation=self.C.recompute_activation,
        )

        # set gradient checkpointing
        if self.C.force_single_checkpoint:
            self.model.set_checkpoint_model(True)
        if self.C.checkpoint_encoder:
            self.model.set_checkpoint_encoder(True)
        if self.C.checkpoint_processor:
            self.model.set_checkpoint_processor(self.C.segments)
        if self.C.checkpoint_decoder:
            self.model.set_checkpoint_decoder(True)

        # JIT compile the model, and specify the device and dtype
        if self.C.jit:
            torch.jit.script(self.model).to(dtype=self.dtype).to(device=dist.device)
            rank_zero_logger.success("JIT compiled the model")
        else:
            self.model = self.model.to(dtype=self.dtype).to(device=dist.device)
        if self.C.watch_model and not self.C.jit and dist.rank == 0:
            wb.watch(self.model)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        rank_zero_logger.info( f"Model parameter count is {count_trainable_params(self.model)}")

        # instantiate the training datapipe
        self.datapipe = MERRA2NCDatapipe( device=dist.device, process_rank=dist.rank, world_size=dist.world_size )
        rank_zero_logger.success(f"Loaded training datapipe of size {len(self.datapipe)}")

        # instantiate the validation
        if dist.rank == 0:
            self.validation = Validation(self.model, self.dtype, self.dist, wb)

        # enable train mode
        self.model.train()

        # get area
        if hasattr(self.model, "module"):
            self.area = grid_cell_area( self.model.module.lat_lon_grid[:, :, 0], unit="deg" )
        else:
            self.area = grid_cell_area(self.model.lat_lon_grid[:, :, 0], unit="deg")
        self.area = self.area.to(dtype=self.dtype).to(device=dist.device)

        # instantiate loss, optimizer, and scheduler
        self.criterion = CellAreaWeightedLossFunction(self.area)
        try:
            self.optimizer = apex.optimizers.FusedAdam( self.model.parameters(), lr=self.C.lr, betas=(0.9, 0.95), weight_decay=0.1 )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.C.lr)

        scheduler1 = LinearLR( self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.C.num_iters_step1 )
        scheduler2 = CosineAnnealingLR( self.optimizer, T_max=self.C.num_iters_step2, eta_min=0.0 )
        scheduler3 = LambdaLR( self.optimizer, lr_lambda=lambda epoch: (self.C.lr_step3 / self.C.lr) )

        self.scheduler = SequentialLR( self.optimizer, schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[self.C.num_iters_step1, self.C.num_iters_step1 + self.C.num_iters_step2] )
        self.scaler = GradScaler(enabled=self.enable_scaler)

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.iter_init = load_checkpoint( os.path.join(self.C.ckpt_path, self.C.ckpt_name),  models=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler, device=dist.device )


if __name__ == "__main__":
    hydra.initialize(version_base=None, config_path="../../config")
    configure('merra2-finetuning')
    DistributedManager.initialize()
    dist = DistributedManager()

    initialize_wandb( project="Modulus-Launch", entity="Modulus", name="GraphCast-Training", group="GraphCast-DDP-Group" )
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    # initialize trainer
    trainer = GraphCastTrainer(wb, dist, rank_zero_logger)
    if dist.rank == 0:
        os.makedirs(trainer.C.ckpt_path, exist_ok=True)
        with open(os.path.join(trainer.C.ckpt_path, trainer.C.ckpt_name.replace(".pt", ".json")), "w") as json_file:
            json_file.write(trainer.C.model_dump_json(indent=4))
    start = time.time()
    rank_zero_logger.info("Training started...")
    loss_agg, iter, tagged_iter, num_rollout_steps = 0, trainer.iter_init, 1, 1
    terminate_training, finetune, update_dataloader = False, False, False

    with torch.autograd.profiler.emit_nvtx() if trainer.C.profile else nullcontext():
        # training loop
        while True:
            assert ( iter < trainer.C.num_iters_step1 + trainer.C.num_iters_step2 + trainer.C.num_iters_step3 ), "Training is already finished!"
            for i, data in enumerate(trainer.datapipe):
                if trainer.C.profile and iter == trainer.C.profile_range[0]:
                    rank_zero_logger.info("Starting profile", "green")
                    profiler.start()
                if trainer.C.profile and iter == trainer.C.profile_range[1]:
                    rank_zero_logger.info("Ending profile", "green")
                    profiler.stop()
                torch.cuda.nvtx.range_push("Training iteration")

                if iter >= trainer.C.num_iters_step1 + trainer.C.num_iters_step2 and not finetune:
                    finetune = True
                    if trainer.C.force_single_checkpoint_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_model(True)
                        else:
                            trainer.model.set_checkpoint_model(True)
                    if trainer.C.checkpoint_encoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_encoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if trainer.C.checkpoint_processor_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_processor(trainer.C.segments)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if trainer.C.checkpoint_decoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_decoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                istep_change = iter - (trainer.C.num_iters_step1 + trainer.C.num_iters_step2)
                if finetune and (istep_change % trainer.C.step_change_freq == 0) and (iter != tagged_iter):
                    update_dataloader = True
                    tagged_iter = iter

                # update the dataloader for finetuning
                if update_dataloader:
                    num_rollout_steps = istep_change // trainer.C.step_change_freq + 2
                    trainer.datapipe = MERRA2NCDatapipe( device=dist.device, process_rank=dist.rank, world_size=dist.world_size )
                    update_dataloader = False
                    rank_zero_logger.info( f"Switching to {num_rollout_steps}-step rollout!" )
                    break

                # prepare the data
                # TODO modify for history > 0
                data_x = data[0]["invar"]
                data_y = data[0]["outvar"]
                # move to device & dtype
                data_x = data_x.to(dtype=trainer.dtype)
                grid_nfeat = data_x
                y = data_y.to(dtype=trainer.dtype).to(device=dist.device)

                # training step
                loss = trainer.train(grid_nfeat, y)
                if dist.rank == 0:
                    loss_agg += loss.detach().cpu()

                # validation
                if dist.rank == 0 and iter % trainer.C.val_freq == 0:
                    # free up GPU memory
                    del data_x, y
                    torch.cuda.empty_cache()
                    error = trainer.validation.step( channels=list(np.arange(trainer.C.num_channels_val)), iter=iter )
                    logger.log(f"iteration {iter}, Validation MSE: {error:.04f}")

                # distributed barrier
                if dist.world_size > 1:
                    torch.distributed.barrier()

                # print logs and save checkpoint
                if dist.rank == 0 and iter % trainer.C.save_freq == 0:
                    save_checkpoint(
                        os.path.join(trainer.C.ckpt_path, trainer.C.ckpt_name),
                        models=trainer.model,
                        optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler,
                        scaler=trainer.scaler,
                        epoch=iter,
                    )
                    logger.info(f"Saved model on rank {dist.rank}")
                    logger.log( f"iteration: {iter}, loss: {loss_agg/trainer.C.save_freq:10.3e}, time per iter: {(time.time()-start)/trainer.C.save_freq:10.3e}" )
                    wb.log( dict( loss= loss_agg / trainer.C.save_freq, learning_rate= trainer.scheduler.get_last_lr()[0] ), step=iter )
                    loss_agg = 0
                    start = time.time()
                iter += 1

                torch.cuda.nvtx.range_pop()

                # wrap up & terminate if training is finished
                if iter >= trainer.C.num_iters_step1 + trainer.C.num_iters_step2 + trainer.C.num_iters_step3:
                    if dist.rank == 0:
                        del data_x, y
                        torch.cuda.empty_cache()
                        error = trainer.validation.step( channels=list(np.arange(trainer.C.num_channels_val)), iter=iter )
                        logger.log(f"iteration {iter}, Validation MSE: {error:.04f}")
                        save_checkpoint(
                            os.path.join(trainer.C.ckpt_path, trainer.C.ckpt_name),
                            trainer.model,
                            trainer.optimizer,
                            trainer.scheduler,
                            trainer.scaler,
                            iter,
                        )
                        logger.info(f"Saved model on rank {dist.rank}")
                        logger.log( f"iteration: {iter}, loss: {loss_agg/trainer.C.save_freq:10.3e}, time per iter: {(time.time()-start)/self.C.save_freq:10.3e}" )
                    terminate_training = True
                    break
            if terminate_training:
                rank_zero_logger.info("Finished training!")
                break
