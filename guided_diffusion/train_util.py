import copy
import functools
import os

import blobfile as bf
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import cv2
import torch.distributed as dist

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
from .ssim_loss import SSIMLoss
import numpy as np
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import tqdm 
from tqdm import tqdm
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from .test_diff import diffusion_test

# import clip
class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_dat,
        batch_size,
        microbatch,
        lr,
        log_interval,
        save_interval,
        test_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        feature_extractor=None,
        feature_loss_weight=0.0,
        feature_loss_type="cosine",
        perceptual_loss_weight=0.0,
        vgg_weight=1.0,
        ssim_weight=1.0,
        stage1_steps=15000,
        stn_alignment_weight=0.0,  # 🌟 新增：STN对齐损失权重
        stn_reg_weight=0.0,  # 🌟 新增：STN正则化权重
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data=val_dat
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr

        self.test_interval = test_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.feature_extractor = feature_extractor
        self.feature_loss_weight = feature_loss_weight
        self.feature_loss_type = feature_loss_type
        
        # 🔥 感知损失相关参数（VGG + SSIM）
        self.perceptual_loss_weight = perceptual_loss_weight
        self.stage1_steps = stage1_steps  # 四阶段训练的第一个分界点
        self.perceptual_loss = None
        if perceptual_loss_weight > 0.0:
            try:
                from .perceptual_losses import CombinedPerceptualLoss
                self.perceptual_loss = CombinedPerceptualLoss(
                    device=dist_util.dev(),
                    vgg_weight=vgg_weight,
                    ssim_weight=ssim_weight
                )
                if dist.get_rank() == 0:
                    logger.log(f"✓ 感知损失已启用: VGG权重={vgg_weight}, SSIM权重={ssim_weight}")
            except Exception as e:
                if dist.get_rank() == 0:
                    logger.log(f"✗ 感知损失初始化失败: {e}")
                    logger.log("  训练将继续，但不使用VGG+SSIM损失")
                self.perceptual_loss = None
        
        # 🌟 STN损失相关参数
        self.stn_alignment_weight = stn_alignment_weight
        self.stn_reg_weight = stn_reg_weight
        self.stn_alignment_loss = None
        self.stn_reg_loss = None
        
        if stn_alignment_weight > 0.0 or stn_reg_weight > 0.0:
            try:
                from .stn_utils import STNAlignmentLoss, STNRegularization
                if stn_alignment_weight > 0.0:
                    self.stn_alignment_loss = STNAlignmentLoss(loss_type='mse')
                if stn_reg_weight > 0.0:
                    self.stn_reg_loss = STNRegularization(reg_type='identity')
                
                if dist.get_rank() == 0:
                    logger.log(f"✓ STN损失已启用: 对齐权重={stn_alignment_weight}, 正则化权重={stn_reg_weight}")
                    logger.log(f"✓ 四阶段渐进式训练策略:")
                    logger.log(f"  - 阶段0 (0-5000步): MSE + STN对齐 (建立空间对齐基础)")
                    logger.log(f"  - 阶段1 (5000-{stage1_steps}步): MSE + STN + ArcFace (建立身份映射)")
                    logger.log(f"  - 阶段2 ({stage1_steps}-{stage1_steps*2}步): MSE + STN + ArcFace + SSIM (优化结构)")
                    logger.log(f"  - 阶段3 (>{stage1_steps*2}步): 全部损失 (精细化纹理)")
            except Exception as e:
                if dist.get_rank() == 0:
                    logger.log(f"✗ STN损失初始化失败: {e}")
                    logger.log("  训练将继续，但不使用STN损失")
                self.stn_alignment_loss = None
                self.stn_reg_loss = None

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = None or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                dict_load = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                self.model.load_state_dict(dict_load, strict=False)
              
        dist_util.sync_params(self.model.parameters())


    def test(self, run, phase='test',skip_timesteps=0,iter=0):
        diffusion_test( self.val_data, self.model,self.diffusion,'./results', run, phase, skip_timesteps,iter )


    def run_loop(self, run):
        num_iter=100000

        for i in tqdm(range(num_iter)):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            # if self.step % self.log_interval == 0:
            #         run.log_metrics(logger.getkvs())

            if (self.step + 1) % self.save_interval == 0:
                self.save()
                
            if (self.step) % self.test_interval == 0:
                self.test(run, phase='train', skip_timesteps=0, iter = i)

            self.step += 1


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)

        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # 🔥 四阶段训练策略：根据当前步数决定使用哪些损失
            current_step = self.step + self.resume_step
            # 感知损失在阶段2和阶段3启用（步数 >= stage1_steps）
            use_perceptual = current_step >= self.stage1_steps
            
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                feature_extractor=self.feature_extractor,
                feature_weight=self.feature_loss_weight,
                feature_loss_type=self.feature_loss_type,
                perceptual_loss=self.perceptual_loss if use_perceptual else None,
                perceptual_weight=self.perceptual_loss_weight if use_perceptual else 0.0,
                current_step=current_step,
                stn_alignment_loss=self.stn_alignment_loss,
                stn_alignment_weight=self.stn_alignment_weight,
                stn_reg_loss=self.stn_reg_loss,
                stn_reg_weight=self.stn_reg_weight,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        if(dist.get_rank()==0):
            logger.logkv("step", self.step + self.resume_step)
            logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint( params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model")
                filename = f"model{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join("./weights", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)


        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        if(dist.get_rank()==0):

            logger.logkv_mean(key, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
