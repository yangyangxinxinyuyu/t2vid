"""
Train a super-resolution model.
"""

import argparse
import os
import torch.distributed as dist
from torch.utils.data import DataLoader

from core.wandb_logger import WandbLogger
from core.face_features import FaceFeatureExtractor, ArcFaceFeatureExtractor
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.valdata import ValData


def main(run):
    args = create_argparser().parse_args()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to DETAIL for runtime logging.

    dist_util.setup_dist()
    if dist.get_rank() == 0:
        logger.configure(dir="./experiments/log/")
    if dist.get_rank() == 0:
        logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data_dir1 = os.path.join(args.data_dir, "TH/")
    gt_dir = os.path.join(args.data_dir, "VIS/")

    val_data = DataLoader(
        ValData(args.test_dir), batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    data = load_superres_data(
        data_dir1,
        gt_dir,
        args.batch_size,
        image_size=128,
        num_workers=args.num_workers,
    )
    if dist.get_rank() == 0:
        logger.log("training...")

    feature_extractor = None
    if args.feature_loss_weight > 0:
        if args.feature_extractor == "clip":
            feature_extractor = FaceFeatureExtractor(dist_util.dev())
        elif args.feature_extractor == "arcface":
            feature_extractor = ArcFaceFeatureExtractor(
                dist_util.dev(), model_name=args.feature_model_name
            )
        else:
            raise ValueError(f"Unsupported feature_extractor: {args.feature_extractor}")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_dat=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        test_interval=args.test_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        feature_extractor=feature_extractor,
        feature_loss_weight=args.feature_loss_weight,
        feature_loss_type=args.feature_loss_type,
        perceptual_loss_weight=args.perceptual_loss_weight,
        vgg_weight=args.vgg_weight,
        ssim_weight=args.ssim_weight,
        stage1_steps=args.stage1_steps,
        stn_alignment_weight=args.stn_alignment_weight,  # 🌟 新增
        stn_reg_weight=args.stn_reg_weight,  # 🌟 新增
    ).run_loop(run)


def load_superres_data(data_dir, gt_dirs, batch_size, image_size, num_workers=4):
    data = load_data(
        data_dir=data_dir,
        gt_dir=gt_dirs,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="./data/train/",
        test_dir="./data/test/TH/",
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=8,
        log_interval=500,
        save_interval=10000,
        test_interval=10000,
        resume_checkpoint="./weights/latest.pt",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        
        # 🎯 ArcFace身份特征损失参数（提升识别率）
        feature_extractor="arcface",
        feature_model_name="buffalo_l",
        feature_loss_weight=1.0,  # ArcFace特征损失权重（推荐1.0-2.0）
        feature_loss_type="cosine",
        
        # 🔥 VGG + SSIM 感知损失参数（提升SSIM和图像质量）
        perceptual_loss_weight=0.15,  # 感知损失总权重（推荐0.1-0.2）
        vgg_weight=1.0,  # VGG损失相对权重
        ssim_weight=2.0,  # SSIM损失相对权重（提高以优化SSIM指标）
        
        # 🌟 STN空间对齐参数（自动对齐热红外和可见光图像）
        use_stn=True,  # 是否使用STN（推荐True）
        stn_type="single",  # STN类型: 'single' 或 'multiscale'
        stn_localization="lightweight",  # 定位网络: 'lightweight'(快) 或 'standard'(准)
        stn_transform="affine",  # 变换类型: 'affine'(仿射) 或 'tps'(薄板样条)
        stn_attention=False,  # 是否使用空间注意力
        stn_num_scales=2,  # 多尺度STN的尺度数量
        stn_alignment_weight=0.1,  # STN对齐损失权重（推荐0.05-0.2）
        stn_reg_weight=0.01,  # STN正则化权重（推荐0.005-0.02）
        
        # 📊 四阶段训练策略（避免损失冲突）
        # 阶段0 (0-5000步): MSE + STN对齐（建立空间对齐基础）
        # 阶段1 (5000-15000步): MSE + STN + ArcFace（建立身份映射）
        # 阶段2 (15000-30000步): MSE + STN + ArcFace + SSIM（优化结构）
        # 阶段3 (>30000步): 全部损失（精细化纹理）
        # 阶段2 (15000-30000步): MSE + ArcFace + STN + SSIM（优化结构）
        # 阶段3 (>30000步): MSE + ArcFace + STN + SSIM + VGG（精细化纹理）
        stage1_steps=15000,  # 阶段1结束点
        
        num_workers=4,  # 数据加载的worker数量
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    run = WandbLogger()
    main(run)
