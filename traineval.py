"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file contains the main training and evaluation pipeline for the MMTwin model,
which predicts hand trajectories using multimodal diffusion models.

Key Components:
1. Hand Trajectory Prediction (HTP) diffusion model
2. Egomotion diffusion model
3. Various encoders and decoders for processing different input modalities
4. Training/evaluation loop with GPU support
"""

import argparse
import os
import random
import datetime
import logging.config
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from options import expopts
from netscripts.get_optimizer import get_optimizer
from netscripts import modelio
from models.step_sample import create_named_schedule_sampler
from models.utils import dist_util, logger
from basic_utils import create_model_and_diffusion,homo_create_model_and_diffusion
from data_utils.EgoPAT3DLoader import build_dataloaders
from netscripts.epoch_feat import TrainEvalLoop

def main(args, parser):
    """
    Main function for training and evaluating the MMTwin model.
    
    Args:
        args: Command line arguments and configurations
        parser: Argument parser for additional parameter handling
    """
    
    # Initialize random seeds for reproducibility
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    
    # Setup distributed training if applicable
    dist_util.setup_dist()
    start_epoch = 0

    # Check and report GPU availability
    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
    
    # Configuration for Hand Trajectory Prediction (HTP) diffusion model
    htp_model_diff_args = {
        "vl_dim": args.vl_dim,
        "patch_dim": args.patch_dim,
        "loc_feat_dim": args.loc_feat_dim,
        "glip_feat_dim": args.glip_feat_dim,
        "target_dim": args.target_dim,
        "n_layers": args.n_layers,
        "diffusion_steps": args.diffusion_steps,
        "noise_schedule": args.noise_schedule,
        "learn_sigma": args.learn_sigma,
        "timestep_respacing": args.timestep_respacing,
        "predict_xstart": args.predict_xstart,
        "rescale_timesteps": args.rescale_timesteps,
        "sigma_small": args.sigma_small,
        "rescale_learned_sigmas": args.rescale_learned_sigmas,
        "use_kl": args.use_kl,
    }
    print("\n\u2022 HTP diffusion setups\n", htp_model_diff_args)
    
    # Initialize HTP model and diffusion components
    htp_denoise_model, pre_encoder, htp_diffusion, post_decoder, motion_encoder, loc_encoder, glip_encoder, voxel_encoder, occ_feat_encoder = create_model_and_diffusion(**htp_model_diff_args)

    # Configuration for Egomotion diffusion model
    em_model_diff_args = {
        "feat_dim": args.em_feat_dim,
        "n_layers": args.em_n_layers,
        "diffusion_steps": args.em_diffusion_steps,
        "noise_schedule": args.em_noise_schedule,
        "learn_sigma": args.em_learn_sigma,
        "timestep_respacing": args.em_timestep_respacing,
        "predict_xstart": args.em_predict_xstart,
        "rescale_timesteps": args.em_rescale_timesteps,
        "sigma_small": args.em_sigma_small,
        "rescale_learned_sigmas": args.em_rescale_learned_sigmas,
        "use_kl": args.em_use_kl,
    }
    print("\n\u2022 EM diffusion setups\n", em_model_diff_args)
    
    # Initialize EM model and diffusion components
    homo_denoise_model, homo_diffusion = homo_create_model_and_diffusion(**em_model_diff_args)

    mmtwin_models = {
        'pre_encoder': pre_encoder,
        'htp_denoise_model': htp_denoise_model,
        'post_decoder': post_decoder,
        'motion_encoder': motion_encoder,
        'loc_encoder': loc_encoder,
        'glip_encoder': glip_encoder,
        'voxel_encoder': voxel_encoder,
        'occ_feat_encoder': occ_feat_encoder,
        'homo_denoise_model': homo_denoise_model
    }

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Print parameter counts for each component
    print(f"\n\u2022 Parameters:")
    total_params = 0
    for name, model in mmtwin_models.items():
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters")
        total_params += params
    print(f"Total: {total_params:,}")

    # Initialize schedule samplers for diffusion models
    schedule_sampler_args = args.schedule_sampler
    htp_schedule_sampler = create_named_schedule_sampler(schedule_sampler_args, htp_diffusion)
    homo_schedule_sampler = create_named_schedule_sampler(schedule_sampler_args, homo_diffusion)

    # Setup data loaders and optimizers based on evaluation mode
    if args.evaluate:
        # Evaluation mode setup
        args.epochs = start_epoch + 1
        traj_val_loader = None
        optimizer=None
        scheduler=None
        test_loader, testnovel_loader = build_dataloaders(args, phase='eval')
    else:
        # Training mode setup
        train_loader, traj_val_loader = build_dataloaders(args, phase='train')
        optimizer, scheduler = get_optimizer(args, 
                                            pre_encoder=pre_encoder, 
                                            model_denoise=htp_denoise_model,
                                            homo_denoise=homo_denoise_model, 
                                            post_decoder=post_decoder, 
                                            loc_encoder=loc_encoder, 
                                            glip_encoder=glip_encoder, 
                                            voxel_encoder=voxel_encoder, 
                                            occ_feat_encoder=occ_feat_encoder, 
                                            train_loader=train_loader, 
                                            motion_encoder=motion_encoder)
    
    if args.evaluate:
        loader = testnovel_loader if args.test_novel else test_loader
    else:
        loader = train_loader

    # Initialize and run the training/evaluation loop
    TrainEvalLoop(
            epochs = args.epochs,
            loader=loader,
            evaluate=args.evaluate,
            use_cuda=True,
            optimizer=optimizer,
            scheduler=scheduler,
            pre_encoder=pre_encoder,
            model_denoise=htp_denoise_model,
            homo_denoise=homo_denoise_model,
            htp_diffusion=htp_diffusion,
            homo_diffusion=homo_diffusion,
            post_decoder=post_decoder,
            motion_encoder=motion_encoder,
            loc_encoder=loc_encoder,
            glip_encoder=glip_encoder,
            voxel_encoder=voxel_encoder,
            occ_feat_encoder=occ_feat_encoder,
            htp_schedule_sampler=htp_schedule_sampler,
            homo_schedule_sampler=homo_schedule_sampler,
            resume=args.resume,
            snapshot=args.snapshot,
            log_dir=args.log_dir,
            ratio=args.ratio,
            infer_gap=args.infer_gap,
            em_infer_gap=args.em_infer_gap,
            losses_homo_w=args.losses_homo_w,
            losses_reg_w=args.losses_reg_w,
            losses_angle_w=args.losses_angle_w,
            test_space=args.test_space,
    ).run_loop()


if __name__ == "__main__":
    # Initialize argument parser with configuration options
    parser = argparse.ArgumentParser(description="Hand Trajectory Prediction")
    expopts.add_nets_opts(parser)       # Add network architecture options
    expopts.add_data_opts(parser)       # Add data loading options
    expopts.add_traineval_opts(parser)  # Add training/evaluation options
    expopts.add_exp_opts(parser)        # Add experiment options
    args = parser.parse_args()

    # Adjust batch size for multi-GPU training   # deprecated multi-GPU for now
    if args.use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        args.batch_size = args.batch_size * num_gpus

    main(args, parser)