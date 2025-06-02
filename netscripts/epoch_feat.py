"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file contains trajectory processing utilities and the main training/evaluation loop 
for the MMTwin model's diffusion-based trajectory prediction.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import os
from tqdm import trange
import functools
import logging.config
import datetime
from functools import partial
from models.step_sample import LossAwareSampler, UniformSampler
from netscripts import modelio
from netscripts.epoch_utils import progress_bar as bar, AverageMeters
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from models.utils import dist_util, logger
from einops import rearrange

# Camera intrinsic parameters for coordinate transformations
intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
              'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
              'w': 3840, 'h': 2160}

def denormalize(traj, target='3d', max_depth=3.0):
    """Convert normalized trajectory to pixel (2D) or world (3D) coordinates"""
    u = traj[:, 0] * intrinsics['w']
    v = traj[:, 1] * intrinsics['h']
    traj2d = np.stack((u, v), axis=1)
    if target == '3d':
        Z = traj[:, 2] * max_depth
        X = (u - intrinsics['ox']) * Z / intrinsics['fx']
        Y = (v - intrinsics['oy']) * Z / intrinsics['fy']
        traj3d = np.stack((X, Y, Z), axis=1)
        return traj3d
    return traj2d

def XYZ_to_uv(traj3d, intrinsics):
    """Convert 3D world coordinates to 2D pixel coordinates"""
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['ox'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['oy'])
    # clip the coordinates 
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['w'])
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['h'])
    traj2d = np.floor(traj2d).astype(np.int32)
    return traj2d

def normalize_2d(uv, intrinsics):
    """Normalize 2D pixel coordinates to [0,1] range"""
    uv_norm = np.copy(uv).astype(np.float32)
    uv_norm[:, 0] /= intrinsics['w']
    uv_norm[:, 1] /= intrinsics['h']
    return uv_norm

def get_traj_observed(traj_all, num_ratios, mask_o):
    """Prepare observed trajectory segments for model input"""
    traj_input = traj_all[:, None].repeat(1, num_ratios, 1, 1)
    traj_input = rearrange(traj_input, 'b n t c-> (b n) t c')
    traj_mask = mask_o.unsqueeze(-1)
    traj_input = traj_input * traj_mask
    return traj_input

def get_masks(batch_size, ratios, max_frames, nframes, device):
    """Generate observation and prediction masks for trajectory segments"""
    num_ratios = ratios.size(1)
    mask_o = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)
    mask_u = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)
    last_frames = torch.zeros((batch_size, num_ratios)).long()
    for b in range(batch_size):
        num_full = int(nframes[b])
        num_obs = torch.floor(num_full * ratios[b]).to(torch.long)
        last_frames[b] = num_obs - 1
        for i, n_o in enumerate(num_obs):
            mask_o[b, i, :n_o] = 1
            mask_u[b, i, n_o: num_full] = 1
    return mask_o, mask_u, last_frames

class TrainEvalLoop:
    """Main training and evaluation loop for MMTwin trajectory prediction"""
    def __init__(self, start_epoch=0, epochs=1000, snapshot=1, loader=None, evaluate=False,
                 use_cuda=True, scheduler=None, optimizer=None, pre_encoder=None,
                 motion_encoder=None, loc_encoder=None, glip_encoder=None,
                 voxel_encoder=None, occ_feat_encoder=None, model_denoise=None,
                 homo_denoise=None, htp_diffusion=None, homo_diffusion=None,
                 post_decoder=None, htp_schedule_sampler=None, homo_schedule_sampler=None,
                 resume=None, log_dir=None, infer_gap=10, em_infer_gap=10, ratio=0.6,
                 test_space="3d", losses_homo_w=0.0, losses_reg_w=0.0, losses_angle_w=0.0):

        # Initialize all model components with DDP wrappers
        self.pre_encoder = DDP(
            pre_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.model_denoise = DDP(
            model_denoise.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.homo_denoise = DDP(
            homo_denoise.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.post_decoder = DDP(
            post_decoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.motion_encoder = DDP(
            motion_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.loc_encoder = DDP(
            loc_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.glip_encoder = DDP(
            glip_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.voxel_encoder = DDP(
            voxel_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.occ_feat_encoder = DDP(
            occ_feat_encoder.to(dist_util.dev()) ,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

        self.htp_diffusion = htp_diffusion
        self.homo_diffusion = homo_diffusion
        self.start_epoch = start_epoch
        self.all_epochs = epochs
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.loader = loader
        self.use_cuda = use_cuda
        self.scheduler = scheduler
        self.htp_schedule_sampler = htp_schedule_sampler
        self.homo_schedule_sampler = homo_schedule_sampler
        self.infer_gap = infer_gap
        self.em_infer_gap = em_infer_gap

        self.ade_list = []
        self.fde_list = []
        self.pred_list = []
        self.gt_list = []

        # Load checkpoints if resuming training
        self.resume = resume
        if resume is not None:
            self.start_epoch = modelio.load_checkpoint_by_name(self.pre_encoder, resume_path=resume[0], state_dict_name="pre_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.model_denoise, resume_path=resume[0], state_dict_name="model_denoise_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.homo_denoise, resume_path=resume[0], state_dict_name="homo_denoise_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.post_decoder, resume_path=resume[0], state_dict_name="post_decoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.motion_encoder, resume_path=resume[0], state_dict_name="motion_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.loc_encoder, resume_path=resume[0], 
            state_dict_name="loc_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.glip_encoder, resume_path=resume[0], 
            state_dict_name="glip_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.voxel_encoder, resume_path=resume[0], 
            state_dict_name="voxel_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.occ_feat_encoder, resume_path=resume[0], 
            state_dict_name="occ_feat_encoder_state_dict", strict=False, device=dist_util.dev())
            print("finish loading diffusion model from epoch {}".format(self.start_epoch))

        dist_util.sync_params(self.pre_encoder.parameters())
        dist_util.sync_params(self.model_denoise.parameters())
        dist_util.sync_params(self.homo_denoise.parameters())
        dist_util.sync_params(self.post_decoder.parameters())
        dist_util.sync_params(self.motion_encoder.parameters())
        dist_util.sync_params(self.loc_encoder.parameters())
        dist_util.sync_params(self.glip_encoder.parameters())
        dist_util.sync_params(self.voxel_encoder.parameters())
        dist_util.sync_params(self.occ_feat_encoder.parameters())

        if self.evaluate:
            self.all_epochs = 1
            self.start_epoch = 0

        # Setup logger
        self.logger = logging.getLogger('main')
        self.logger.setLevel(level=logging.DEBUG)
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        if evaluate:
            handler = logging.FileHandler(os.path.join(log_dir, f"eval_{time_str}.log"))
        else:
            handler = logging.FileHandler(os.path.join(log_dir, f"train_{time_str}.log"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.snapshot = snapshot
        self.ratio = ratio
        self.losses_homo_w = losses_homo_w
        self.losses_reg_w = losses_reg_w
        self.losses_angle_w = losses_angle_w
        self.test_space = test_space


    def _get_loc_features(self, traj, num_ratios, mask):
        """Extract location features from trajectory segments"""
        traj_input = get_traj_observed(traj, num_ratios, mask)
        loc_feat = self.loc_encoder(traj_input)
        return loc_feat

    def run_loop(self):
        """Main training/evaluation loop"""
        for model in [self.pre_encoder, self.model_denoise, self.homo_denoise, 
                     self.post_decoder, self.motion_encoder, self.loc_encoder,
                     self.glip_encoder, self.voxel_encoder, self.occ_feat_encoder]:
            model.train() if not self.evaluate else model.eval()

        print("\n\u2022 Epochs num = ", self.all_epochs)
        for epoch in range(self.start_epoch, self.all_epochs):
            if not self.evaluate:
                print("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
                self.logger.info("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
                self.epoch_pass(phase='train', epoch=epoch, train=True,)
            else:
                self.epoch_pass(phase='traj', epoch=epoch, train=False,)


    def epoch_pass(self, epoch, phase, train=True):
        """Single epoch pass (training or evaluation)"""
        time_meters = AverageMeters()

        if train:
            loss_meters = AverageMeters()
        else:
            print(f"\u2022 Evaluate models from {self.resume}")
            preds_traj, gts_traj, valids_traj = [], [], []

        end = time.time()
        loss_all = []
        for batch_idx, sample in enumerate(self.loader):
            if train:
                self.optimizer.zero_grad()

                filename, clip, odometry, nframes, traj_gt, glip_feat_raw, motion_feats_raw, voxel_feats_raw = sample
                time_meters.add_loss_value("data_time", time.time() - end)
                clip = clip.to(dist_util.dev())
                traj_gt = traj_gt.to(dist_util.dev())
                voxel_feats_raw = voxel_feats_raw.to(dist_util.dev())

                # Process voxel features
                voxel_feats_raw = voxel_feats_raw.unsqueeze(1)
                voxel_feats_all_space = self.voxel_encoder(voxel_feats_raw)
                voxel_feats_all_space = voxel_feats_all_space.squeeze(1)
                voxel_feats_all_space = voxel_feats_all_space.view(voxel_feats_all_space.shape[0], voxel_feats_all_space.shape[1], -1).contiguous()
                voxel_feats_all_space = voxel_feats_all_space.permute(0,2,1)
                occ_feat_encoded = self.occ_feat_encoder(voxel_feats_all_space.contiguous())

                # Temporarily use the unified ratio.
                ratios = torch.ones((traj_gt.shape[0], 1)) * self.ratio
                batch_size, max_frames, num_ratios = traj_gt.size(0), traj_gt.size(1), ratios.size(1)

                # Process motion and GLIP features
                motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                motion_feat_encoded = self.motion_encoder(motion_feats_raw) 
                glip_feat_raw = glip_feat_raw.to(dist_util.dev())
                glip_feat = self.glip_encoder(glip_feat_raw)

                mask_o, mask_u, last_obs_frames = get_masks(batch_size, ratios, max_frames, nframes, dist_util.dev())
                mask_o_for_input = rearrange(mask_o, 'b n t -> (b n) t')
                mask_u_for_input = rearrange(mask_u, 'b n t -> (b n) t')
                mask_ou_for_input = mask_o_for_input + mask_u_for_input
                all_loc_feats = self._get_loc_features(traj_gt, num_ratios, mask_ou_for_input)

                assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                # Combine features and compute losses
                right_feat = torch.stack((glip_feat, all_loc_feats), dim=2)
                right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                right_feat_encoded = self.pre_encoder(right_feat)

                t, ada_weights = self.htp_schedule_sampler.sample(glip_feat.shape[0], dist_util.dev()) 

                # Egomotion prediction
                homo_compute_losses = functools.partial(
                    self.homo_diffusion.training_losses,
                    self.homo_denoise,
                    None,
                    [motion_feat_encoded, motion_feat_encoded],
                    t,
                    [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                    motion_feat_encoded,
                )
                homo_loss_feat_dict = homo_compute_losses()
                homo_loss_feat_level = homo_loss_feat_dict["loss_feat_level"]

                losses_homo = homo_loss_feat_level['mse_r'] + homo_loss_feat_level['tT_loss_r']
                losses_homo = losses_homo.mean() * self.losses_homo_w

                motion_feat_encoded_new = motion_feat_encoded.clone()
                len_observation = last_obs_frames + 1 
                rec_motion_feat = homo_loss_feat_dict['rec_feature_r']
                for lo in range(rec_motion_feat.shape[0]):
                    nframes_this_b = nframes[lo]
                    nfuture = int(nframes_this_b - len_observation[lo])
                    motion_feat_encoded_new[lo, len_observation[lo]:, :] = rec_motion_feat[lo, len_observation[lo]:, :]

                # Hand motion prediction
                compute_losses = functools.partial(
                    self.htp_diffusion.training_losses,
                    self.model_denoise,
                    self.post_decoder,
                    [right_feat_encoded, right_feat_encoded],
                    t,
                    [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                    [motion_feat_encoded_new,occ_feat_encoded],
                )

                # Diffusion losses, Reg losses, Angle losses, Traj losses
                loss_feat_dict = compute_losses()
                loss_feat_level = loss_feat_dict["loss_feat_level"]
                rec_feature_r = loss_feat_dict["rec_feature_r"]
                
                future_feature = rec_feature_r
                pred_future_traj = self.post_decoder(future_feature)
                pred_future_traj_r = pred_future_traj

                broaded_future_mask = torch.broadcast_to(mask_ou_for_input.unsqueeze(dim=-1), traj_gt.shape)
                reg_future_traj_r = self.post_decoder(right_feat_encoded.contiguous())
                reg_future_traj_r_masked = reg_future_traj_r * broaded_future_mask
                traj_gt_masked = traj_gt * broaded_future_mask

                losses_reg_r = torch.sum((reg_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                losses_reg_r = losses_reg_r.sum(-1) * self.losses_reg_w

                broaded_future_mask_for_angle = mask_u_for_input
                pred_future_traj_r_shifted_back = pred_future_traj_r[:, 1:, :]
                pred_future_traj_r_shifted_forward = pred_future_traj_r[:, 0:-1, :]
                pred_future_traj_r_shifted_delta = pred_future_traj_r_shifted_back - pred_future_traj_r_shifted_forward
                future_head_r =  traj_gt
                future_head_r_shifted_back = future_head_r[:, 1:, :]
                future_head_r_shifted_forward = future_head_r[:, 0:-1, :]
                future_head_r_shifted_delta = future_head_r_shifted_back - future_head_r_shifted_forward
                cos_sim = F.cosine_similarity(pred_future_traj_r_shifted_delta, future_head_r_shifted_delta, dim=-1)
                cos_distance = 1 - cos_sim
                cos_distance = torch.cat((cos_distance, torch.zeros((pred_future_traj_r.shape[0],1)).to(dist_util.dev())),dim=-1)
                cos_distance = cos_distance * broaded_future_mask_for_angle
                losses_future_angle_r = torch.sum(cos_distance, dim=-1) * self.losses_angle_w

                pred_future_traj_r_masked = pred_future_traj_r * broaded_future_mask
                traj_gt_masked = traj_gt * broaded_future_mask
                losses_future_traj_r = torch.sum((pred_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                losses_future_traj_r = losses_future_traj_r.sum(-1)

                losses_two_hand = loss_feat_level['mse_r'] + loss_feat_level['tT_loss_r'] + losses_future_traj_r + losses_reg_r + losses_future_angle_r

                if isinstance(self.htp_schedule_sampler, LossAwareSampler):
                    self.htp_schedule_sampler.update_with_local_losses(
                        t, losses_two_hand.detach()
                    )

                loss = (losses_two_hand * ada_weights).mean() + losses_homo

                model_losses = {
                        "losses_future_traj_r":losses_future_traj_r.mean(),
                        "losses_future_angle_r":losses_future_angle_r.mean(),
                        "losses_rec_r":loss_feat_level['mse_r'].mean(),
                        "losses_homo":losses_homo,
                        "losses_in_total":loss,
                }

                loss.backward()
                self.optimizer.step()

                for key, val in model_losses.items():
                    if val is not None:
                        loss_meters.add_loss_value(key, val.detach().cpu().item())

                time_meters.add_loss_value("batch_time", time.time() - end)

                # Logging
                if dist_util.get_rank() == 0:
                    self.logger.info(loss_meters.average_meters["losses_in_total"].avg)
                    suffix = "Epoch:{epoch} " \
                            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                            "| losses_future_traj_r: {losses_future_traj_r:.3f} " \
                            "| losses_future_angle_r: {losses_future_angle_r:.3f} " \
                            "| losses_rec_r: {losses_rec_r:.3f}" \
                            "| losses_homo: {losses_homo:.3f}" \
                            "| losses_in_total: {losses_in_total:.3f} ".format(
                            epoch=epoch, batch=batch_idx + 1,
                            size=len(self.loader), 
                            data=time_meters.average_meters["data_time"].val,
                            bt=time_meters.average_meters["batch_time"].avg,
                            losses_future_traj_r=model_losses["losses_future_traj_r"],
                            losses_future_angle_r=model_losses["losses_future_angle_r"],
                            losses_rec_r=model_losses["losses_rec_r"],
                            losses_homo=model_losses["losses_homo"],
                            losses_in_total=model_losses["losses_in_total"],
                            )
                    self.logger.info(suffix)
                    bar(suffix)
                end = time.time()

            else:

                filename, clip, odometry, nframes, traj_gt, glip_feat_raw, motion_feats_raw, voxel_feats_raw = sample

                voxel_feats_raw = voxel_feats_raw.to(dist_util.dev())
                voxel_feats_raw = voxel_feats_raw.unsqueeze(1)
                voxel_feats_all_space = self.voxel_encoder(voxel_feats_raw)
                voxel_feats_all_space = voxel_feats_all_space.squeeze(1)
                voxel_feats_all_space = voxel_feats_all_space.view(voxel_feats_all_space.shape[0], voxel_feats_all_space.shape[1], -1).contiguous()
                voxel_feats_all_space = voxel_feats_all_space.permute(0,2,1)
                occ_feat_encoded = self.occ_feat_encoder(voxel_feats_all_space.contiguous())
                clip = clip.to(dist_util.dev())
                traj_gt = traj_gt.to(dist_util.dev())

                ratios = torch.ones((traj_gt.shape[0], 1)) * self.ratio
                batch_size, max_frames, num_ratios = traj_gt.size(0), traj_gt.size(1), ratios.size(1)

                glip_feat_raw = glip_feat_raw.to(dist_util.dev()) 
                glip_feat = self.glip_encoder(glip_feat_raw)

                # Get masks of observed and unobserved frames
                mask_o, mask_u, last_obs_frames = get_masks(batch_size, ratios, max_frames, nframes, dist_util.dev()) 

                motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                motion_feat_encoded = self.motion_encoder(motion_feats_raw)

                mask_o_for_input = rearrange(mask_o, 'b n t -> (b n) t')
                mask_u_for_input = rearrange(mask_u, 'b n t -> (b n) t')

                mask_ou_for_input = mask_o_for_input + mask_u_for_input
                all_loc_feats = self._get_loc_features(traj_gt, num_ratios, mask_ou_for_input) 

                assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                right_feat = torch.stack((glip_feat, all_loc_feats), dim=2)
                right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                right_feat_encoded = self.pre_encoder(right_feat)

                sample_fn = (
                    self.htp_diffusion.p_sample_loop
                )
                homo_sample_fn = (
                    self.homo_diffusion.p_sample_loop
                )

                # Generate predictions using diffusion sampling
                with torch.no_grad():
                    len_observation = last_obs_frames + 1
                    assert len_observation[0] == int(torch.sum(mask_o_for_input[0]==1).item())

                    for sample_idx in range(1):  # TODO: multiple samplings
                        for lo in range(len_observation.shape[0]):
                            nframes_this_b = nframes[lo]
                            nfuture = int(nframes_this_b - len_observation[lo])
                            pseudo_future = torch.zeros((nfuture,1024))
                            noise_r = torch.randn_like(pseudo_future).to(dist_util.dev())
                            right_feat_encoded[lo, len_observation[lo]:nframes_this_b, :] = noise_r
                        sample_shape = (right_feat_encoded.shape[0], right_feat_encoded.shape[1], right_feat_encoded.shape[2])

                        model_kwargs = {}
                        print("denoising ...")
                        for lo in range(len_observation.shape[0]):
                            nframes_this_b = nframes[lo]
                            nfuture = int(nframes_this_b - len_observation[lo])
                            pseudo_future = torch.zeros((nfuture, motion_feat_encoded.shape[-1]))
                            noise_r = torch.randn_like(pseudo_future).to(dist_util.dev())
                            motion_feat_encoded[lo, len_observation[lo]:nframes_this_b, :] = noise_r

                        homo_samples_r = homo_sample_fn(
                            model_denoise=self.homo_denoise,
                            shape=sample_shape,
                            noise=[motion_feat_encoded, motion_feat_encoded],
                            motion_feat_encoded=motion_feat_encoded,
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            clamp_step=0,
                            clamp_first=True,
                            x_start=[motion_feat_encoded, motion_feat_encoded],
                            gap=int(self.em_infer_gap-1),
                            device=dist_util.dev(),
                            valid_mask = [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                        )
                        homo_samples_r = homo_samples_r[-1]
                        motion_feat_encoded=homo_samples_r

                        samples_r = sample_fn(
                            model_denoise=self.model_denoise,
                            shape=sample_shape,
                            noise=[right_feat_encoded, right_feat_encoded],
                            motion_feat_encoded=[motion_feat_encoded, occ_feat_encoded],
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            clamp_step=0,
                            clamp_first=True,
                            x_start=[right_feat_encoded, right_feat_encoded],
                            gap=int(self.infer_gap-1),
                            device=dist_util.dev(),
                            valid_mask = [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                        )

                        samples_r = samples_r[-1]
                        pred_future_traj = self.post_decoder(samples_r)

                        # Compute metrics (ADE, FDE)
                        for b in range(samples_r.shape[0]):
                            num_full = nframes[b]  
                            traj_gt_per = traj_gt[b]
                            samples_r_per = pred_future_traj[b]
                            traj_gt_per += 0.5
                            samples_r_per += 0.5
                            if self.test_space == "3d":
                                traj_gt_per = denormalize(traj_gt_per.cpu().numpy(), target='3d')
                                samples_r_per = denormalize(samples_r_per.cpu().numpy(), target='3d') 
                            else:
                                traj_gt_per = normalize_2d(XYZ_to_uv(denormalize(traj_gt_per.cpu().numpy(), target='3d'), intrinsics), intrinsics)
                                samples_r_per = normalize_2d(XYZ_to_uv(denormalize(samples_r_per.cpu().numpy(), target='3d'), intrinsics), intrinsics)

                            start_ = len_observation[b]
                            end_ = num_full
                            traj_gt_per_valid = traj_gt_per[start_:end_]
                            samples_r_per_valid = samples_r_per[start_:end_]

                            for ttt in range(traj_gt_per_valid.shape[0]):
                                self.pred_list.append(samples_r_per_valid[ttt:ttt+1])
                                self.gt_list.append(traj_gt_per_valid[ttt:ttt+1])
                            assert len(self.pred_list) == len(self.gt_list)

                            displace_errors = np.sqrt(np.sum((samples_r_per_valid - traj_gt_per_valid)**2, axis=-1))
                            
                            ade = np.mean(displace_errors)
                            self.ade_list.append(ade)  
                            final_displace_errors = np.sqrt(np.sum((samples_r_per_valid[-1] - traj_gt_per_valid[-1])**2, axis=-1))
                            fde = final_displace_errors
                            self.fde_list.append(fde)

                        print(str(batch_idx+1) + "/" + str(len(self.loader)) + " ADE " + str(np.array(self.ade_list).mean()) + " FDE " + str(np.array(self.fde_list).mean()))
                        self.logger.info(str(batch_idx) + " mmtwin "+str(np.array(self.ade_list).mean())+" fde "+ str(np.array(self.fde_list).mean()))

        # End-of-epoch handling 
        if train:
            warmup_epochs = 0
            if (epoch + 1 - warmup_epochs) % self.snapshot == 0 and (dist_util.get_rank() == 0):
                print("save epoch "+str(epoch+1)+" checkpoint")
                modelio.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "pre_encoder_state_dict": self.pre_encoder.state_dict(),
                    "model_denoise_state_dict": self.model_denoise.state_dict(),
                    "homo_denoise_state_dict": self.homo_denoise.state_dict(),
                    "post_decoder_state_dict": self.post_decoder.state_dict(),
                    "motion_encoder_state_dict": self.motion_encoder.state_dict(),
                    "loc_encoder_state_dict": self.loc_encoder.state_dict(),
                    "glip_encoder_state_dict": self.glip_encoder.state_dict(),
                    "voxel_encoder_state_dict": self.voxel_encoder.state_dict(),
                    "occ_feat_encoder_state_dict": self.occ_feat_encoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                checkpoint="./mmtwin_weights",
                filename = f"checkpoint_{epoch+1}.pth.tar")
                torch.save(self.optimizer.state_dict(), "optimizer.pt")

                return loss_meters
        else:
            val_info = {}
            print("================================")
            print("MMTwin ADE ", np.array(self.ade_list).mean())
            print("MMTwin FDE ", np.array(self.fde_list).mean())
            print("================================")
            self.logger.info("MMTwin ADE "+str(np.array(self.ade_list).mean()))
            self.logger.info("MMTwin FDE "+str(np.array(self.fde_list).mean()))

            # single GPU for now
            # device_id_for_save = int(os.environ['LOCAL_RANK'])
            # saved_dir = "./collected_pred/"
            # np.save(saved_dir+"ours_ade"+str(device_id_for_save), np.array(self.ade_list))
            # if int(os.environ['LOCAL_RANK']) == 0:
            #     while 1:
            #         if len(os.listdir(saved_dir))==2:  # your gpu num
            #             ade_mean = []
            #             for i in range(2):
            #                 ade_mean = ade_mean + np.load(saved_dir+"ours_ade"+str(i)+".npy").tolist()
            #             print("collected ADE final ", np.array(ade_mean).mean())
            #             break
            #         time.sleep(2)
            #     self.logger.info("collected ADE final "+str(np.array(ade_mean).mean()))

            return val_info
