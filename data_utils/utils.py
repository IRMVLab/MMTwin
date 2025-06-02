"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import random
import os
import numpy as np
import torch
from copy import deepcopy
from functools import reduce
import cv2
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter



def set_deterministic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def send_to_gpu(data_tuple, device, non_blocking=False):
    gpu_tensors = []
    for item in data_tuple:
        gpu_tensors.append(item.to(device, non_blocking=non_blocking))
    return tuple(gpu_tensors)


def get_depth_anchors(traj_gt, num_obs, num_full, offset_anchor='Zc', time_orders=None):
    if offset_anchor == 'Zc':
        # use the last observed depth as the offset anchor
        anchors = traj_gt[num_obs-1: num_obs, 2:3]
    else:
        zs_obs = traj_gt[:num_obs, 2:3]
        if isinstance(zs_obs, np.ndarray):
            zs_obs = torch.from_numpy(zs_obs).to(time_orders.device)
        # use historical (z, t) values to fit a four-order multinomial model
        solution = torch.linalg.lstsq(time_orders[:num_obs, :], zs_obs).solution
        anchors = torch.matmul(time_orders[num_obs:num_full, :], solution)
        if isinstance(traj_gt, np.ndarray):
            anchors = anchors.cpu().numpy()
    return anchors


def gather_eval_results(outputs, nframes, trajectories, ignore_depth=False, centralize=False, return_unct=False):
    """ gather predictions and gts for evaluation purpose
        return: all_preds (N, 9, 3)
                all_gts (N, 9, 3)
    """
    # constant camera parameters
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    max_depth = 3
    num_order = 3
    
    trajectories = trajectories.cpu().numpy()
    all_preds, all_gts = {}, {}
    if return_unct: all_uncts = {}
    target = '3d' if trajectories.shape[-1] == 3 else '2d'
    # parse results
    for b, ratio_preds in enumerate(outputs):
        # get the number observed frames and full frames
        num_full = nframes[b]
        traj_gt = trajectories[b]
        if ignore_depth:
            traj_gt = traj_gt[:, :2]
        if centralize:
            traj_gt += 0.5
        traj_gt = denormalize(traj_gt, target, intrinsics, max_depth)
        
        for i, (r, results) in enumerate(ratio_preds.items()):
            num_obs = torch.floor(num_full * float(r)).to(torch.long)
            preds = results['traj'].cpu().numpy()
            gts = traj_gt[num_obs:num_full, :]
            # denormalize
            if centralize:
                preds += 0.5
            preds = denormalize(preds, target, intrinsics, max_depth)
            # get uncertainty if any
            if return_unct:
                logvars = results['unct'].cpu().numpy() if 'unct' in results else np.ones_like(preds)
                uncts = np.exp(logvars)
            
            # gather results
            if r in all_preds and r in all_gts:
                all_preds[r].append(preds)
                all_gts[r].append(gts)
                if return_unct: all_uncts[r].append(uncts)
            else:
                all_preds[r] = [preds]
                all_gts[r] = [gts]
                if return_unct: all_uncts[r] = [uncts]
    if return_unct: 
        return all_preds, all_uncts, all_gts
    return all_preds, all_gts


def XYZ_to_uv(traj3d, intrinsics):
    """ traj3d: (T, 3), a list of 3D (X, Y,Z) points 
    """
    # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['ox'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['oy'])
    # clip the coordinates 
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['w'])
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['h'])
    traj2d = np.floor(traj2d).astype(np.int32)
    return traj2d


def compute_cle(all_preds, all_gts):
    """Compute the Center Location Error (CLE)"""
    all_cles = dict()
    for r in list(all_gts.keys()):
        average_cle = np.mean(np.sum((all_preds[r] - all_gts[r])**2, axis=-1))
        all_cles[r] = average_cle
    
    # compute the mean of results from 9 observation ratios
    mCLE = np.mean(list(all_cles.values()))
    
    return all_cles, mCLE


def normalize_2d(uv, intrinsics):
    uv_norm = np.copy(uv).astype(np.float32)
    uv_norm[:, 0] /= intrinsics['w']
    uv_norm[:, 1] /= intrinsics['h']
    return uv_norm


def compute_displacement_errors(all_preds, all_gts, target='3d', eval_space='3d', intrinsics=None):
    """Compute the Displacement Errors (ADE and FDE)"""
    all_ades, all_fdes = dict(), dict()
    for r in list(all_gts.keys()):
        preds, gts = all_preds[r], all_gts[r]
        
        if target == '3d' and eval_space == '2d':
            preds = XYZ_to_uv(preds, intrinsics)
            gts = XYZ_to_uv(gts, intrinsics)
        
        if target == '3d' and eval_space == 'norm2d':
            preds = normalize_2d(XYZ_to_uv(preds, intrinsics), intrinsics)
            gts = normalize_2d(XYZ_to_uv(gts, intrinsics), intrinsics)
        
        if target == '2d' and eval_space == 'norm2d':
            preds = normalize_2d(preds, intrinsics)
            gts = normalize_2d(gts, intrinsics)

        displace_errors = np.sqrt(np.sum((preds - gts)**2, axis=-1))
        # ADE score
        all_ades[r] = np.mean(displace_errors)
        # FDE score
        all_fdes[r] = displace_errors[-1]
    
    return all_ades, all_fdes


def compute_block_distances(all_preds, all_gts, target='3d', eval_space='3d', intrinsics=None):
    """Compute the block distances along x, y, and z dimensions"""
    all_dxs, all_dys, all_dzs = dict(), dict(), dict()
    for r in list(all_gts.keys()):
        preds, gts = all_preds[r], all_gts[r]
        
        if target == '3d' and eval_space == '2d':
            preds = XYZ_to_uv(preds, intrinsics)
            gts = XYZ_to_uv(gts, intrinsics)
        
        if target == '3d' and eval_space == 'norm2d':
            preds = normalize_2d(XYZ_to_uv(preds, intrinsics), intrinsics)
            gts = normalize_2d(XYZ_to_uv(gts, intrinsics), intrinsics)
        
        if target == '2d' and eval_space == 'norm2d':
            preds = normalize_2d(preds, intrinsics)
            gts = normalize_2d(gts, intrinsics)
        
        # delta X
        all_dxs[r] = np.mean(np.fabs(preds[:, 0] - gts[:, 0]))
        # delta Y
        all_dys[r] = np.mean(np.fabs(preds[:, 1] - gts[:, 1]))
        if preds.shape[-1] == 3:
            # delta Z
            all_dzs[r] = np.mean(np.fabs(preds[:, 2] - gts[:, 2]))
    return all_dxs, all_dys, all_dzs


def first_last_nonzero_indices(mask, dim=1):
    """ Find the first and the last non-zero indices 
        mask: (B, T)
    """
    assert mask.sum(dim=dim).all(), "mask should not contain all-zero rows!"
    mask_cumsum = mask.cumsum(dim=dim)
    first_nz = (mask_cumsum == 0).sum(dim=dim)
    r2l_cumsum = mask + torch.sum(mask, dim=dim, keepdims=True) - torch.cumsum(mask, dim=dim)
    last_nz = mask.size(dim) - (r2l_cumsum == 0).sum(dim=dim) - 1
    return first_nz, last_nz


def get_subsequent_mask(seq_len, device, diagonal=1):
    """ Get the sequential mask for transformer decoder
        seq_len: int
        return: (1, seq_len, seq_len) bottom-triangular bool matrix
    """
    mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=diagonal)).bool()
    return mask


def random_ratios_batch(nframes, cfg, min_obs=2):
    """ Generate a batch of random ratios
    """
    if hasattr(cfg, 'ratios'):  # fixed ratio list
        ratios_batch = torch.tensor([cfg.ratios], dtype=torch.float32).repeat(nframes.size(0), 1)
        return ratios_batch
    # random selection 
    assert cfg.num_ratios <= 9 and cfg.num_ratios >= 1, "invalid number of ratios!"
    ratios_batch = torch.zeros((nframes.size(0), cfg.num_ratios), dtype=torch.float32)
    for b, len_valid in enumerate(nframes):
        ratios = []
        while len(ratios) != cfg.num_ratios:
            r = np.random.choice(np.arange(1, 10)/10, 1, replace=False)[0]
            if r * int(len_valid) >= min_obs and r not in ratios:  # ignore ratios that produce too short clips
                ratios.append(r)
        ratios_batch[b] = torch.tensor(ratios, dtype=torch.float32)
    return ratios_batch


def get_test_ratios(test_ratios, nframes):
    """ test_ratios: list of decimal numbers
        nframes: (B,)
    """
    valid_ratios = [r for r in test_ratios if all(nframes * r >= 1)]
    ratios = torch.tensor([valid_ratios]).repeat(nframes.size(0), 1)  # (B, M)
    return ratios


def save_the_latest(data, ckpt_file, topK=3, ignores=[]):
    """ Only keeping the latest topK checkpoints.
    """
    # find the existing checkpoints in a sorted list
    folder = os.path.dirname(ckpt_file)
    num_exist = len(os.listdir(folder))
    if num_exist >= topK + len(ignores):
        # remove the old checkpoints
        ext = ckpt_file.split('.')[-1]
        all_ckpts = list(filter(lambda x: x.endswith('.' + ext), os.listdir(folder)))
        all_epochs = [int(filename.split('.')[-2].split('_')[-1]) for filename in all_ckpts]
        fids = np.argsort(all_epochs)
        # iteratively remove
        for i in fids[:(num_exist - topK + 1)]:
            if all_epochs[i] in ignores:
                continue
            file_to_remove = os.path.join(folder, all_ckpts[i])
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)
    torch.save(data, ckpt_file)


def read_video(video_file, toRGB=True):
    assert os.path.exists(video_file), "File does not exist! {}".format(video_file)
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        if toRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def read_odometry(odom_file):
    assert os.path.exists(odom_file), "File does not exist! {}".format(odom_file)
    all_transforms = np.load(odom_file) 
    return all_transforms


def len_quad(max_len, i):
    return i * max_len - int(i * (i - 1) / 2)


def write_video(mat, video_file, fps=5):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)

def vis_frame_traj(frame, traj2d):
    cv2.polylines(frame, [traj2d], False, (0, 255, 255))
    for t, uv in enumerate(traj2d):
        cv2.circle(frame, uv, radius=5, color=(0, 0, 255), thickness=-1)

def video_to_gif(video, giffile, fps=5.0, toBGR=False):
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=1.0/fps) as writer:
        for frame in video:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if toBGR else np.copy(frame)
            writer.append_data(frame_vis)


def global_to_local(traj3d, odometry, refer_idx=0, start_idx=1):
    """ Transform 3D trajectory of global frame to local frame
        traj3d: (Tu, 3)
    """
    assert start_idx >= refer_idx
    length = traj3d.shape[0]
    traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))
    traj3d_local = []
    for i, pt_homo in enumerate(traj3d_homo):
        # collect transformation matrices from the first frame to the current fram
        trans = reduce(np.dot, odometry[refer_idx: i + start_idx - refer_idx])
        trans_inv = np.dot(np.linalg.inv(np.dot(trans.T, trans)), trans.T)  # (A^TxA)^-1xA, (4, 4)
        pt_local = trans_inv.dot(pt_homo)[:3]
        traj3d_local.append(pt_local)
    traj3d_local = np.array(traj3d_local)
    return traj3d_local


def transform_to_local(traj3d_global, odometry, num_obs=0):
    traj3d_local = []
    for i, loc3d in enumerate(traj3d_global):
        t = num_obs + i
        # camera_0 to camera_t (Note: all T-t points are in the same camera_0 ,thus their odometry are ignored by setting to identity matrix)
        odom_new = np.copy(odometry)
        odom_new[(t+1):] = np.repeat(np.eye(4)[None, :, :], odometry.shape[0]-t-1, axis=0)
        # global to local transform
        loc3d_local = global_to_local(loc3d[None, :], odom_new, start_idx=t+1)
        traj3d_local.append(loc3d_local)
    traj3d_local = np.concatenate(traj3d_local, axis=0)
    return traj3d_local


def normalize(traj, target='3d', intrinsics=None, vis_ratio=0.25, max_depth=3.0):
    width = intrinsics['w'] * vis_ratio
    height = intrinsics['h'] * vis_ratio
    if target == '2d':
        traj[:, 0] /= width
        traj[:, 1] /= height
    elif target == '3d':
        # transform the (X, Y, Z) into the normalized (u,v,z) by PINHOLE camera model
        traj[:, 0] = (traj[:, 0] * intrinsics['fx'] / traj[:, 2] + intrinsics['ox']) / intrinsics['w']
        traj[:, 1] = (traj[:, 1] * intrinsics['fy'] / traj[:, 2] + intrinsics['oy']) / intrinsics['h']
        traj[:, 2] = traj[:, 2] / max_depth
    traj[:, 0] = np.clip(traj[:, 0], a_min=0, a_max=(width-1)/width)
    traj[:, 1] = np.clip(traj[:, 1], a_min=0, a_max=(height-1)/height)
    return traj


def denormalize(traj, target='3d', intrinsics=None, max_depth=3.0):
    """ transform the normalized (u,v,depth) to (X,Y,Z)
    """
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


def output_transform(outputs, trajectories, nframes, odometry, 
        intrinsics=None, vis_ratio=0.25, max_depth=3.0, centralize=False, use_odom=False, ignore_depth=False):
    """ outputs: predictions of observed and unobserved (torch cuda tensor)
        trajectories: (B, T, 3), ndarray, normalized
    """
    outputs_unobs, outputs_obs = deepcopy(outputs[0]), deepcopy(outputs[1])
    trajectory_local = np.copy(trajectories)
    if centralize:
        trajectory_local += 0.5
    for b, ratio_preds in enumerate(outputs[0]):
        # get the number observed frames and full frames
        num_full = int(nframes[b])
        traj_gt = trajectory_local[b, :num_full]
        
        for i, (r, pred_dict) in enumerate(ratio_preds.items()):
            num_obs = int(num_full * float(r))
            preds = pred_dict['traj'].cpu().numpy()
            if centralize:
                preds += 0.5
            if ignore_depth and traj_gt.shape[1] == 3:
                preds = np.hstack((preds, traj_gt[num_obs:, 2:3]))  # (Tu, 3), compensate the depth for prediction
            
            # global to local
            if use_odom:
                # de-normalize & 2D-3D
                preds = denormalize(preds, target='3d', intrinsics=intrinsics, max_depth=max_depth)
                # transform from global (camera_0) to local (camera_t) in 3D space
                preds = transform_to_local(preds, odometry[b, :num_full], num_obs)
                # normalize back
                preds = normalize(preds, target='3d', intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth)
            outputs_unobs[b][r]['traj'] = preds

            if outputs_obs is not None:
                preds_obs = outputs_obs[b][r]['traj'].cpu().numpy()
                if centralize:
                    preds_obs += 0.5
                if ignore_depth and traj_gt.shape[1] == 3:
                    preds_obs = np.hstack((preds_obs, traj_gt[:num_obs, 2:3]))
                if use_odom:
                    preds_obs = denormalize(preds_obs, target='3d', intrinsics=intrinsics, max_depth=max_depth)
                    preds_obs = transform_to_local(preds_obs, odometry[b, :num_full])
                    preds_obs = normalize(preds_obs, target='3d', intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth)
                outputs_obs[b][r]['traj'] = preds_obs

        if use_odom:
            traj_gt = denormalize(traj_gt, target='3d', intrinsics=intrinsics, max_depth=max_depth)
            traj_gt = transform_to_local(traj_gt, odometry[b, :num_full])
            trajectory_local[b, :num_full] = normalize(traj_gt, target='3d', intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth)

    return (outputs_unobs, outputs_obs), trajectory_local


def draw_trajectory_3d_with_gt(pred3d, gt3d, video_size=[270, 480], \
    ratio=0.5, fps=5, dpi=30, views=[30, -120], minmax=None, savefile='temp.mp4'):

    if minmax is None:
        all_points = np.concatenate([pred3d, gt3d], axis=0)
        xmin, ymin, zmin = np.min(all_points, axis=0).tolist()
        xmax, ymax, zmax = np.max(all_points, axis=0).tolist()
    else:
        xmin, ymin, zmin = minmax[0]
        xmax, ymax, zmax = minmax[1]
    fig_w, fig_h = int(video_size[1] // dpi), int(video_size[0] // dpi)

    # init a 3D figure
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(views[0], views[1])  # elev: 30, azim: -120
    # Make a 3D quiver plot
    ox, oy, oz = np.zeros((3,3))
    dx, dy, dz = np.array([[xmax-xmin, 0, 0], [0, zmax-zmin, 0], [0, 0, ymax-ymin]]) * 0.2
    ax.quiver(ox, oy, oz, dx, dy, dz)
    # Setup coordinate system
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(ymin, ymax)
    ax.invert_zaxis()
    ax.set_xlabel('X')  # x: rightward
    ax.set_ylabel('Z')  # z: innerward
    ax.set_zlabel('Y')  # y: downward
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=5)
    plt.tight_layout()

    # init video writer
    writer = FFMpegWriter(fps=fps, metadata=dict(title='3D Trajectory', artist='Matplotlib'))
    future_start = int(len(pred3d) * ratio)
    # process
    with writer.saving(fig, savefile, dpi=dpi):
        pred_past, pred_future = [], []
        gt_past, gt_future = [], []
        for t, (pred, gt) in enumerate(zip(pred3d, gt3d)):
            if t <= future_start:
                pred_past.append(pred)
                gt_past.append(gt)
            else:
                pred_future.append(pred)
                gt_future.append(gt)
            
            if t == future_start:  # connect the future and past
                pred_future.append(pred)
                gt_future.append(gt)

            # draw past ground truth trajectory
            xs, ys, zs = np.array(gt_past)[:, 0], np.array(gt_past)[:, 1], np.array(gt_past)[:, 2]
            ax.plot(xs, zs, ys, '-o', mfc='green', c='blue', fillstyle='full', mec='green', mew=0.0)
            # draw past predicted trajectory
            xs, ys, zs = np.array(pred_past)[:, 0], np.array(pred_past)[:, 1], np.array(pred_past)[:, 2]
            ax.plot(xs, zs, ys, '-o', mfc='blue', c='blue', fillstyle='full', mec='blue', mew=0.0)

            if t > future_start:
                # draw future ground truth trajectory
                xs, ys, zs = np.array(gt_future)[:, 0], np.array(gt_future)[:, 1], np.array(gt_future)[:, 2]
                ax.plot(xs, zs, ys, '-o', mfc='limegreen', c='limegreen', fillstyle='full', mec='limegreen', mew=0.0)
                # draw future predicted trajectory
                xs, ys, zs = np.array(pred_future)[:, 0], np.array(pred_future)[:, 1], np.array(pred_future)[:, 2]
                ax.plot(xs, zs, ys, '-o', mfc='red', c='red', fillstyle='full', mec='red', mew=0.0)
            # grab a frame
            writer.grab_frame()
    plt.close()
