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

from tqdm import tqdm
import numpy as np
from data_utils.utils import random_ratios_batch, get_test_ratios, send_to_gpu
import data_utils.utils as utils
import data_utils.utils_h2o as utils_h2o



def train_epoch(cfg, model, traindata_loader, criterion, optimizer, writer, epoch):

    model.train()
    loss = 0
    pbar = tqdm(total=len(traindata_loader), ncols=0, desc='train epoch {}/{}'.format(epoch + 1, cfg.TRAIN.epoch))
    for batch_id, batch_data in enumerate(traindata_loader):

        filename, clip, odometry, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # generate a batch of random observation ratios
        ratios = random_ratios_batch(nframes, cfg.TRAIN)

        # run inference
        outputs = model(clip, nframes, ratios, traj=traj_gt)

        # compute losses
        losses = criterion(outputs, nframes, traj_gt)
        loss += losses['total_loss']

        # backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # write the lr and losses
        iter_cur = epoch * len(traindata_loader) + batch_id
        writer.add_scalar('misc/lr', optimizer.state_dict()['param_groups'][0]['lr'], iter_cur)
        for k, v in losses.items():
            writer.add_scalars('train/{}'.format(k), {k: v}, iter_cur)  # draw loss curves in different figures
        
        pbar.set_postfix({"train loss": losses['total_loss'].item()})
        pbar.update()
    
    loss /= len(traindata_loader)
    pbar.close()

    return loss


def eval_epoch(cfg, model, valdata_loader, criterion, writer, epoch):
    
    model.eval()

    all_preds, all_gts = {}, {}
    pbar = tqdm(total=len(valdata_loader), ncols=0, desc='eval epoch {}'.format(epoch + 1))
    for batch_id, batch_data in enumerate(valdata_loader):
        
        _, clip, _, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # fixed set of observation ratios
        ratios = get_test_ratios(cfg.TEST.ratios, nframes)
        
        # run inference
        outputs = model(clip, nframes, ratios, traj=traj_gt)

        # compute losses
        losses = criterion(outputs, nframes, traj_gt)

        # write the lr and losses
        iter_cur = epoch * len(valdata_loader) + batch_id
        for k, v in losses.items():
            writer.add_scalars('test/{}'.format(k), {k: v}, iter_cur)

        # gather unobserved predictions and ground truths 
        preds, gts = utils.gather_eval_results(outputs[0], nframes, traj_gt, 
                                                ignore_depth=model.ignore_depth, centralize=model.centralize)
        if batch_id == 0:
            all_preds.update(preds)
            all_gts.update(gts)
        else:
            for r in list(preds.keys()):
                all_preds[r].extend(preds[r])
                all_gts[r].extend(gts[r])
        
        pbar.set_postfix({"val loss": losses['total_loss'].item()})
        pbar.update()
    
    for r in list(all_preds.keys()):
        all_preds[r] = np.vstack(all_preds[r])
        all_gts[r] = np.vstack(all_gts[r])

    pbar.close()
    
    return all_preds, all_gts


def eval_h2o_epoch(cfg, model, valdata_loader, criterion, writer, epoch):

    model.eval()

    all_preds, all_gts, all_poses = {}, {}, {}
    pbar = tqdm(total=len(valdata_loader), ncols=0, desc='eval epoch {}'.format(epoch + 1))
    for batch_id, batch_data in enumerate(valdata_loader):
        
        filename, clip, campose, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # fixed set of observation ratios
        ratios = get_test_ratios(cfg.TEST.ratios, nframes)
        
        # run inference
        outputs = model(clip, nframes, ratios, traj=traj_gt)

        # compute losses
        losses = criterion(outputs, nframes, traj_gt)

        # write the lr and losses
        iter_cur = epoch * len(valdata_loader) + batch_id
        for k, v in losses.items():
            writer.add_scalars('test/{}'.format(k), {k: v}, iter_cur)

        # gather unobserved predictions and ground truths 
        preds, gts, poses = utils_h2o.gather_eval_results(outputs[0], nframes, traj_gt, campose, ignore_depth=model.ignore_depth, 
                                                    use_global=cfg.MODEL.use_global, centralize=model.centralize)

        if batch_id == 0:
            all_preds.update(preds)
            all_gts.update(gts)
            all_poses.update(poses)
        else:
            for r in list(preds.keys()):
                all_preds[r].extend(preds[r])
                all_gts[r].extend(gts[r])
                all_poses[r].extend(poses[r])
        
        pbar.set_postfix({"val loss": losses['total_loss'].item()})
        pbar.update()
    
    for r in list(all_preds.keys()):
        all_preds[r] = np.vstack(all_preds[r])
        all_gts[r] = np.vstack(all_gts[r])
        all_poses[r] = np.concatenate(all_poses[r])
    
    pbar.close()

    return all_preds, all_gts, all_poses



def run_test(cfg, model, data_loader):

    all_preds, all_gts = {}, {}
    for batch_id, batch_data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Run testing'):
        
        filename, clip, odometry, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # fixed set of observation ratios
        ratios = get_test_ratios(cfg.TEST.ratios, nframes)
        
        # run inference
        outputs = model.inference(clip, nframes, ratios, traj=traj_gt)

        # gather unobserved predictions and ground truths 
        # Note: the preds and gts are in Global 3D Space for target == '3d, or in pixel space of the 1st frame
        preds, gts = utils.gather_eval_results(outputs[0], nframes, traj_gt, 
                                            ignore_depth=model.ignore_depth, centralize=model.centralize)
        if batch_id == 0:
            all_preds.update(preds)
            all_gts.update(gts)
        else:
            for r in list(preds.keys()):
                all_preds[r].extend(preds[r])
                all_gts[r].extend(gts[r])
    
    for r in list(all_preds.keys()):
        all_preds[r] = np.vstack(all_preds[r])
        all_gts[r] = np.vstack(all_gts[r])
        
    return all_preds, all_gts


def run_h2o_test(cfg, model, data_loader):

    all_preds, all_gts, all_poses = {}, {}, {}
    for batch_id, batch_data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Run testing'):
        
        filename, clip, campose, nframes, traj_gt = batch_data
        # send data to device 
        clip, traj_gt = send_to_gpu((clip, traj_gt), cfg.device, non_blocking=True)
        
        # fixed set of observation ratios
        ratios = get_test_ratios(cfg.TEST.ratios, nframes)
        
        # run inference
        outputs = model.inference(clip, nframes, ratios, traj=traj_gt)

        # gather unobserved predictions and ground truths 
        # Note: the preds and gts are in Global 3D Space for target == '3d, or in pixel space of the 1st frame
        preds, gts, poses = utils_h2o.gather_eval_results(outputs[0], nframes, traj_gt, campose, ignore_depth=model.ignore_depth, 
                                                    use_global=cfg.MODEL.use_global, centralize=model.centralize)
        if batch_id == 0:
            all_preds.update(preds)
            all_gts.update(gts)
            all_poses.update(poses)
        else:
            for r in list(preds.keys()):
                all_preds[r].extend(preds[r])
                all_gts[r].extend(gts[r])
                all_poses[r].extend(poses[r])
    
    for r in list(all_preds.keys()):
        all_preds[r] = np.vstack(all_preds[r])
        all_gts[r] = np.vstack(all_gts[r])
        all_poses[r] = np.concatenate(all_poses[r])
        
    return all_preds, all_gts, all_poses


def print_eval_results(writer, all_ades_val, all_fdes_val, all_ades_train=None, all_fdes_train=None, epoch=0, loss_train=0):
    mADE_val = np.mean(list(all_ades_val.values()))
    mFDE_val = np.mean(list(all_fdes_val.values()))
    writer.add_scalars('test/DE_val', {'mADE': mADE_val, 'mFDE': mFDE_val}, epoch)
    info = "==> [Epoch {}]: Loss(train) = {:.9f}, mADE(val) = {:.3f}, mFDE(val) = {:.3f}".format(epoch + 1, loss_train, mADE_val, mFDE_val)
    if all_ades_train and all_fdes_train:
        mADE_train = np.mean(list(all_ades_train.values()))
        mFDE_train = np.mean(list(all_fdes_train.values()))
        writer.add_scalars('test/DE_train', {'mADE': mADE_train, 'mFDE': mFDE_train}, epoch)
        info += ', mADE(train) = {:.3f}, mFDE(train)'.format(mADE_train, mFDE_train)
    print(info)
