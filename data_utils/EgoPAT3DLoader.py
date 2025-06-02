"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file contains the EgoPAT3D dataset loader for hand trajectory prediction,
including data loading, preprocessing and augmentation for multimodal inputs.
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os, re
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from functools import reduce
from data_utils.utils import denormalize, global_to_local, normalize, vis_frame_traj, write_video, video_to_gif, read_video
import open3d as o3d
import copy

def homography_to_displacements(H, img_width, img_height):
    corners = np.array([[0, 0, 1],
                        [img_width, 0, 1],
                        [img_width, img_height, 1],
                        [0, img_height, 1]])
    corners = corners[np.newaxis, :, :].repeat(H.shape[0], axis=0)
    transformed_corners = np.einsum('ijk,ikl->ijl', H, corners.transpose((0, 2, 1)))
    transformed_corners /= transformed_corners[:, 2:3, :]
    displacements = transformed_corners[:, :2, :].transpose((0, 2, 1)) - corners[:, :, :2]
    displacements[:,:,0] = displacements[:,:,0] / img_width
    displacements[:,:,1] = displacements[:,:,1] / img_height
    displacements = displacements.reshape(displacements.shape[0], 8)
    return displacements


class EgoPAT3D(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, data_cfg=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self._MAX_FRAMES = data_cfg.max_frames
        self._MAX_DEPTH = data_cfg.max_depth
        self.target = data_cfg.target
        self.scenes = data_cfg.scenes
        self.modalities = data_cfg.modalities
        self.tinyset = data_cfg.tinyset
        self.load_all = data_cfg.load_all
        self.use_odom = data_cfg.use_odom
        if self.use_odom: assert self.target == '3d', "Do not support 2D target when using odometry!"
        self.centralize = data_cfg.centralize
        self.glip_feats_path = data_cfg.glip_feats_path
        self.motion_feats_path = data_cfg.motion_feats_path
        self.voxel_path = data_cfg.voxel_path
        self.split_ratio = data_cfg.ratio
        self.res = data_cfg.voxel_res
        self.grid_size = data_cfg.grid_size
        self.origin_xyz = data_cfg.origin_xyz

        # Our own dataset split
        self.scene_splits = {'train': {'1': ['1', '2', '3', '4', '5', '6', '7'], 
                                       '2': ['1', '2', '3', '4', '5', '6', '7'], 
                                       '3': ['1', '2', '3', '4', '5', '6'],
                                       '4': ['1', '2', '3', '4', '5', '6', '7'],
                                       '5': ['1', '2', '3', '4', '5', '6'], 
                                       '6': ['1', '2', '3', '4', '5', '6'], 
                                       '7': ['1', '2', '3', '4', '5', '6', '7'],
                                       '9': ['1', '2', '3', '4', '5', '6', '7'],
                                       '10': ['1', '2', '3', '4', '5', '6', '7'],
                                       '11': ['1', '2', '3', '4', '5', '6', '7'],
                                       '12': ['1', '2', '3', '4', '5', '6', '7']
                                        },
                             'val': {'1': ['8'], 
                                     '2': ['8'], '3': ['7'], '4': ['8'], '5': ['7'], '6': ['7'], 
                                     '7': ['8'], '9': ['8'], '10': ['8'], '11': ['8'], '12': ['8']
                                     },
                             'test': {'1': ['9', '10'], 
                                      '2': ['9', '10'], '3': ['9', '10'], '4': ['9', '10'], '5': ['8', '9'], '6': ['8', '9'], 
                                      '7': ['9', '10'], '9': ['9', '10'], '10': ['9', '10'], '11': ['9', '10'], '12': ['9', '10']
                                     },
                             'test_novel': {'13': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                                            '14': ['2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                                            '15': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}}
        assert self.phase in self.scene_splits, "Invalid dataset split: {}".format(self.phase)
        if self.scenes is not None:
            data = dict(filter(lambda elem: elem[0] in self.scenes, self.scene_splits[self.phase].items()))
            self.scene_splits[self.phase] = data
        # camera intrinsics
        self.intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                           'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                           'w': 3840, 'h': 2160}
        self.vis_ratio = 0.25
        
        self.rgb_dir, self.traj_dir, self.odom_dir = self._init_data_path()
        if self.tinyset:
            selects = self._read_selected_list()
            samples = selects[:64] if self.phase == 'train' else selects[64:]
            self.rgb_paths, self.traj_data, self.odom_data, self.preserves = self._read_selected_data(samples)
        else:    
            # read dataset lists
            self.rgb_paths, self.traj_data, self.odom_data, self.preserves = self._read_list()

        if self.load_all and 'rgb' in self.modalities:
            self.rgb_data = self._read_all_videos()
        
        
    def _read_traj_file(self, filepath):
        assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        traj2d = data['traj2d']
        traj3d = data['traj3d']
        num_preserve = data['num_preserve'] if 'num_preserve' in data else len(traj2d)
        return traj2d, traj3d, num_preserve


    def _read_odom_file(self, filepath):
        assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
        all_transforms = np.load(filepath)
        return all_transforms

    
    def _init_data_path(self):
        # rgb video root
        rgb_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'video_clips_hand')
        assert os.path.exists(rgb_dir), 'Path does not exist! {}'.format(rgb_dir)
        # trajectory root
        traj_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'trajectory_repair')
        assert os.path.exists(traj_dir), 'Path does not exist! {}'.format(traj_dir)
        # visual odometry path (used only if self.use_odom=True)
        odom_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'odometry')
        assert os.path.exists(odom_dir), 'Path does not exist! {}'.format(traj_dir)
        return rgb_dir, traj_dir, odom_dir


    def _read_selected_list(self):
        select_listfile = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'selected.txt')
        assert os.path.exists(select_listfile), 'Path does not exist! {}'.format(select_listfile)
        selects = []
        with open(select_listfile, 'r') as f:
            for line in f.readlines():
                selects.append(line.strip())
        return selects
    
    
    def _read_list(self):
        rgb_paths, traj_data, odom_data, preserves = [], [], [], []
        for scene_id, record_splits in tqdm(self.scene_splits[self.phase].items(), ncols=0, desc='\u2022 Read trajectories'):
            scene_rgb_path = os.path.join(self.rgb_dir, scene_id)
            scene_traj_path = os.path.join(self.traj_dir, scene_id)
            scene_odom_path = os.path.join(self.odom_dir, scene_id)
            
            record_names = list(filter(lambda x: x.split('_')[-1] in record_splits, os.listdir(scene_traj_path)))
            for record in record_names:
                record_rgb_path = os.path.join(scene_rgb_path, record)
                record_traj_path = os.path.join(scene_traj_path, record)
                record_odom_path = os.path.join(scene_odom_path, record)
                
                traj_files = list(filter(lambda x: x.endswith('.pkl'), os.listdir(record_traj_path)))
                for traj_name in traj_files:
                    traj2d, traj3d, num_preserve = self._read_traj_file(os.path.join(record_traj_path, traj_name))
                    traj_data.append({'traj2d': traj2d, 'traj3d': traj3d})
                    rgb_paths.append(os.path.join(record_rgb_path, traj_name[:-4] + '.mp4'))
                    if self.use_odom:
                        odom = self._read_odom_file(os.path.join(record_odom_path, traj_name[:-4] + '.npy'))
                    else:
                        print("!please use odom!")
                        odom = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_preserve, axis=0)
                    odom_data.append(odom)
                    preserves.append(num_preserve)

        return rgb_paths, traj_data, odom_data, preserves


    def _read_selected_data(self, sample_list):
        rgb_paths, traj_data, odom_data, preserves = [], [], [], []
        for sample in sample_list:
            scene_id, record, clip_name = sample.split('/')
            traj2d, traj3d, num_preserve = self._read_traj_file(os.path.join(self.traj_dir, scene_id, record, clip_name + '.pkl'))
            traj_data.append({'traj2d': traj2d, 'traj3d': traj3d})
            rgb_paths.append(os.path.join(self.rgb_dir, scene_id, record, clip_name + '.mp4'))
            if True:
                odom = self._read_odom_file(os.path.join(self.odom_dir, scene_id, record, clip_name + '.npy'))
            else:
                odom = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_preserve, axis=0)
            odom_data.append(odom)
            preserves.append(num_preserve)

        return rgb_paths, traj_data, odom_data, preserves
    
    
    def _read_all_videos(self):
        video_data = []
        for filename, preserve in zip(self.rgb_paths, self.preserves):
            rgb = self._read_rgb(filename)
            rgb = rgb[:preserve]
            if self.transform:
                rgb = self.transform(rgb)
            else:
                rgb = torch.from_numpy(np.transpose(rgb, (3, 0, 1, 2)))
            video_data.append(rgb)
        return video_data


    def _read_rgb(self, video_file):
        cap = cv2.VideoCapture(video_file)
        success, frame = cap.read()
        videos = []
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            videos.append(frame)
            success, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        videos = np.array(videos)
        return videos
    

    def _XYZ_to_uv(self, traj3d):
        width = self.intrinsics['w'] * self.vis_ratio
        height = self.intrinsics['h'] * self.vis_ratio
        u = (traj3d[:, 0] * self.intrinsics['fx'] / traj3d[:, 2] + self.intrinsics['ox']) * self.vis_ratio
        v = (traj3d[:, 1] * self.intrinsics['fy'] / traj3d[:, 2] + self.intrinsics['oy']) * self.vis_ratio
        u = torch.clamp(u, min=0, max=width-1)
        v = torch.clamp(v, min=0, max=height-1)
        traj2d = torch.stack((u, v), dim=-1)
        return traj2d
    
    
    def _normalize(self, traj):
        traj_new = traj.clone()
        width = self.intrinsics['w'] * self.vis_ratio
        height = self.intrinsics['h'] * self.vis_ratio
        if self.target == '2d':
            traj_new[:, 0] /= width
            traj_new[:, 1] /= height
        elif self.target == '3d':
            traj2d = self._XYZ_to_uv(traj)
            traj_new[:, 0] = traj2d[:, 0] / width
            traj_new[:, 1] = traj2d[:, 1] / height
            traj_new[:, 2] = traj[:, 2] / self._MAX_DEPTH
        if self.centralize:
            traj_new -= 0.5
        return traj_new

    
    def _get_projected_traj3d(self, traj3d, odometry):
        length = len(odometry)
        traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))
        all_traj3d_proj = []
        for i in range(length):
            traj3d_proj = [traj3d[i]]
            for j in range(i + 1, length):
                odom = reduce(np.dot, odometry[(i+1):(j+1)])
                future_point = odom.dot(traj3d_homo[j].T)
                traj3d_proj.append(future_point[:3])
            all_traj3d_proj.append(np.array(traj3d_proj))
        all_traj3d_proj = np.concatenate(all_traj3d_proj, axis=0)
        return all_traj3d_proj
    
    
    def __len__(self):
        return len(self.rgb_paths)
    
    
    def __getitem__(self, index):
        traj2d = torch.from_numpy(self.traj_data[index]['traj2d']).to(torch.float32)
        traj3d = torch.from_numpy(self.traj_data[index]['traj3d']).to(torch.float32)
        traj = traj2d if self.target == '2d' else traj3d
        num_preserve = self.preserves[index]
        if 'rgb' in self.modalities:
            if self.load_all:
                rgb = self.rgb_data[index]
            else:
                rgb = self._read_rgb(self.rgb_paths[index])
                rgb = rgb[:num_preserve]
                if self.transform:
                    rgb = self.transform(rgb)
                else:
                    rgb = torch.from_numpy(np.transpose(rgb, (3, 0, 1, 2)))
            assert rgb.shape[1] == traj.shape[0], "Trajectory is inconsistent with RGB video!"
        
        # hard path for now
        rgb2glip_path_name = self.rgb_paths[index]
        rgb2glip_path_name = rgb2glip_path_name.replace('/', '__')[:-4] + ".npy"
        rgb2glip_path_name = rgb2glip_path_name.replace('__data__HTPdata__EgoPAT3D-postproc__video_clips_hand', '__mnt__share_disk__mjy3__EgoPAT3D-postproc__video_clips_hand')
        glip_feats = np.load(os.path.join(self.glip_feats_path, rgb2glip_path_name))
        glip_feats = glip_feats[:num_preserve]
        assert glip_feats.shape[0] == traj.shape[0]

        # hard path for now
        motion_feats = np.load(os.path.join(self.motion_feats_path, rgb2glip_path_name.replace('mjy3', 'mjy')))
        motion_feats = motion_feats[:num_preserve]
        assert motion_feats.shape[0] == traj.shape[0]

        # clipping or padding with zero
        max_frames = traj.size(0) if self._MAX_FRAMES < 0 else self._MAX_FRAMES
        len_valid = min(traj.size(0), max_frames)

        if self.phase != 'train':
            num_obs = np.floor(len_valid * self.split_ratio).astype(int)
            motion_feats[num_obs:len_valid] = motion_feats[num_obs-1]
        
        input_data = torch.tensor([])  # a placeholder (not used)
        if 'rgb' in self.modalities:
            input_data = torch.zeros([rgb.size(0), max_frames, rgb.size(2), rgb.size(3)], dtype=rgb.dtype)
            input_data[:, :len_valid] = rgb[:, :len_valid]
        
        glip_feats_filled = np.zeros([max_frames, *glip_feats.shape[1:]], dtype=np.float32)
        glip_feats_filled[:len_valid] = glip_feats[:len_valid]
        motion_feats_filled = np.zeros([max_frames, 3, 3], dtype=np.float32)
        motion_feats_filled[:len_valid] = motion_feats[:len_valid]
        
        len_pad, len_real = max_frames, len_valid
        odometry = torch.eye(4)[None, :, :].repeat(len_pad, 1, 1).to(torch.float32)
        output_traj = torch.zeros([len_pad, traj.size(1)], dtype=traj.dtype)
        traj_valid = traj[:len_valid]
        if self.use_odom:
            odom = self.odom_data[index][:num_preserve]
            odom_valid = odom[:len_valid]
            traj3d_valid_all = self._get_projected_traj3d(traj3d[:len_valid].numpy(), odom_valid)
            traj3d_valid = torch.from_numpy(traj3d_valid_all[:len_real])
            traj_valid_this_scheme = self._XYZ_to_uv(traj3d_valid) if self.target == '2d' else traj3d_valid

        odom_for_motion_feats = self.odom_data[index][:num_preserve]
        odom_for_motion_feats_valid = odom_for_motion_feats[:len_valid]
        odometry[:len_valid] = torch.from_numpy(odom_for_motion_feats_valid)
        if self.phase != 'train':
            num_obs = np.floor(len_valid * self.split_ratio).astype(int)
            odometry[num_obs:len_valid] = odometry[num_obs-1]
        
        traj_norm_valid = self._normalize(traj_valid_this_scheme)
        output_traj[:len_real] = traj_norm_valid

        voxel_grid_list = []
        rgb2glip_path_name_raw = self.rgb_paths[index]
        matches = re.search(r"s(\d+)_e(\d+)", rgb2glip_path_name_raw)
        start_index_pc = int(matches.group(1))
        end_index_pc = int(matches.group(2))

        pointcloud_path_name_raw = rgb2glip_path_name_raw.replace('EgoPAT3D-postproc/video_clips_hand', 'egopat_pointcloud_filtered')
        scene_name_raw = pointcloud_path_name_raw.split('/')[5]
        pointcloud_path_name_raw = pointcloud_path_name_raw.replace(scene_name_raw, 'pointcloud_'+scene_name_raw)
        voxel_grid = torch.zeros((self.grid_size, self.grid_size, self.grid_size, 4))
        save_path_voxel=os.path.join(self.voxel_path, pointcloud_path_name_raw.replace('/','__').replace("__data__HTPdata__","__mnt__share_disk__mjy__"))
        
        if os.path.exists(save_path_voxel+'.npz'):
            saved_voxel = np.load(save_path_voxel+'.npz', allow_pickle=True)['arr_0']
            transformed_points_with_colors = saved_voxel
            ones_column = torch.ones((transformed_points_with_colors.shape[0], 1))
            transformed_points_with_colors = torch.from_numpy(transformed_points_with_colors)
            voxel_grid[torch.round(transformed_points_with_colors[:,0]).long(), torch.round(transformed_points_with_colors[:,1]).long(),torch.round(transformed_points_with_colors[:,2]).long()] = torch.cat((ones_column, transformed_points_with_colors[:, -3:]),dim=1).to(torch.float32)
        else:
            odom_tmp_list = []
            point_cloud_arr_all = []
            point_cloud_arr_all_transformed = []
            traj_valid_transformed = []
            num_obs_end = np.floor(len_valid * self.split_ratio).astype(int)
            for cur_index in range(start_index_pc, start_index_pc+len_valid):
                cur_pointcloud_path_name = os.path.join(*(pointcloud_path_name_raw.split('/')[:-1]))
                cur_pointcloud_path_name = "/"+os.path.join(cur_pointcloud_path_name, str(cur_index)) + ".ply"

                if not os.path.exists(cur_pointcloud_path_name):
                    cur_pointcloud_path_name = cur_pointcloud_path_name.replace('egopat_pointcloud_filtered', 'egopat_pointcloud')

                point_cloud = o3d.io.read_point_cloud(cur_pointcloud_path_name)
                assert np.asarray(point_cloud.points).shape[0] != 0

                points_revise = np.asarray(point_cloud.points)
                points_revise[:, 1] = -points_revise[:, 1]
                points_revise[:, 2] = -points_revise[:, 2]
                point_cloud.points = o3d.utility.Vector3dVector(points_revise)

                if cur_index == start_index_pc:
                    point_cloud_arr_all.append(point_cloud)
                    point_cloud_arr_all_transformed.append(point_cloud)
                    odom_tmp_list.append(np.eye(4))
                    traj_valid_transformed.append(traj_valid[0:1])
                else:
                    transformation = np.eye(4)
                    reg = o3d.pipelines.registration.registration_icp(
                    point_cloud, point_cloud_arr_all[-1], 0.03, transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    odom_now = reg.transformation
                    odom_tmp_list.append(odom_now)
                    point_cloud_new =copy.deepcopy(point_cloud) 
                    odom_tmp_list = odom_tmp_list[::-1]
                    cur_traj_point = traj_valid[cur_index-start_index_pc].clone()
                    cur_traj_point[1:] = -1.0 * cur_traj_point[1:] 
                    point_cloud_traj = np.expand_dims(cur_traj_point, axis=0)
                    point_cloud_traj_o3d = o3d.geometry.PointCloud()
                    point_cloud_traj_o3d.points = o3d.utility.Vector3dVector(point_cloud_traj)
                    point_cloud_traj_o3d_new =copy.deepcopy(point_cloud_traj_o3d) 
                    for j in range(len(odom_tmp_list)):
                        point_cloud_new = point_cloud_new.transform(odom_tmp_list[j])
                        point_cloud_traj_o3d_new = point_cloud_traj_o3d_new.transform(odom_tmp_list[j])

                    point_cloud_arr_all.append(point_cloud)
                    traj_valid_transformed.append(np.asarray(point_cloud_traj_o3d_new.points))

                    if cur_index < (start_index_pc+num_obs_end):
                        point_cloud_arr_all_transformed.append(point_cloud_new)

            traj_valid_transformed = np.concatenate(traj_valid_transformed, axis=0)
            traj_valid_transformed[:, 1:] = -1.0 * traj_valid_transformed[:, 1:]
            np.save(save_path_traj, traj_valid_transformed)
            assert len_valid==traj_valid_transformed.shape[0]


            merged_pcd = point_cloud_arr_all[0]
            for pcd in point_cloud_arr_all_transformed[1:]:
                merged_pcd += pcd
            transformed_points = np.asarray(merged_pcd.points)
            transformed_points[:,1] = -transformed_points[:,1]
            transformed_points[:,2] = -transformed_points[:,2]
            transformed_colors = np.zeros_like(transformed_points)
            transformed_points_with_colors = np.concatenate((transformed_points, transformed_colors), axis=1)

            coord_origin_point = np.tile(np.array(self.origin_xyz), (transformed_points_with_colors.shape[0], 1))
            transformed_points_with_colors[:, :3] = (transformed_points_with_colors[:, :3] - coord_origin_point)/self.res

            kept = np.all((transformed_points_with_colors[:, :3] >= 0) & (transformed_points_with_colors[:, :3] < self.grid_size-1), axis=1)
            transformed_points_with_colors = transformed_points_with_colors[kept]
            
            ones_column = torch.ones((transformed_points_with_colors.shape[0], 1))
            transformed_points_with_colors = torch.from_numpy(transformed_points_with_colors)
            voxel_grid[torch.round(transformed_points_with_colors[:,0]).long(), torch.round(transformed_points_with_colors[:,1]).long(),torch.round(transformed_points_with_colors[:,2]).long()] = torch.cat((ones_column, transformed_points_with_colors[:, -3:]),dim=1).to(torch.float32)
            np.savez(save_path_voxel, transformed_points_with_colors)

        coord_origin_point = np.tile(np.array(self.origin_xyz), (traj_valid.shape[0], 1))

        return self.rgb_paths[index], input_data, odometry, len_valid, output_traj, glip_feats_filled, motion_feats_filled, voxel_grid



def build_dataloaders(args, phase='train'):
    """Loading the dataset"""

    if phase == 'train':
        # train set
        trainset = EgoPAT3D(args.data_path, phase='train', transform=None, data_cfg=args)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # validation set
        valset = EgoPAT3D(args.data_path, phase='val', transform=None, data_cfg=args)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print("Number of train/val: {}/{}".format(len(trainset), len(valset)))
        return train_loader, val_loader
    else:
        # test set
        testset = EgoPAT3D(args.data_path, phase='test', transform=None, data_cfg=args)
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # test_novel set
        testnovel_set = EgoPAT3D(args.data_path, phase='test_novel', transform=None, data_cfg=args)
        testnovel_loader = DataLoader(testnovel_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return test_loader, testnovel_loader