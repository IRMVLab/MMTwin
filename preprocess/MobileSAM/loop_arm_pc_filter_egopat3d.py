from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import os
import pickle
import open3d as o3d
import re

# ====================================
with open('ms.yml', 'r') as f:
    config = yaml.safe_load(f)
video_all_dir = config['loop']['videos_root']
skip_parts = config['loop']['skip_parts']
# ====================================

# ====================================
# Model configuration
model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Camera intrinsics
camera_intrinsics = {
    'fx': config['camera']['intrinsics']['fx'],
    'fy': config['camera']['intrinsics']['fy'],
    'ox': config['camera']['intrinsics']['ox'],
    'oy': config['camera']['intrinsics']['oy'],
    'w': config['camera']['intrinsics']['width'],
    'h': config['camera']['intrinsics']['height']
}
# ====================================

# Load MobileSAM model
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# Create SAM predictor
predictor = SamPredictor(mobile_sam)

camera_matrix = np.array([
    [camera_intrinsics['fx'], 0, camera_intrinsics['ox']],
    [0, camera_intrinsics['fy'], camera_intrinsics['oy']],
    [0, 0, 1]
])

pattern = re.compile(r"s(\d+)_e(\d+)")
for root, dirs, files in os.walk(video_all_dir):
    for file in files:
        if file.endswith(".mp4"):
            match = pattern.search(file)
            s_value = int(match.group(1))
            e_value = int(match.group(2))
            video_path = os.path.join(root, file) 
            test_point_path = video_path.replace('video_clips_hand', 'trajectory_repair').replace('mp4', 'pkl')

            if not os.path.exists(test_point_path): 
                continue

            scene_id = root.split('/')[-1]
            pointcloud_dir = root.replace('EgoPAT3D-postproc/video_clips_hand', 'egopat_pointcloud').replace(scene_id, 'pointcloud_'+scene_id)
            
            save_pointcloud_dir = pointcloud_dir.replace('egopat_pointcloud', 'egopat_pointcloud_filtered')

            part_id = pointcloud_dir.split('/')[-2]
            if part_id in skip_parts:
                print("Skipping part ", part_id)
                continue
                
            os.makedirs(save_pointcloud_dir, exist_ok=True)

            # Read test point file (trajectory file)
            with open(test_point_path, 'rb') as f:
                data = pickle.load(f)
            traj2d = data['traj2d']

            # Open video file
            cap = cv2.VideoCapture(video_path)
            start_idx = s_value
            end_idx = e_value  # Note: this frame is inclusive
            frame_idx=0

            end_idx_true = min(end_idx+1, start_idx+traj2d.shape[0])
            for index_ply in range(start_idx, end_idx_true):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                # Read corresponding point cloud file
                pointcloud_path = os.path.join(pointcloud_dir, f"{index_ply}.ply")
                if not os.path.exists(pointcloud_path):
                    print(f"Point cloud file {pointcloud_path} not found, skipping.")
                    frame_idx += 1
                    continue

                # Load point cloud
                pointcloud = o3d.io.read_point_cloud(pointcloud_path)
                points = np.asarray(pointcloud.points)
                # Transform point cloud coordinates to match test points
                points[:,1] = -points[:,1]
                points[:,2] = -points[:,2]
                # Image preprocessing
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if image_rgb.ndim != 3 or image_rgb.dtype != np.uint8:
                    image_rgb = image_rgb.astype(np.uint8)
                # Get test point coordinates
                test_point = traj2d[frame_idx]
                test_x, test_y = int(test_point[0]), int(test_point[1])
                # Perform segmentation prediction
                predictor.set_image(image_rgb)
                test_point_array = np.array([[test_x, test_y]])
                test_point_indicator = np.ones(test_point_array.shape[0])
                masks, _, _ = predictor.predict(point_coords=test_point_array, point_labels=test_point_indicator)
                masks = masks[0, ...]
                masks_uint8 = masks.astype(np.uint8) * 255


                # # Define dilation kernel (typically 3x3 or 5x5)
                # kernel = np.ones((10, 10), np.uint8)  # Can adjust kernel size, e.g. (5, 5)
                # # Perform dilation using cv2.dilate
                # dilated_mask = cv2.dilate(masks_uint8, kernel, iterations=1)
                # masks_uint8 = masks_uint8.astype(np.uint8)
                # # Ensure binary output with only 0 and 255
                # masks_uint8[masks_uint8 > 0] = 255
                # # Mark test point and save segmentation result
                # cv2.circle(masks_uint8, (test_x, test_y), radius=10, color=(0, 0, 255), thickness=-1)
                # output_frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
                # cv2.imwrite(output_frame_path, masks_uint8)

                # Project point cloud to image plane
                points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
                uv = camera_matrix @ points_homogeneous[:, :3].T
                uv[:2] /= uv[2]  # Normalize to image plane
                u, v = uv[0].astype(int), uv[1].astype(int)

                # Filter out points outside image boundaries
                h, w = camera_intrinsics['h'], camera_intrinsics['w']
                mask_valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u, v = u[mask_valid], v[mask_valid]
                points = points[mask_valid]

                u = np.round(u * 0.25).astype(int)
                v = np.round(v * 0.25).astype(int)

                mask_valid = (u >= 0) & (u < w*0.25) & (v >= 0) & (v < h*0.25)
                u, v = u[mask_valid], v[mask_valid]
                points = points[mask_valid]


                # Filter point cloud based on mask
                filtered_points = []
                for i in range(len(u)):
                    if masks_uint8[v[i], u[i]] == 0:  # If not foreground in mask
                        filtered_points.append(points[i])
                filtered_points = np.array(filtered_points)

                # Save filtered point cloud
                filtered_pointcloud = o3d.geometry.PointCloud()
                filtered_pointcloud.points = o3d.utility.Vector3dVector(filtered_points)
                filtered_pointcloud_path = os.path.join(save_pointcloud_dir, f"{index_ply}.ply")
                o3d.io.write_point_cloud(filtered_pointcloud_path, filtered_pointcloud)

                print(f"Processed and saved frame {frame_idx} with filtered point cloud {filtered_pointcloud_path}")

                # Update index
                frame_idx += 1

            cap.release()
            print("Video and point cloud processing completed")