from mobile_sam import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import os
import pickle
import open3d as o3d
import copy
import yaml
from pathlib import Path

# ====================================
with open('ms.yml', 'r') as f:
    config = yaml.safe_load(f)

# Demo configs
video_path = config['demo']['video_path']
test_point_path = config['demo']['test_point_path']
pointcloud_dir = config['demo']['pointcloud_dir']
output_mask_dir = config['demo']['output']['mask_dir']
output_pointcloud_dir = config['demo']['output']['pointcloud_dir']
output_raw_dir = config['demo']['output']['raw_image_dir']
start_idx = int(config['demo']['start_idx'])
show_filtered_pc = config['demo']['output']['show_filtered_pc']

# Create output directories
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_pointcloud_dir, exist_ok=True)
os.makedirs(output_raw_dir, exist_ok=True)
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

# Read test point file
with open(test_point_path, 'rb') as f:
    data = pickle.load(f)
traj2d = data['traj2d']

# Open video file
cap = cv2.VideoCapture(video_path)
frame_idx = 0
point_idx = 0

# Check if video opened successfully
if not cap.isOpened():
    print("Failed to open video file")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}, Frames to process: {len(traj2d)}")

while point_idx < len(traj2d):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}, terminating processing.")
        break

    # Read corresponding point cloud file
    pointcloud_path = os.path.join(pointcloud_dir, f"{start_idx}.ply")
    if not os.path.exists(pointcloud_path):
        print(f"Point cloud file {pointcloud_path} not found, skipping.")
        start_idx += 1
        frame_idx += 1
        continue

    # Load point cloud
    pointcloud = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pointcloud.points)

    points[:,1] = -points[:,1]
    points[:,2] = -points[:,2]

    # Image preprocessing
    image_rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb = copy.deepcopy(image_rgb_converted)
    if image_rgb.ndim != 3 or image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    # Get test point coordinates
    test_point = traj2d[point_idx]
    test_x, test_y = int(test_point[0]), int(test_point[1])

    # Perform segmentation prediction
    predictor.set_image(image_rgb)
    test_point_array = np.array([[test_x, test_y]])
    test_point_indicator = np.ones(test_point_array.shape[0])
    masks, _, _ = predictor.predict(point_coords=test_point_array, point_labels=test_point_indicator)
    masks = masks[0, ...]
    masks_uint8 = masks.astype(np.uint8) * 255

    # Save segmentation result
    # cv2.circle(masks_uint8, (test_x, test_y), radius=10, color=(0, 0, 255), thickness=-1)
    output_frame_path = os.path.join(output_mask_dir, f"frame_{start_idx:04d}.png")
    cv2.imwrite(output_frame_path, masks_uint8)
    output_frame_raw_path = os.path.join(output_raw_dir, f"frame_{start_idx:04d}.png")
    cv2.imwrite(output_frame_raw_path, image_rgb_converted)

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

    # Scale u,v by 0.25
    u = (u * 0.25).astype(int)
    v = (v * 0.25).astype(int)

    # Filter point cloud based on mask
    filtered_points = []
    for i in range(len(u)):
        if masks_uint8[v[i], u[i]] == 0:  # If not foreground in mask
            filtered_points.append(points[i])
    filtered_points = np.array(filtered_points)

    # Save filtered point cloud
    filtered_pointcloud = o3d.geometry.PointCloud()
    filtered_pointcloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pointcloud_path = os.path.join(output_pointcloud_dir, f"filtered_{start_idx}.ply")
    o3d.io.write_point_cloud(filtered_pointcloud_path, filtered_pointcloud)

    print(f"Processed and saved frame {frame_idx} and filtered point cloud {filtered_pointcloud_path}")

    # Update indices
    point_idx += 1
    frame_idx += 1
    start_idx += 1

cap.release()
print("Video and point cloud processing completed")

if show_filtered_pc:
    # Load point cloud file
    file_path = "./filtered_pointclouds/filtered_1225.ply" 

    point_cloud = o3d.io.read_point_cloud(file_path)

    points = np.asarray(point_cloud.points)

    # Flip y and z back for better visualization
    points[:,1] = -points[:,1]
    points[:,2] = -points[:,2]

    # Create coordinate axes
    def create_coordinate_axes(size=0.1):
        # Define start and end points for axes
        axis_points = np.array([[0, 0, 0], [size, 0, 0], [0, 0, 0], [0, size, 0], [0, 0, 0], [0, 0, size]])
        
        # Define connections between points
        axis_lines = np.array([[0, 1], [2, 3], [4, 5]])
        
        # Create LineSet object
        coordinate_axes = o3d.geometry.LineSet()
        coordinate_axes.points = o3d.utility.Vector3dVector(axis_points)
        coordinate_axes.lines = o3d.utility.Vector2iVector(axis_lines)
        
        # Set colors
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Red (right), Green (up), Blue (back)
        coordinate_axes.colors = o3d.utility.Vector3dVector(colors)
        
        return coordinate_axes

    # Create coordinate axes
    axes = create_coordinate_axes(size=0.1)

    # Visualize point cloud and axes, set line width
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud and axes
    vis.add_geometry(point_cloud)
    vis.add_geometry(axes)

    # Set line width
    opt = vis.get_render_option()
    opt.line_width = 100.0  # Set line width to 5.0, adjust as needed

    # Start visualization
    vis.run()
    vis.destroy_window()