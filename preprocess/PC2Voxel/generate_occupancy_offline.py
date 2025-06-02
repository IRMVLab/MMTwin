"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file shows a demo that converts raw point clouds to voxel grids.
"""

import open3d as o3d
import numpy as np
import yaml
import copy

def load_config(config_path="p2v.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Extract parameters
    odometry_path = config['paths']['odometry_path']
    pointcloud_base_dir = config['paths']['pointcloud_dir']
    
    start_frame = config['processing']['start_frame']
    end_frame = config['processing']['end_frame']
    frames_ratio = config['processing']['frames_ratio']
    
    max_correspondence = config['registration']['max_correspondence_distance']
    estimation_method = config['registration']['estimation_method']

    # Load odometry data
    odometry_data = np.load(odometry_path)

    # Initialize containers
    original_pointclouds = []
    transformed_pointclouds = []
    transformations_list = []

    # Calculate end frame index
    processed_end_idx = start_frame + int((end_frame - start_frame + 1) * frames_ratio)

    # Get appropriate estimation method
    estimation = {
        "PointToPoint": o3d.pipelines.registration.TransformationEstimationPointToPoint,
        "PointToPlane": o3d.pipelines.registration.TransformationEstimationPointToPlane
    }[estimation_method]()

    # Process each frame
    for current_frame in range(start_frame, processed_end_idx):
        # Load point cloud
        pc_path = f"{pointcloud_base_dir}/{current_frame}.ply"
        current_pc = o3d.io.read_point_cloud(pc_path)
        
        if current_frame == start_frame:
            # First frame - no transformation needed
            original_pointclouds.append(current_pc)
            transformed_pointclouds.append(current_pc)
            transformations_list.append(odometry_data[0])
        else:
            # Perform ICP registration
            icp_result = o3d.pipelines.registration.registration_icp(
                current_pc, 
                original_pointclouds[-1], 
                max_correspondence,
                np.eye(4),
                estimation
            )
            
            # Store and apply transformations
            transformations_list.append(icp_result.transformation)
            transformed_pc = copy.deepcopy(current_pc)
            for transform in reversed(transformations_list):
                transformed_pc = transformed_pc.transform(transform)
            
            # Store results
            original_pointclouds.append(current_pc)
            transformed_pointclouds.append(transformed_pc)

    # Merge and visualize
    merged_pc = transformed_pointclouds[0]
    for pc in transformed_pointclouds[1:]:
        merged_pc += pc
        
    o3d.visualization.draw_geometries([merged_pc])

if __name__ == "__main__":
    main()