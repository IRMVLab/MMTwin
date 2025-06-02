"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file shows how to generate camera egomotion homography offline.
"""

import os
import numpy as np
import cv2
import yaml
from pathlib import Path


def load_config(config_path="ceg.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def match_keypoints(kpsA, kpsB, featuresA, featuresB, match_type, ratio=0.7, reprojThresh=4.0):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0]))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        if match_type == "ransac":
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        elif match_type == "magsac":
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.MAGSAC, reprojThresh)
        else:
            raise RuntimeError("Wrong matching type")
        matchesMask = status.ravel().tolist()
        return matches, H, matchesMask
    return None

def get_pair_homography(frame_1, frame_2, descriptor_type, match_type):
    flag = True

    if descriptor_type == "sift":
        descriptor = cv2.SIFT_create()
    elif descriptor_type == "orb":
        descriptor = cv2.ORB_create()
    elif  descriptor_type == "brisk":
        descriptor = cv2.BRISK_create()
    else:
        raise RuntimeError("Wrong descriptor type")

    (kpsA, featuresA) = descriptor.detectAndCompute(frame_1, mask=None)
    (kpsB, featuresB) = descriptor.detectAndCompute(frame_2, mask=None)
    matches, matchesMask = None, None
    try:
        (matches, H_BA, matchesMask) = match_keypoints(kpsB, kpsA, featuresB, featuresA, match_type)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False

    NoneType = type(None)
    if type(H_BA) == NoneType:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    try:
        np.linalg.inv(H_BA)
    except Exception:
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    return matches, H_BA, matchesMask, flag

def _read_rgb(video_file):
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    videos = []
    while success:
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        videos.append(frame)
        success, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return videos


def main(config):
    # Load configuration
    config = load_config(config)
    
    descriptor_type = config['descriptor']['type']
    match_type = config['descriptor']['match_type']
    videos_root = config['paths']['videos_root']
    dst_dir = config['paths']['dst_dir']

    scene_id_list = os.listdir(videos_root)
    rgb_paths = []
    idx = 0
    for scene_id in scene_id_list:
        idx += 1
        print(idx)
        path1 = os.path.join(videos_root, scene_id)
        record_list = os.listdir(path1)
        for record in record_list:
            path2 = os.path.join(path1, record)
            clips = os.listdir(path2)
            for clip_name in clips:
                filename = os.path.join(path2, clip_name)
                print("filename: ", filename)
                new_name = filename.replace('/', '__')[:-4]
                new_name = os.path.join(dst_dir, new_name)

                if os.path.exists(new_name + '.npy'):
                    continue
                rgb = _read_rgb(filename)
                feats_per_video_list = []
                homography_stack = [np.eye(3)]
                for f_idx in range(len(rgb)): 
                    if f_idx > 0:
                        matches, H_BA, matchesMask, flag = get_pair_homography(rgb[f_idx - 1], rgb[f_idx], descriptor_type, match_type)
                        homography_stack.append(np.dot(homography_stack[-1], H_BA))
                
                homography_stack_arr = np.stack(homography_stack) 
                # print(homography_stack_arr.shape)
                np.save(new_name, homography_stack_arr)

if __name__ == "__main__":
    config_path="ceg.yml"
    main(config_path)