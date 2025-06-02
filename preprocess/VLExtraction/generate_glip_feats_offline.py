"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file shows how to generate vision-language features by GLIP.
"""

import os
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import os
import pickle
from tqdm import trange

config_file = "configs/pretrain/glip_Swin_L.yaml"
weight_file = "MODEL/glip_large_model.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


def _read_rgb(video_file):
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


config_path = os.path.join(os.path.dirname(__file__), 'vle.yml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
videos_root = config['videos_root']
dst_dir = config['dst_dir']

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
            for img_idx in range(rgb.shape[0]):
                pil_image = rgb[img_idx, ...]
                image = np.array(pil_image)[:, :, [2, 1, 0]]
                caption = 'hand'
                result, _, visual_features = glip_demo.run_on_web_image(image, caption, 0.55)
                feats_per_video_list.append(visual_features[-1].cpu().numpy())
            
            feats_per_video_arr = np.concatenate(feats_per_video_list)
            np.save(new_name, feats_per_video_arr)