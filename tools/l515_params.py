import os
import os.path as osp

import cv2
import numpy as np

import json
import pyrealsense2 as rs


def init_realsense():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    frames = pipeline.wait_for_frames()

    depth = frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
    rgb = frames.get_color_frame()

    depth_intri = depth.profile.as_video_stream_profile().intrinsics
    rgb_intri = rgb.profile.as_video_stream_profile().intrinsics

    lidar_K = np.array([
        [depth_intri.fx, 0, depth_intri.ppx],
        [0, depth_intri.fy, depth_intri.ppy],
        [0,0,1],
    ])

    lidar_w, lidar_h = depth_intri.width, depth_intri.height

    rgb_K = np.array([
        [rgb_intri.fx, 0, rgb_intri.ppx],
        [0, rgb_intri.fy, rgb_intri.ppy],
        [0,0,1],
    ])

    rgb_imgsize = [rgb_intri.width, rgb_intri.height]
    rgb_coeffs = rgb_intri.coeffs

    params = {
        'type': 'l515',
        'rgb':{
            'img_size': rgb_imgsize,
            'K': rgb_K.tolist(),
            'D': rgb_coeffs,
        },
        'lidar':{
            'K': lidar_K.tolist(),
            'img_size': [lidar_w, lidar_h],
        }
    }
    with open('config/l515_params.json', 'w') as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    init_realsense()
