import os
import os.path as osp
import time

import cv2
import numpy as np

import pyrealsense2 as rs
import click


@click.command()
@click.option('--data-dir', default='data_l515')
def capture(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(osp.join(data_dir, 'mono_calib'), exist_ok=True)
    os.makedirs(osp.join(data_dir, 'pc_calib'), exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    pipeline.start(config)
    for _ in range(20):
        print('start')
        frames = pipeline.wait_for_frames()

        depth = frames.get_depth_frame()
        rgb = np.asarray(frames.get_color_frame().get_data())

        pc = rs.pointcloud()
        points = pc.calculate(depth)

        vtx = np.asarray(points.get_vertices())

        pc = [list(i) for i in vtx]
        pc = np.vstack(pc)
        pc = pc[pc[:,2] > 0]

        name = str(time.time())
        cv2.imwrite(osp.join(data_dir, f'mono_calib/{name}.jpg'), rgb)
        np.save(osp.join(data_dir, f'pc_calib/{name}.npy'), pc)
        print('end\n')


if __name__ == "__main__":
    capture()
