#!/usr/bin/env python
import os
import os.path as osp
import sys
sys.path.append(".")
import numpy as np
from glob import glob
import json
import pyvista as pv
from skimage.io import imread
from easydict import EasyDict as edict
import click
import toml
from tqdm import tqdm
from tools.common import transform_points, project_points, file_name
from remap import main as remap

@click.command()
@click.argument('device')
@click.argument('img_dir')
@click.argument('pc_dir')
@click.argument('rectified_img_dir')
@click.option('--rectified_pc', default='./calib_result/rectified_pc')
def main(device, img_dir, pc_dir, rectified_img_dir, rectified_pc):
    if device == 'livox':
        config = './config/camera_livox_config.toml'
        rt = "./calib_result/camera_livox_transform.json"
        mono_calib = './config/mono_calib_livox.json'
    elif device == 'l515':
        config = './config/camera_l515_config.toml'
        rt = "./calib_result/camera_l515_transform.json"
        mono_calib = './config/mono_calib_l515.json'
    else:
        raise 'device not surported'

    with open(config) as f:
        config = edict(toml.load(config))

    with open(rt) as f:
        rt = edict(json.load(f))
    T_cl = np.array(rt.T_cl)
    K_c = np.array(rt.K_c)
    (imw, imh) = rt.size_c

    remap(img_dir, rectified_img_dir, mono_calib)

    os.makedirs(rectified_pc, exist_ok=True)
    os.makedirs(rectified_img_dir, exist_ok=True)
    names = [file_name(i) for i in sorted(glob(osp.join(rectified_img_dir, "*.jpg")))]
    pds = []
    for name in tqdm(names):
        img_f = osp.join(rectified_img_dir, name+'.jpg')
        img = imread(img_f)
        assert img.shape[:2] == (imh, imw)

        pc_f = osp.join(pc_dir, name+'.npy')
        lidar_Xs = np.load(pc_f)[:, :3]

        cam_xs = transform_points(T_cl, lidar_Xs)
        cam_us = project_points(K_c, cam_xs)

        cam_us = cam_us.astype('i4')

        mask = [(0 <= v < imh and 0 <= u < imw) for u, v in cam_us]

        cam_xs = cam_xs[mask]
        colors = [img[v, u] for m, (u, v) in zip(mask, cam_us) if m]
        pd = pv.PolyData(cam_xs)
        pd['colors'] = colors
        out_pc_f = osp.join(rectified_pc, name+'.vtp')

        pd.save(out_pc_f)
        pc_rgb = {
            'pc': pd,
            'rgb': img_f,
        }
        pds.append(pc_rgb)
    return pds

if __name__ == "__main__":
    main()

