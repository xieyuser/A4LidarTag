#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import cv2
import json
from easydict import EasyDict as edict
from glob import glob

def load_config(C):
    C0 = C['original']
    C1 = C['rectified']

    K0 = np.array(C0['K'])
    D0 = np.array(C0['D'])
    K1 = np.array(C1['K'])
    R1 = np.array(C1['R'])
    return edict({
        'model': C['model'],
        'img_size0': C0['img_size'],
        'K0': K0,
        'D0': D0,

        'img_size': C1['img_size'],
        'K': K1,
        'R': R1,
    })

def get_remap_param(C):
    if C.model == 'ocv_fisheye':
        init_undistort_map_fn = cv2.fisheye.initUndistortRectifyMap
    elif C.model == 'ocv':
        init_undistort_map_fn = cv2.initUndistortRectifyMap
    else:
        raise

    return init_undistort_map_fn(C.K0, C.D0, C.R, C.K, C.img_size, cv2.CV_16SC2)

def main(img_d='images', out_d='rectified_images/', cfg='calib.json'):
    C = load_config(json.load(open(cfg)))
    map_x, map_y = get_remap_param(C)
    os.makedirs(out_d, exist_ok=True)
    for img_f in sorted(glob(img_d+'/*.jpg')):
        img = cv2.imread(img_f)
        img1 = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(out_d+'/'+osp.basename(img_f), img1)


if __name__ == '__main__':
    main()

