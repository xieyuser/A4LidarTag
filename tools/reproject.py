import os
import os.path as osp
import sys
cur_d = osp.dirname(__file__)
sys.path.append(osp.join(cur_d, '..'))
import json
import click
from glob import glob
from easydict import EasyDict as edict
from tqdm import tqdm
from tools.common import *
from tools.remap import main as remap


def normal(x):
    Min = np.min(x[:1000])
    Max = np.max(x[:1000])
    x = (x - Min) / (Max - Min);
    return x

@click.command()
@click.argument('device')
@click.argument('img_dir')
@click.argument('pc_dir')
@click.argument('rectified_img_dir')
@click.option('--reproject', default='./calib_result/reproject')
def main(device, img_dir, pc_dir, rectified_img_dir, reproject):
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

    os.makedirs(reproject, exist_ok=True)
    os.makedirs(reproject, exist_ok=True)
    names = [file_name(i) for i in sorted(glob(osp.join(rectified_img_dir, "*.jpg")))]
    pds = []

    for name in tqdm(names):
        rgb = cv2.imread(osp.join(rectified_img_dir, name+'.jpg'))
        lidar_Xs = np.load(osp.join(pc_dir, name+'.npy'))[:, :3]

        # lidar_Xs = lidar_Xs[np.where((lidar_Xs[:,2] < 3000) & (lidar_Xs[:,2] > 0))]

        cam_xs = transform_points(T_cl, lidar_Xs)

        xs = np.asarray(cam_xs)
        cam_us = xs.dot(K_c.T)

        np.random.shuffle(cam_us)
        cam_us = cam_us[:40000]

        d = normal(cam_us[:, 2].copy())
        ds = (d*255).astype('i4')

        cam_us = cam_us[:,:2] / cam_us[:,[2]]
        cam_us = cam_us.astype('i4')

        canvas = rgb.copy()
        for c,d in zip(cam_us, ds):
            cv2.circle(canvas, tuple(c), 2, (0, 255-int(d), int(d)), -1)
        cv2.imwrite(osp.join(reproject, f'{name}.jpg'), canvas)


if __name__ == '__main__':
    main()




