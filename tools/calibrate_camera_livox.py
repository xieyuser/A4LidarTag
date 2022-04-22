import os
import os.path as osp
import sys

cur_d = osp.dirname(__file__)
sys.path.insert(0, "..")

import numpy as np
import cv2
from skimage.io import imread, imsave
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator, RectBivariateSpline

import click
import json
import toml
from easydict import EasyDict as edict
from glob import glob
import pyvista as pv
from pylab import plt

from detector import detect
from common import *
from remap import main as remap


def solve_pnp(K, Xs, us, reproj_th=8):
    print(f"solving pnp {Xs.shape[0]} points pair...")
    print("object points")
    print(Xs[:20])
    print("image points")
    print(us[:20])
    ret, rvec, t, inliers = cv2.solvePnPRansac(Xs, us, K, None, reprojectionError=reproj_th)
    assert ret
    R, _ = cv2.Rodrigues(rvec)
    inliers = inliers.T[0]
    return np.hstack((R, t)), inliers

@click.command()
@click.option('--calib-config', default='./config/camera_livox_config.toml')
@click.option('--mono-calib', default="./config/mono_calib_livox.json")
@click.option('--pattern-dic', default="./config/redict.json")
@click.option('--out-f', default='./calib_result/camera_livox_transform.json')
@click.option('--verify-dir', default='./calib_result/verify')
def main(calib_config, mono_calib, pattern_dic,  out_f, verify_dir):
    print("Calibrating...")

    with open(pattern_dic, 'r') as f:
        pattern_dic = edict(json.load(f))
    with open(calib_config, 'r') as f:
        C = edict(toml.load(f))

    if verify_dir:
        os.makedirs(verify_dir, exist_ok=True)

    cam_C = load_camera_config(mono_calib)
    (imw, imh) = cam_C.img_size

    cam_K = np.asarray(cam_C.K)
    remap(C.mono.mono_dir, C.mono.rectified_mono_dir, mono_calib)

    verify_dir = C.basic.verify_dir

    pc_dir = C.lidar.pc_dir
    lidar_K = calc_lidar_K(C.lidar.fov, C.lidar.min_angle)
    lidar_w, lidar_h = M.ceil(lidar_K[0, 2]*2), M.ceil(lidar_K[1,2]*2)
    corr_img_us, corr_lidar_Xs = [], []

    names = [file_name(i) for i in sorted(glob(osp.join(C.mono.rectified_mono_dir, "*.jpg")))]
    if True:
        for name in names:
            print(f"\nCalibrating {name} ...")
            img_f = osp.join(C.mono.rectified_mono_dir, name + '.jpg')

            img = imread(img_f)
            assert img.shape[:2] == (imh, imw)

            img_info = detect(C.marker.rgb, pattern_dic, img, verify_dir)

            if not img_info:
                continue
            if verify_dir:
                imsave(osp.join(verify_dir, f"{name}_img_markers.jpg"), img_info['canvas'])
                imsave(osp.join(verify_dir, f"blob_{name}_img_markers.jpg"), img_info['canvas_blob'])

            # lidar
            pc_f = osp.join(pc_dir, name + '.npy')
            pd = np.load(pc_f)

            # plot
            try:
                Xs = pd[:, :3]
                I = pd[:, -1].copy()
            except:
                continue

            ds = Xs[:, 2].copy()
            us = project_points(lidar_K, Xs)

            # plot
            interp_src = np.vstack((Xs[:,2], I)).T
            interp_data = interpolate_projected_points(us, interp_src, (lidar_w, lidar_h), mode=C.lidar.interp_mode, fill_value=C.lidar.interp_fill_value)
            Z_map, I_map = interp_data[..., 0], interp_data[..., 1].astype('u1')

            # plt.imshow(I_map)
            # plt.show()
            lidar_img = to_rgb((normalize_2d(C.basic.clip_depth, Z_map)*255).astype('u1'))

            imsave(osp.join(verify_dir, f"{name}_lidar_depth.jpg"), lidar_img)
            lidar_info = detect(C.marker.lidar, pattern_dic, lidar_img, verify_dir)

            if not lidar_info:
                continue

            if verify_dir:
                imsave(osp.join(verify_dir, f"{name}_lidar_markers.jpg"), lidar_info['canvas'])
                imsave(osp.join(verify_dir, f"blob_{name}_lidar_markers.jpg"), lidar_info['canvas_blob'])

            lidar_markers = {}
            img_markers = {}

            lidar_x = []
            img_x = []
            keys = img_info.keys() & lidar_info.keys()
            for k in keys:
                if 'canvas' in k:
                    continue
                lidar_x += lidar_info[k]['marker_points'].tolist()
                img_x += img_info[k]['marker_points'].tolist()
            for i in range(lidar_x.__len__()):
                lidar_markers[i] = lidar_x[i]

            for i in range(img_x.__len__()):
                img_markers[i] = img_x[i]

            interp_depth = InterpDepth(Z_map)
            lidar_Xs = unproject_points(lidar_markers, interp_depth, lidar_K)
            keys = img_markers.keys() & lidar_Xs.keys()
            for k in keys:
                if lidar_Xs[k] is not None and img_markers[k] is not None:
                    print(f"selecting points...")
                    corr_img_us += list(np.array([img_markers[k][:2]]))
                    corr_lidar_Xs += list(lidar_Xs[k])

        corr_img_us = np.array(corr_img_us)
        corr_lidar_Xs = np.array(corr_lidar_Xs)
        np.save('data/corr_img_us_livox.npy', corr_img_us)
        np.save('data/corr_lidar_Xs.npy', corr_lidar_Xs)
    else:
        corr_img_us = np.load('data/corr_img_us_livox.npy')
        corr_lidar_Xs = np.load('data/corr_lidar_Xs.npy')
    T_cl, inliers = solve_pnp(cam_K, corr_lidar_Xs, corr_img_us, reproj_th=C.basic.pnp_thr)

    proj_us = project_points(cam_K, transform_points(T_cl, corr_lidar_Xs[inliers, :]))
    reproject_xs = npl.norm(proj_us - corr_img_us[inliers], axis=1)
    rms = reproject_xs.mean()
    inliers_p = corr_img_us[inliers]
    print(inliers_p)
    for i in [0.5, 1, 5, 10]:
        print(len(np.where(reproject_xs < i)[0])/len(inliers))
    print ("rms", rms, "inlier_nr", len(inliers))

    out = {
        'pre_transform': C.lidar.pre_T,

        'K_c': cam_K.tolist(),
        'size_c': (imw, imh),

        'K_l': lidar_K.tolist(),
        'size_l': (lidar_w, lidar_h),

        'T_cl': T_cl.tolist(),
        'rms': rms,
        'inlier_nr': len(inliers),
        'corr_nr': len(corr_img_us),
    }

    json.dump(out, open(out_f, 'w'), indent=2)


if __name__ == "__main__":
    main()
