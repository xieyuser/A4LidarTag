#!/usr/bin/env python
import os
import os.path as osp

import time
import sys
import random
import numpy as np
import numpy.linalg as npl
import cv2
from glob import glob
import click
from tqdm import tqdm
from easydict import EasyDict as edict
import toml
import json
import math as M
from sklearn.cluster import DBSCAN
cur_d = osp.dirname(__file__)
sys.path.append("../tools")
from common import *

def find_in_dic(config, info, res, dic, corners, x_list, y_list, thres, H_inverse, correct_axis=None):
    res_int = [int(res[:8], 2), int(res[8:], 2)]
    for i in range(4):
        try:
            info['lr'] = corners[i]
            info['id'] = dic['#'.join(list(map(str, res_int)) + [str(i)] + [res])]
            points = []
            if i in [0, 2]:
                for j_, j in enumerate(y_list):
                    for k_, k in enumerate(x_list):
                        if correct_axis is not None and [j_, k_] in correct_axis:
                            continue
                        value = thres[round(j)][round(k)]
                        if value > config.gray_thresh:
                            points.append([k, j, 1])

                if i != 0:
                    points.reverse()

            else:
                for k_, k in enumerate(range(3, -1, -1)):
                    for j_, j in enumerate(range(4)):
                        if correct_axis is not None and [j_, k_] in correct_axis:
                            continue
                        value = thres[round(y_list[j])][round(x_list[k])]
                        if value > config.gray_thresh:
                            points.append([x_list[k], y_list[j], 1])
                if i == 1:
                    points.reverse()
            p = H_inverse.dot(np.array(points).T).T
            p /= p[:, [2]]
            info['marker_points'] = p[:, :2]
            break
        except:
            continue
    return info

def wrap(config, dic, img, box):
    info = {
        "id": None,
        "lr": None,
        "marker_points": None,
    }

    d1 = calc_distance(box[0], box[1])
    d2 = calc_distance(box[0], box[3])

    dst = np.array([[0, 0], [0, d2], [d1, d2], [d1, 0]])

    H, _ = cv2.findHomography(box, dst)
    H_inverse = npl.inv(H)
    trans_img = cv2.warpPerspective(img, H, (round(d1), round(d2)))
    re = blob_detector(config, trans_img)
    if re.shape[0] < 4:
        return info

    re = re[:, :2]

    canvas = trans_img.copy()

    box = cv2.boxPoints(cv2.minAreaRect(re.astype(np.float32)))
    (xmin, ymin), (xmax, ymax) = np.min(box, axis=0), np.max(box, axis=0)
    x_, y_ = (xmax - xmin)/3, (ymax - ymin)/3
    x_list = [(xmin + i*x_) for i in range(4)]
    y_list = [(ymin + i*y_) for i in range(4)]

    _, thres = cv2.threshold(to_gray(trans_img),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h, w = thres.shape[:2]
    res = ''
    for j in y_list:
        for i in x_list:
            j, i = round(j), round(i)
            if j >= h or i >= w:
                res += '0'
            else:
                value = thres[j][i]
                res += '0' if value > config.gray_thresh else '1'
    corners = []
    corners.append([x_list[0], y_list[0], 1])
    corners.append([x_list[0], y_list[3], 1])
    corners.append([x_list[3], y_list[3], 1])
    corners.append([x_list[3], y_list[0], 1])

    corners = H_inverse.dot(np.array(corners).T).T
    corners /= corners[:, [2]]
    info = find_in_dic(config, info, res, dic, corners, x_list, y_list, thres, H_inverse)
    if config.enable_correction and info['id'] is  None:
        corr_res, corr_axis = error_correction(res, dic)
        if corr_res is None:
            return info
        info = find_in_dic(config, info, corr_res, dic, corners, x_list, y_list, thres, H_inverse, correct_axis=corr_axis)

    return info

def compare(src, dst):
    dif_i = []
    for i, (a, b) in enumerate(zip(src, dst)):
        if a!= b:
            dif_i.append(i)
    dif_i = [[i//4, i%4] for i in dif_i]
    return dif_i

def error_correction(res, dic):
    dis = []
    demos = []
    ids = []
    for k, _ in dic.items():
        demo = k.split('#')[-1]
        ids.append(k.split('#')[:2])
        demos.append(demo)
        dis.append(ham_dis(demo, res))
    dis = np.array(dis)
    indexs = np.where(dis == np.min(dis))[0]
    if indexs.shape[0] > 1:
        return None, None
    index = indexs[0]
    return demos[index], compare(demos[index], res)


def pattern_rec(config, dic, img, points):
    info = {
        'id': None,
        'marker_points': None,
        'lr': None,
        'box': None,
        'expand_box': None,
    }
    p = [tuple(i) for i in points.tolist()]
    box = get_corners(p)
    center = np.tile(np.mean(box, axis=0), (box.shape[0], 1))
    expand_box = center + (box - center) * config.expand_scale

    info['box'] = box
    info['expand_box'] = expand_box

    if box.shape[0] == 4:
        info.update(wrap(config, dic, img, expand_box))
    return info


def detect(config, dic, img, verify_dir=None):
    h, w = img.shape[:2]

    re = blob_detector(config, img)
    if re.shape[0] < 4:
        return

    det_dic = {}
    rmean = np.mean(re[:, 2])
    circle_clusters = DBSCAN(eps=config.eps_scale*rmean, min_samples=config.min_samples).fit(re[:, :2])
    labels = circle_clusters.labels_
    markers = [re[np.where(labels == i)] for i in range(-1, np.max(labels) + 1)]

    canvas_blob = img.copy()

    # verify
    if verify_dir:
        color = [0, 255, 0]
        for clsname, marker in enumerate(markers):
            for i, circle in enumerate(marker):
                cv2.circle(canvas_blob, (round(circle[0]), round(circle[1])), round(circle[2]/2), color, round(0.1 * circle[2]))
                cv2.putText(canvas_blob, str(clsname), (round(circle[0]), round(circle[1])), cv2.FONT_HERSHEY_SIMPLEX, round(round(0.002*h)),  (0, 0, 255), round(0.001*h), cv2.LINE_AA)

        det_dic.update({
            'canvas_blob': canvas_blob,
        })
    canvas = img.copy()
    for cls, marker in enumerate(markers):
        if marker.shape[0] <= 2:
            continue

        re_info = pattern_rec(config, dic, img, marker[:, :2])
        id, lr, box, expand_box = re_info['id'], re_info['lr'], re_info['box'], re_info['expand_box']

        if id:
            det_dic[id] = re_info
        else:
            continue
        if verify_dir:
            cv2.circle(canvas, (round(lr[0]), round(lr[1])), int(0.002*w), (0, 0, 255), -1)
            count = 0
            for i, j in re_info['marker_points'].tolist():
                cv2.circle(canvas, (round(i), round(j)), int(0.002*w), (255, 0, 0), -1)
                cv2.putText(canvas, str(count), (round(i), round(j)), cv2.FONT_HERSHEY_SIMPLEX, round(round(0.001*w)),  (0, 255, 0), round(0.001*w), cv2.LINE_AA)
                count += 1
            cv2.drawContours(canvas, [np.int0(box)], 0, (255, 0, 0), 1)
            cv2.drawContours(canvas, [np.int0(expand_box)], 0, (255, 0, 0), 1)
            cv2.circle(canvas, (round(np.mean(box[:, [0]])), round(np.mean(box[:, [1]]))), round(round(0.004*w)), (0, 0, 255), -1)
            cv2.putText(canvas, str(id), (round(np.mean(box[:, [0]])), round(np.mean(box[:, [1]]))), cv2.FONT_HERSHEY_SIMPLEX, round(round(0.001*w)),  (0, 0, 255), round(0.001*w), cv2.LINE_AA)
    det_dic.update({
        'canvas': canvas,
    })
    return det_dic


def main(config, dic, src, verify_dir):
    with open(config, 'r') as f:
        config = edict(toml.load(f))

    with open(dic, 'r') as f:
        dic = json.load(f)

    if osp.isdir(src):
        im_p = glob(f'{src}/*.jpg')
        imgs = [cv2.imread(img) for img in im_p]
    else:
        if src[-4:] == '.jpg':
            im_p = [src]
            imgs = [cv2.imread(src)]
        elif src[-4:] == '.mp4':
            im_p = [src]
            imgs = read_video(src)
        else:
            raise FileNotFoundError(f"No such file or no access: '{src}'")

    nums = len(im_p)
    if nums == 0:
        return

    if verify_dir:
        os.makedirs(verify_dir, exist_ok=True)

    frames = []
    for frame in tqdm(imgs):
        info = detect(config.marker, dic, frame, verify_dir)
        if info:
            frames.append(info["canvas"])
        else:
            frames.append(frame)

    if im_p[0][-4:] == '.jpg':
        for i in range(nums):
            cv2.imwrite(osp.join(verify_dir, osp.basename(im_p[i])), frames[i])
    else:
        print(frames.__len__())
        video_write(osp.join(verify_dir, "verify-video.avi"), frames)

@click.command()
@click.option('--config', default='config/demo.toml')
@click.option('--dic', default='config/redict.json')
@click.option('--verify-dir', default='verify')
@click.argument('src')
def cli(config, dic, src, verify_dir):
    main(config, dic, src, verify_dir)

if __name__ == '__main__':
    cli()
