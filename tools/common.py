import os
import os.path as osp

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, CloughTocher2DInterpolator, RectBivariateSpline
from skimage import color
import cv2
import numpy as np
import numpy.linalg as npl
import math as M
import toml
import json
from easydict import EasyDict as edict
from shapely.geometry import Polygon

class InterpDepth:
    def __init__(self, depth):
        self.depth = depth
        h, w = depth.shape[:2]
        self.interp = RectBivariateSpline(np.arange(h), np.arange(w), depth)

    def __call__(self, v, u):
        return self.interp(v, u)

def transform_points(T, Xs):
    T = np.asarray(T)
    Xs = np.asarray(Xs)
    Xs = Xs.dot(T[:3,:3].T) + T[:3,3]
    return Xs

def project_points(K, xs):
    xs = np.asarray(xs)
    us = xs.dot(K.T)
    us = us[:,:2] / us[:,[2]]
    return us

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def ham_dis(c, binary):
    return sum(el1 != el2 for el1, el2 in zip(c, binary))

def calc_lidar_K(fov, min_angle):
    fov = [M.radians(i) for i in fov]
    min_angle = [M.radians(i) for i in min_angle]

    fx, fy = 1 / min_angle[0], 1 / min_angle[1]
    cx = fx * M.tan(fov[0] / 2)
    cy = fy * M.tan(fov[1] / 2)

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

def project_points(K, xs):
    xs = np.asarray(xs)
    us = xs.dot(K.T)
    us = us[:,:2] / us[:,[2]]
    return us

def unproject_points(uss, interp_depth, K):
    for usk, us in uss.items():
        if us is None:
            uss[usk] = None
            continue
        us = np.asarray([us])
        us = us[:, :2]
        ds = np.array([interp_depth(v, u)[0,0] for u, v in us])
        K = np.asarray(K)
        n = len(us)
        us = np.hstack((us, np.ones((n, 1))))
        us = us.dot(npl.inv(K).T)
        uss[usk] = [d*u for d, u in zip(ds, us)]
    return uss

def interpolate_projected_points(us, values, size, mode='linear', **kws):
    if mode == 'linear':
        interp = LinearNDInterpolator(us, values, **kws)
    elif mode == 'nearest':
        kws.pop('fill_value')
        interp = NearestNDInterpolator(us, values, **kws)
    elif mode == 'cubic':
        kws.pop('fill_value')
        interp = CloughTocher2DInterpolator(us, values, **kws)
    w, h = size
    X, Y = np.meshgrid(range(w), range(h))
    Z = interp(X, Y)
    return Z


def load_camera_config(Cp):
    with open(Cp, 'r') as f:
        C = json.load(f)
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

def normalize_2d(depth, matrix):
    x = np.clip(matrix, 0, depth)
    x = (np.max(x)- x)/(np.max(x)-np.min(x))
    return x

def to_rgb(img):
    if img.ndim == 2:
        img = color.gray2rgb(img)
    return img

def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img.astype('u1')

def blob_detector(config, img):
    imgray = to_gray(img)
    gray = cv2.medianBlur(imgray, 3)
    gray = cv2.erode(gray, np.ones((5, 5), np.uint8),iterations = 1)
    gray = cv2.dilate(gray, kernel=np.ones((5, 5), np.uint8))

    _, w = gray.shape[:2]

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.filterByArea = config.enable_filterby_gray
    params.minThreshold = config.min_gray_thresh
    params.maxThreshold = config.max_gray_thresh

    # Filter by Area.
    params.filterByArea = config.enable_filterby_area
    params.minArea = 3.14* config.min_r **2
    params.maxArea = 3.14* config.max_r **2

    # Filter by Circularity
    params.filterByCircularity = config.enable_filterby_circularity
    params.minCircularity = config.min_circularity
    params.maxCircularity = config.max_circularity

    # Filter by Convexity
    params.filterByConvexity = config.enable_filterby_convexity
    params.minConvexity = config.min_convexity
    params.maxConvexity = config.max_convexity

    # Filter by Inertia
    params.filterByInertia = config.enable_filterby_inertiaradio
    params.minInertiaRatio = config.min_inertiaradio
    params.maxInertiaRatio = config.max_inertiaradio

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    info = []
    for kp in keypoints:
        (x, y) = kp.pt
        r = kp.size
        info.append([x, y, r])
    result = np.array(info)
    return result

def read_video(path):
    capture = cv2.VideoCapture(path)
    imgs = []
    if capture.isOpened():
        while True:
            ret,img=capture.read()
            if not ret:break
            imgs.append(img)
    return imgs

def video_write(path, frames):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, 20.0, (w,h),True)
    for frame in frames:
        out.write(frame)
    print(f'saved to {path}')


def get_corners(p):
    p_result = convex_hull(p)
    corners = []
    for i, p in enumerate(p_result):
        p1 = p_result[i-1]
        p = p_result[i]
        if i+1 == len(p_result):
            p2 = p_result[0]
        else:
            p2 = p_result[i+1]

        theta = angle_of_vector([p[0]-p1[0], p[1]-p1[1]], [p[0]-p2[0], p[1]-p2[1]])
        if theta<170:
            corners.append(p)
    corners = np.array(corners)
    return corners

def convex_hull(ps):
    p = Polygon(ps)
    x = p.convex_hull
    a, b = x.exterior.coords.xy
    re = list(zip(a,b))
    assert re[0]==re[-1]
    return re[1:]

def angle_of_vector(v1, v2):
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = M.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * M.sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (M.acos(cos) / M.pi) * 180

def calc_distance(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    return M.sqrt(x**2 + y**2)

