import math

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpolate_poses import *
import os
import open3d as o3d
from pyntcloud import PyntCloud

def read_gt(gt_file):
    data = pd.read_csv(gt_file)
    return data

import os

import cv2
import numpy as np
from radar import *
from lidar import *
from tqdm import tqdm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def velodyne_timestamps():
    radar_fp = '../data/oxford/sample/radar.timestamps'
    velodyne_right_fp = "../data/oxford/sample/velodyne_right.timestamps"
    velodyne_left_fp = "../data/oxford/sample/velodyne_left.timestamps"

    radar_file = open(radar_fp, 'r')
    velodyne_right_file = open(velodyne_right_fp, 'r')
    velodyne_left_file = open(velodyne_left_fp, 'r')

    radar_lines = radar_file.readlines()
    velodyne_right_lines = velodyne_right_file.readlines()
    velodyne_left_lines = velodyne_left_file.readlines()

    radar_time, vr_time, vl_time = [], [], []

    for line in radar_lines:
        radar_time.append(int(line.strip().split(" ")[0]))

    for line in velodyne_right_lines:
        vr_time.append(int(line.strip().split(" ")[0]))

    for line in velodyne_left_lines:
        vl_time.append(int(line.strip().split(" ")[0]))

    radar_time = np.array(radar_time)
    vr_time = np.array(vr_time)
    vl_time = np.array(vl_time)

    vr_closest_times = []
    vl_closest_times = []

    for time in radar_time:
        vr_closest_times.append(find_nearest(vr_time, time))
        vl_closest_times.append(find_nearest(vl_time, time))

    vr_closest_times = list(vr_closest_times)
    vl_closest_times = list(vl_closest_times)

    return vr_closest_times, vl_closest_times

def extract_ptcld(vl_path, vr_path):
    side_range = (-20, 20)  # left-most to right-most
    fwd_range = (-20, 20)  # back-most to forward-most
    vl_ptcld, vr_ptcld = return_ptcld(vl_path, vr_path, side_range, fwd_range)

    return vl_ptcld, vr_ptcld

def create_map(vr_time, vl_time):
    vl_loc = "../data/oxford/sample/velodyne_left"
    vr_loc = "../data/oxford/sample/velodyne_right"
    gt_pose_file = '../data/oxford/sample/gt/radar_odometry.csv'
    gt_poses = interpolate_vo_poses(gt_pose_file, vr_time, vr_time[0])

    extrinsics_dir = '../robotcar-dataset-sdk/extrinsics'

    with open(os.path.join(extrinsics_dir, 'velodyne_left.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_vl = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    with open(os.path.join(extrinsics_dir, 'velodyne_right.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_vr = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    # ipdb.set_trace()

    map = None
    for idx in tqdm(range(0, 500, 1)):
        # Process Velodyne
        vl_im_path = os.path.join(vl_loc, f"{vl_time[idx]}.png")
        vr_im_path = os.path.join(vr_loc, f"{vr_time[idx]}.png")
        vl_ptcld, vr_ptcld = extract_ptcld(vl_im_path, vr_im_path)
        i_l = vl_ptcld[3, :]
        i_r = vr_ptcld[3, :]
        scan_l = np.dot(np.dot(gt_poses[idx], G_posesource_vl),
                        np.vstack((vl_ptcld[0:3, :], np.ones(vl_ptcld.shape[1]))))
        scan_r = np.dot(np.dot(gt_poses[idx], G_posesource_vr),
                        np.vstack((vr_ptcld[0:3, :], np.ones(vr_ptcld.shape[1]))))
        scan = np.hstack((scan_l, scan_r))
        scan[3, :] = np.hstack((i_l, i_r))

        if map is None:
            map = scan
        else:
            map = np.hstack((map, scan))

    return map

if __name__ == '__main__':
    timestamps = [int(file[0:-4]) for file in sorted(os.listdir('../data/oxford/sample/radar')) if file.endswith('.png')]
    # timestamps = list(data['source_timestamp'])

    poses = interpolate_vo_poses('../data/oxford/sample/gt/radar_odometry.csv', timestamps, timestamps[0])
    # plot_trajectory(data)

    # x = [pose[0, 3] for pose in poses]
    # y = [pose[1, 3] for pose in poses]
    #
    # plt.plot(x, y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title("Absolute pose")
    # plt.show()

    # vr_time, vl_time = velodyne_timestamps()
    # # ipdb.set_trace()
    # map = create_map(vr_time, vl_time)
    map_file = 'vo.npz'
    # np.savez(map_file, map)

    map = np.load(map_file)['arr_0']
    # ipdb.set_trace()

    map = np.array(map.transpose())[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(map)
    pcd = pcd.uniform_down_sample(every_k_points=20)
    # pcd = o3d.io.read_point_cloud("map.ply")

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
