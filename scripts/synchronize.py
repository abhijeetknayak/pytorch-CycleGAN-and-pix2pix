import os

import cv2
import ipdb
import numpy as np
from radar import *
from lidar import *
from tqdm import tqdm
import sys
# from pyboreas.utils.radar import *

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

dataroot = '/export/nayaka/radar_robotcar_sequences'
folders = sorted([os.path.join(dataroot, f) for f in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, f))])

for folder_idx in range(len(folders)):
    print(folders[folder_idx])
    data_folder = folders[folder_idx]
    radar_fp = f'{data_folder}/radar.timestamps'
    # velodyne_right_fp = f"{data_folder}/velodyne_right.timestamps"
    # velodyne_left_fp = f"{data_folder}/velodyne_left.timestamps"

    radar_file = open(radar_fp, 'r')
    # velodyne_right_file = open(velodyne_right_fp, 'r')
    # velodyne_left_file = open(velodyne_left_fp, 'r')

    radar_lines = radar_file.readlines()
    # velodyne_right_lines = velodyne_right_file.readlines()
    # velodyne_left_lines = velodyne_left_file.readlines()

    radar_time, vr_time, vl_time = [], [], []

    for line in radar_lines:
        radar_time.append(int(line.strip().split(" ")[0]))

    # for line in velodyne_right_lines:
    #     vr_time.append(int(line.strip().split(" ")[0]))

    # for line in velodyne_left_lines:
    #     vl_time.append(int(line.strip().split(" ")[0]))

    radar_time = np.array(radar_time)
    # vr_time = np.array(vr_time)
    # vl_time = np.array(vl_time)

    # vr_closest_times = []
    # vl_closest_times = []

    # for time in radar_time:
    #     vr_closest_times.append(find_nearest(vr_time, time))
    #     vl_closest_times.append(find_nearest(vl_time, time))

    # vr_closest_times = np.array(vr_closest_times)
    # vl_closest_times = np.array(vl_closest_times)

    radar_loc = f"{data_folder}/radar"
    # vl_loc = f"{data_folder}/velodyne_left"
    # vr_loc = f"{data_folder}/velodyne_right"

    img_size = 256
    img_res = 0.5

    radar_results = f"{data_folder}/radar_cart_{img_size}_{img_res}"
    os.makedirs(radar_results, exist_ok=True)

    # lidar_results = f"{data_folder}/lidar_cart"
    # os.makedirs(lidar_results, exist_ok=True)

    # combined_results = f"{data_folder}/combined"
    # os.makedirs(combined_results, exist_ok=True)

    # counter = 0

    for idx, time in enumerate(tqdm(radar_time)):
        # Process Radar
        radar_im_path = os.path.join(radar_loc, f"{time}.png")
        timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
        radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, img_res, img_size)

        # Process Velodyne
        # vl_im_path = os.path.join(vl_loc, f"{vl_closest_times[idx]}.png")
        # vr_im_path = os.path.join(vr_loc, f"{vr_closest_times[idx]}.png")
        # v_im = post_process_lidar(vl_im_path, vr_im_path)

        # v_im = cv2.resize(v_im, (600, 600)).astype(np.uint8)
        radar_img = radar_img[:, :, 0]
        radar_img = ((radar_img - np.min(radar_img)) * (1/(np.max(radar_img) -
                                                        np.min(radar_img))) * 255).astype('uint8')

        file_name = f"{time}.png"
        cv2.imwrite(os.path.join(radar_results, file_name),
                    radar_img.astype(np.uint8))
        # cv2.imwrite(os.path.join(lidar_results, file_name), v_im)

        # zero_channel = np.zeros((600, 600))
        #
        # r_b_im = np.dstack((radar_img, zero_channel, zero_channel))
        # v_r_im = np.dstack((zero_channel, zero_channel, v_im))
        #
        # added_image = cv2.addWeighted(r_b_im, 0.1, v_r_im, 0.4, 0)

        # cv2.imwrite(os.path.join(combined_results, file_name), added_image)
        # cv2.waitKey(0)

        # counter += 1





