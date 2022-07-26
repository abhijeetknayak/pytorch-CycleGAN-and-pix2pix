import os

import cv2
import ipdb
import numpy as np
from radar import *
from lidar import *
from tqdm import tqdm


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

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

vr_closest_times = np.array(vr_closest_times)
vl_closest_times = np.array(vl_closest_times)

radar_loc = "../data/oxford/sample/radar"
vl_loc = "../data/oxford/sample/velodyne_left"
vr_loc = "../data/oxford/sample/velodyne_right"

radar_results = "./oxford/radar"
os.makedirs(radar_results, exist_ok=True)

lidar_results = "./oxford/lidar"
os.makedirs(lidar_results, exist_ok=True)

combined_results = "./oxford/combined"
os.makedirs(combined_results, exist_ok=True)

counter = 1

for idx, time in tqdm(enumerate(radar_time)):
    # Process Radar
    radar_im_path = os.path.join(radar_loc, f"{time}.png")
    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
    radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, 0.2, 600)

    # Process Velodyne
    vl_im_path = os.path.join(vl_loc, f"{vl_closest_times[idx]}.png")
    vr_im_path = os.path.join(vr_loc, f"{vr_closest_times[idx]}.png")
    v_im = post_process_lidar(vl_im_path, vr_im_path)

    v_im = cv2.resize(v_im, (600, 600)).astype(np.uint8)
    radar_img = radar_img[:, :, 0]
    radar_img = ((radar_img - np.min(radar_img)) * (1/(np.max(radar_img) -
                                                       np.min(radar_img))) * 255).astype('uint8')

    ipdb.set_trace()
    file_name = f"{counter}.jpg"
    cv2.imwrite(os.path.join(radar_results, file_name),
                radar_img.astype(np.uint8))
    cv2.imwrite(os.path.join(lidar_results, file_name), v_im)

    # zero_channel = np.zeros((600, 600))
    #
    # r_b_im = np.dstack((radar_img, zero_channel, zero_channel))
    # v_r_im = np.dstack((zero_channel, zero_channel, v_im))
    #
    # added_image = cv2.addWeighted(r_b_im, 0.1, v_r_im, 0.4, 0)

    # cv2.imwrite(os.path.join(combined_results, file_name), added_image)
    # cv2.waitKey(0)

    counter += 1





