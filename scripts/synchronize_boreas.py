import os

import cv2
import ipdb
import numpy as np
from radar import *
from lidar import *
from tqdm import tqdm
import sys
from pyboreas.utils.radar import *


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

dataroot = '/export/nayaka/boreas'
folders = sorted([os.path.join(dataroot, f) for f in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, f))])

for folder_idx in range(0, len(folders)):
    print(folders[folder_idx])
    data_folder = folders[folder_idx]

    radar_results = f"{data_folder}/radar_cart_256_0.5"
    os.makedirs(radar_results, exist_ok=True)

    radar_folder_path = os.path.join(dataroot, data_folder, 'radar')

    for file in tqdm(os.listdir(radar_folder_path)):
        if not file.endswith('.png'):
            continue
        radar_im_path = os.path.join(radar_folder_path, file)

        timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
        radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, 0.5, 256)

        # ipdb.set_trace()

        # radar_img = radar_img[:, :, 0]
        radar_img = ((radar_img - np.min(radar_img)) * (1/(np.max(radar_img) -
                                                        np.min(radar_img))) * 255).astype('uint8')

        cv2.imwrite(os.path.join(radar_results, file), radar_img.astype(np.uint8))