"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import ipdb
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
# from ..scripts.radar import *
import matplotlib.pyplot as plt
import cv2

class RadarToLidarDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # Radar and Lidar Image Paths
        self.radar_root = f'{opt.dataroot}/radar'
        self.lidar_root = f'{opt.dataroot}/lidar'

        self.lidar_paths = sorted(make_dataset(self.lidar_root, opt.max_dataset_size))
        self.radar_paths = sorted(make_dataset(self.radar_root, opt.max_dataset_size))

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        radar_path = self.radar_paths[index]
        lidar_path = self.lidar_paths[index]

        radar_img = Image.open(radar_path)
        lidar_img = Image.open(lidar_path)

        data_A = self.transform(radar_img)  # needs to be a tensor
        data_B = self.transform(lidar_img)    # needs to be a tensor

        return {'A': data_A, 'B': data_B, 'A_paths': self.radar_paths[index],
                'B_paths': self.lidar_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.radar_paths)
