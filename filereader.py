import numpy as np
import copy as copy_mod
import cv2
from tensorpack import *
# from .base import RNGDataFlow
# from .common import MapDataComponent, MapData
# from ..utils import logger
# from ..utils.argtools import shape2d


class ImageFromFile(RNGDataFlow):
    """ Produce images read from a list of files. """
    def __init__(self, files, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of file paths. containing of input image and gt seperated by space
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle

    def size(self):
        return len(self.files)

    def get_img(self,path):
        im = cv2.imread(path, self.imread_mode)
        if self.channel == 3:
            im = im[:, :, ::-1]
        if self.resize is not None:
            im = cv2.resize(im, tuple(self.resize[::-1]))
        if self.channel == 1:
            im = im[:, :, np.newaxis]
        return im

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.files)
        for input_im,gt_im in self.files:
            im = self.get_img(input_im)
            gt_im = self.get_img(gt_im)
            yield [im,gt_im]