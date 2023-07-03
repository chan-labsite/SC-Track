#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: utils.py
# @Author: Li Chengxin 
# @Time: 2023/6/30 22:15

from __future__ import annotations

import logging
import time
import os
from copy import deepcopy
from functools import wraps
import skimage.exposure as exposure
from skimage.util import img_as_ubyte
import json
from libtiff import TIFF
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

import template

PHASE_MAP = {
    "G1/G2": 0,
    "M": 1,
    "S": 2
}


def time_it(logger: logging.Logger = None, using=False):
    """
    Decorator used to record the execution time of a function, using a log to output the execution time
    :param logger: Python logger
    :param using: bool flag, is the decorator function enabled?
    :return: decorator
    """
    def switch(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None or using is False:
                func(*args, **kwargs)
            else:
                start = time.time()
                func(*args, **kwargs)
                end = time.time()
                logger.info(f'{func} cost time:  {end - start:.4f}s')

        return wrapper

    return switch


def tif2png(img: str|os.PathLike, png_dir, gamma=0.1):
    """

    :param img: TIF image filepath
    :param png_dir: storing pngs dir
    :param gamma: image γ coefficient
    :return: None
    """
    tif = TIFF.open(img, )
    index = 0
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
    for i in tif.iter_images():
        filename = os.path.basename(img).replace('.tif', '-' + str(index).zfill(4) + '.png')
        img_mcy = exposure.adjust_gamma(i, gamma)
        png = img_as_ubyte(img_mcy)
        plt.imsave(os.path.join(png_dir, filename), png, cmap='gray')
        index += 1


def readTif(filepath):
    tif = tifffile.TiffFile(filepath)
    num = len(tif.pages)
    if num <= 1:
        tif.close()
        tif = tifffile.imread(filepath)
        for i, frame in enumerate(tif):
            filename = os.path.basename(filepath).replace('.tif', '-' + str(i).zfill(4) + '.tif')
            yield frame, filename
    else:
        for i in range(num):
            frame = tif.pages[i].asarray()
            filename = os.path.basename(filepath).replace('.tif', '-' + str(i).zfill(4) + '.tif')
            yield frame, filename


def convert_dtype(__image: np.ndarray) -> np.ndarray:
    """将图像从uint16转化为uint8"""
    min_16bit = np.min(__image)
    max_16bit = np.max(__image)
    image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def mask_to_coords(mask: np.ndarray | os.PathLike, filename):
    if type(mask) is str:
        mask = tifffile.imread(mask)
    mask_values = np.delete(np.unique(mask), np.where(np.unique(mask) == 0))
    frame_tmp = deepcopy(template.FRAME_TMP)
    regions = []
    for i in mask_values:
        new = np.where(mask == i, 1, 0)
        new = new.astype(np.uint8)
        contours, hierarchy = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x_coords = []
        y_coords = []
        for contour in contours:
            for pt in contour:
                x_coords.append(int(pt[0][0]))
                y_coords.append(int(pt[0][1]))
        region = deepcopy(template.REGIONS_TMP)
        region["shape_attributes"]["all_points_x"] = x_coords
        region["shape_attributes"]["all_points_y"] = y_coords
        regions.append(region)
    frame_tmp["regions"] = regions
    if os.path.splitext(filename)[1] == '.tif':
        filename = filename.replace('.tif', '.png')
    frame_tmp["filename"] = filename
    assert len(mask.shape) == 2
    frame_tmp["size"] = mask.shape[0] * mask.shape[1]
    return frame_tmp


def mask_seq_to_json(mask_dir):
    tif_files = glob.glob(os.path.join(mask_dir, '*.tif'))
    json_result = {}
    for file in tif_files:
        filename = os.path.basename(file).replace('.tif', '.png')
        ret = mask_to_coords(file, filename)
        json_result[filename] = ret
    return json_result


if __name__ == '__main__':
    filedir = r'G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\SEG'
    jsons = mask_seq_to_json(filedir)
    with open(r'G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\test.json', 'w') as f:
        json.dump(jsons, f)
