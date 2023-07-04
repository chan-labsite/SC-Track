#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: utils.py
# @Author: Li Chengxin 
# @Time: 2023/6/30 22:15

from __future__ import annotations

import logging
import sys
import time
import os
from copy import deepcopy
from functools import wraps
import skimage.exposure as exposure
from skimage.util import img_as_ubyte
from libtiff import TIFF
import tifffile
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from tqdm import tqdm

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
    :param using: bool flag, whether the decorator function enabled.
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


def tif2png(img: str | os.PathLike, png_dir, gamma=0.1):
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
    """
    A generator, from a multi-frame tif file, read frame by frame and return the image and filename of each frame.

    """
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
    """
    Convert image format from uint16 to uint8
    :param __image: uint16 image ndarray
    :return: uint8 image ndarray
    """
    min_16bit = np.min(__image)
    max_16bit = np.max(__image)
    image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def mask_to_coords(mask: np.ndarray | os.PathLike, filename):
    """
    Convert mask to contour coordinates. For the mask of an image, each different instance requires the same and unique
    pixel values within its range, and the pixel values of different instances are different. This requirement is only
    for instances within the same frame of image.

    :param mask: mask grayscale image, np.ndarray
    :param filename: filename for the mask
    :return: a region tmp dict for each mask, the format see template.py
    """
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


def mask_seq_to_json(mask_dir, xrange=None):
    """
    Convert mask sequence files to json annotation files.
    :param mask_dir: The folder path for the mask image sequence location
    :param xrange: The number of conversions, counting from the beginning of the mask sequence.
    :return: json annotation dict, can be directly dump to the json file.
    """
    tif_files = glob.glob(os.path.join(mask_dir, '*.tif'))
    json_result = {}
    if xrange is None:
        xrange = len(tif_files)
    elif xrange > len(tif_files) - 1:
        xrange = len(tif_files)
    for file in tqdm(tif_files[:xrange + 1], desc='convert process'):
        filename = os.path.basename(file).replace('.tif', '.png')
        ret = mask_to_coords(file, filename)
        json_result[filename] = ret
    return json_result


def mask_tif_to_json(image, xrange=None):
    """
    Convert Multi-frame TIF mask image to json annotation files.
    :param image: The TIF filepath for mask location
    :param xrange: The number of conversions, counting from the beginning of the mask.
    :return: json annotation dict, can be directly dump to the json file.
    """
    if not (image.endswith('.tif') or image.endswith('.tiff')):
        logging.error(f'image {image} must be a tif/tiff file !')
        sys.exit(-1)
    json_result = {}
    count = 0
    with tifffile.TiffFile(image) as tif:
        num_frames = len(tif.pages)
    if xrange is None or xrange > num_frames:
        it = num_frames
    else:
        it = xrange
    for img, frame_name in tqdm(readTif(image), total=it, desc='convert process'):
        filename = frame_name.replace('.tif', '.png')
        ret = mask_to_coords(img, filename)
        json_result[filename] = ret
        if xrange:
            if count >= xrange:
                break
        count += 1
    return json_result


def mask_to_json(annotation: 'file or folder', xrange=None):
    """
     Convert  mask image to json annotation files
    :param annotation: mask filepath.
    :param xrange: The number of conversions, counting from the beginning of the annotation sequence.
    :return: json annotation dict, can be directly dump to the json file.
    """
    if os.path.isdir(annotation):
        return mask_seq_to_json(annotation, xrange)
    elif os.path.isfile(annotation):
        return mask_tif_to_json(annotation, xrange)


if __name__ == '__main__':
    # filedir = r'G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\SEG'
    # jsons = mask_seq_to_json(filedir)
    # with open(r'G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\test.json', 'w') as f:
    #     json.dump(jsons, f)
    mask_tif_to_json(r'G:\paper\evaluate_data\copy_of_1_xy10\mcy.tif')
