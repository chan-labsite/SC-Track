from __future__ import annotations

import os
from functools import lru_cache
import math
from typing import Tuple, List
import cv2
import json

import matplotlib.pyplot as plt
import numpy as np
from tifffile import tifffile

import config
from base import Cell,  Vector


class Mask(object):
    def __init__(self, mask=None, center=None, coord=None):
        self._mask = mask
        self._center = center
        self.id = None
        self.frame_index = None
        self.coord = coord

    @property
    def mask(self):
        if self._mask is None:
            raise ValueError("No mask")
        return self._mask

    @property
    def center(self) -> Tuple[int | float]:
        if self._center is None:
            raise ValueError("No available center position")
        return self._center

    def __str__(self):
        return f"Mask object at center of {self._center[0], self._center[1]}"

    def __repr__(self):
        return self.__str__()


class Feature(object):
    """
    All features contained in each cell instance, including the following:
     - area: cell area
     - bbox_area: bounding box area
     - shape: sequence of cell outline coordinates
     - center: cell center coordinates
     - vector: the vector of the cell relative to the origin
     - bbox: the bounding box coordinates of the cell [y_min, y_max, x_min, x_max]
     - dic_intensity: the dic gray value intensity of the mask area
     - mcy_intensity: mcy gray value intensity of the mask area
     - phase forecast period


    """

    def __init__(self, center, bbox, area=None, shape=None, phase=None,
                 dic_intensity=None, dic_variance=None, mcy_intensity=None, mcy_variance=None, frame=None):
        self.center = center
        self.bbox = bbox
        self.phase = phase
        self.mcy_intensity = mcy_intensity
        self.mcy_variance = mcy_variance
        self.dic_intensity = dic_intensity
        self.dic_variance = dic_variance
        self.shape = shape
        self.area = area
        self.vector = Vector(*center)
        self.frame_id = frame




class FeatureExtractor(object):
    """Extract available features for each cell in a single image"""
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super(FeatureExtractor, cls).__new__(cls)
            cls._instances[key]._init_cell_flag = False
            cls._instances[key]._init_flag = False
            cls._instances[key].__cells = None
        return cls._instances[key]

    def __init__(self, image_dic: np.ndarray | None = None, image_mcy: np.ndarray | None = None, annotation: dict = None,
                 *args, **kwargs):
        """
        image_dic: np.ndarray dic image information 2048x2048
        image_mcy: np.ndarray mcy image information 2048x2048
        annotation: dict Annotation information such as the outline and cycle of cells, regions in the json file
        """
        if not self._init_flag:
            # config.USING_IMAGE_FOR_TRACKING = config.USING_IMAGE_FOR_TRACKING
            self.frame_id = None
            if (image_mcy is not None) and  (image_dic is not None):
                if type(image_mcy) != np.uint8:
                    self.mcy = self.convert_dtype(image_mcy)
                else:
                    self.mcy = image_mcy
                if type(image_dic) != np.uint8:
                    self.dic = self.convert_dtype(image_dic)
                else:
                    self.dic = image_dic
            else:
                # config.py configures using image, but the parameters are not compliant, turn off this option
                if config.USING_IMAGE_FOR_TRACKING is True:
                    config.USING_IMAGE_FOR_TRACKING = False
            self.annotation = annotation
            # self.image_shape = self.mcy.shape
            self.frame_index = kwargs.get('frame_index')
            self._init_flag = True

    def coord2counter(self, coord):
        points = []
        for j in range(len(coord[0])):
            x = int(coord[0][j])
            y = int(coord[1][j])
            points.append((x, y))
        contours = np.array(points)
        return contours

    def coordinate2mask(self, coords: np.ndarray | list | tuple, shape, value: int = 255) -> \
            List[Mask]:
        """
        Draw the mask according to the contour coordinates. If you only pass in a set of contour coordinate values,
        be sure to put them in the list and pass in the function.
         For example, coord = ([x1 x2 ... xn], [y1 y2 ... yn]), please call it according to coordinate2mask([coord])
        """
        results = []
        for coord in coords:
            mask = np.zeros(shape, dtype=np.uint8)
            points = []
            for j in range(len(coord[0])):
                x = int(coord[0][j])
                y = int(coord[1][j])
                points.append((x, y))
            contours = np.array(points)
            cv2.fillConvexPoly(mask, contours, (value, 0, 0))
            mMask = Mask(mask, center=(round(float(np.mean(coord[0]))), round(float(np.mean(coord[1])))), coord=coord)
            results.append(mMask)
        return results

    def get_roi_from_mask(self, mask, image):
        pass

    def GetAreaByVector(self, points: List):
        # Calculate polygon area based on vector cross product
        area = 0
        if len(points) < 3:
            raise Exception("error")

        for i in range(0, len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            triArea = (p1.x * p2.y - p2.x * p1.y) / 2
            area += triArea
        return abs(area)


    def area(self, cell):
        """cell area"""
        _area = cv2.contourArea(self.coord2counter(cell.position))
        return _area

    def bbox(self, cell: Cell):
        """bounding box coordinates"""
        x0 = math.floor(np.min(cell.position[0])) if math.floor(np.min(cell.position[0])) > 0 else 0
        x1 = math.ceil(np.max(cell.position[0]))
        y0 = math.floor(np.min(cell.position[1])) if math.floor(np.min(cell.position[1])) > 0 else 0
        y1 = math.ceil(np.max(cell.position[1]))
        return y0, y1, x0, x1

    def get_roi_from_coord(self, cell: Cell, image: np.ndarray):
        """
        Use the cell outline to obtain the dic image or the mcy image, depending on the incoming image parameters.
        :param cell: Cell object
        :param image: dic image or mcy image, that is, the parameter self.mcy or self.dic
        :return: roi np.ndarray
        """
        # x0 = int(np.min(cell.position[0]))
        # x1 = math.ceil(np.max(cell.position[0]))
        # y0 = int(np.min(cell.position[1]))
        # y1 = math.ceil(np.max(cell.position[1]))
        y0, y1, x0, x1 = self.bbox(cell)
        return image[y0: y1, x0: x1]

    def ellipse_points(self, center, rx, ry, num_points=100, theta=0):
        all_x = []
        all_y = []
        for i in range(num_points):
            t = i * 2 * np.pi / num_points
            x = center[0] + rx * np.cos(t) * np.cos(theta) - ry * np.sin(t) * np.sin(theta)
            y = center[1] + rx * np.cos(t) * np.sin(theta) + ry * np.sin(t) * np.cos(theta)
            all_x.append(x)
            all_y.append(y)
        return all_x, all_y

    @lru_cache(maxsize=None)
    def get_cell_list(self):
        """
        Get all cells in a single frame image
        """
        cell_list = []
        for region in self.annotation:
            try:
                all_x = region['shape_attributes']['all_points_x']
                all_y = region['shape_attributes']['all_points_y']
                all_x = [0 if i < 0 else i for i in all_x]
                all_y = [0 if j < 0 else j for j in all_y]
                phase = region['region_attributes'].get('phase')
                # phase = None
                cell = Cell(position=(all_x, all_y), phase=phase, frame_index=self.frame_index)
                cell.set_region(region)
                cell_list.append(cell)
            except KeyError:
                # print(region)
                if region['shape_attributes'].get('name') == 'ellipse':
                    rx = region['shape_attributes'].get('rx')
                    ry = region['shape_attributes'].get('ry')
                    cx = region['shape_attributes'].get('cx')
                    cy = region['shape_attributes'].get('cy')
                    theta = region['shape_attributes'].get('theta')
                    phase = region['region_attributes'].get('phase')
                    # phase = None
                    all_x, all_y = self.ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
                    cell = Cell(position=(all_x, all_y), phase=phase, frame_index=self.frame_index)
                    cell.set_region(region)
                    cell_list.append(cell)
                else:
                    print(region)
        return cell_list

    def get_cell_image(self, cell: Cell):
        """
        Get dic and mcy images of cells
        """
        if config.USING_IMAGE_FOR_TRACKING:
            dic = self.get_roi_from_coord(cell, self.dic)
            mcy = self.get_roi_from_coord(cell, self.mcy)
            return dic, mcy
        else:
            return None

    def set_cell_image(self, cell: Cell):
        """
        Set dic information and mcy information for cell instance
        :param cell: Cell object
        :return: Cell object containing image information
        """
        if config.USING_IMAGE_FOR_TRACKING:
            dic, mcy = self.get_cell_image(cell)
            cell.dic = dic
            cell.mcy = mcy
        return cell


    @property
    @lru_cache(maxsize=None)
    def _cells(self):
        cells = self.get_cell_list()
        if config.USING_IMAGE_FOR_TRACKING:
            for cell in cells:
                self.set_cell_image(cell)
        return cells

    def extract(self, cell: Cell) -> Feature:
        """
        :param cell: Cell object
        :return: Feature object
        """
        if config.USING_IMAGE_FOR_TRACKING:
            mcy_intensity = np.mean(cell.mcy)
            mcy_variance = np.std(cell.mcy) ** 2
            dic_intensity = np.mean(cell.dic)
            dic_variance = np.std(cell.dic) ** 2
        else:
            mcy_intensity = mcy_variance = dic_intensity = dic_variance = None
        feature = Feature(center=cell.center, bbox=self.bbox(cell), shape=cell.position,
                          mcy_intensity=mcy_intensity, mcy_variance=mcy_variance,
                          dic_intensity=dic_intensity, dic_variance=dic_variance)

        return feature

    @lru_cache(maxsize=None)
    def __cells_(self):
        _cells = []
        for cell in self._cells:
            feature = self.extract(cell)
            cell.set_feature(feature)
            _cells.append(cell)
        return _cells

    @property
    @lru_cache(maxsize=None)
    def cells(self):
        if self._init_cell_flag is False:
            self.__cells = self.__cells_()
            self._init_cell_flag = True
            return self.__cells
        else:
            return self.__cells

    def add_cell(self, new_cell: Cell):
        self.__cells.append(new_cell)

    def convert_dtype(self, __image: np.ndarray) -> np.ndarray:
        """Convert image from uint16 to uint8"""
        min_16bit = np.min(__image)
        max_16bit = np.max(__image)
        image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    @staticmethod
    def show(image):
        plt.imshow(image, cmap='gray')
        plt.show()

    def get_regions(self):
        return self.annotation

    def __str__(self):
        return f'Feature Extractor of {self.frame_index if self.frame_index else 0} at {id(self)}'

    def __repr__(self):
        return self.__str__()


def imread(filepath: str | os.PathLike) -> np.ndarray:
    return tifffile.imread(filepath)


def get_frame_by_index(image: np.ndarray, index: int) -> np.ndarray:
    return image[index]


def feature_extract(mcy, dic, jsonfile: str | dict):
    """
    Return the FeatureExtractor instance frame by frame,
    including the current frame, the previous frame, and the next frame
    """
    if type(jsonfile) is str:
        with open(jsonfile, encoding='utf-8') as f:
            annotations = json.load(f)
    elif type(jsonfile) is dict:
        annotations = jsonfile
    else:
        raise TypeError(f"type {type(jsonfile)} are not invalid")
    if mcy and dic:
        _dic = imread(dic)
        _mcy = imread(mcy)
        _frame_len = _mcy.shape[0]
        using_image = True
    else:
        _frame_len = len(annotations)
        using_image = False

    def get_fe(frame_index, frame_name, using_image=False):
        if using_image:
            dic_image = get_frame_by_index(_dic, frame_index)
            mcy_image = get_frame_by_index(_mcy, frame_index)
        else:
            dic_image = mcy_image = None
        region = annotations[frame_name.replace('.tif', '.png')]['regions']
        return FeatureExtractor(image_dic=dic_image, image_mcy=mcy_image, annotation=region, frame_index=frame_index)

    def get_base_name(annotation, index):
        return list(annotation.keys())[index]

    for i in range(_frame_len):
        current_frame_index = i
        if i == 0:
            before_frame_index = 0
        else:
            before_frame_index = i - 1
        if i == _frame_len - 1:
            after_frame_index = i
        else:
            after_frame_index = i + 1
        if using_image:
            before_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(before_frame_index).zfill(4) + '.tif')
            after_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(after_frame_index).zfill(4) + '.tif')
            current_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(current_frame_index).zfill(4) + '.tif')
        else:
            before_frame_name = get_base_name(annotations, before_frame_index)
            current_frame_name = get_base_name(annotations, current_frame_index)
            after_frame_name = get_base_name(annotations, after_frame_index)
        before_fe = get_fe(before_frame_index, before_frame_name, using_image=using_image)
        current_fe = get_fe(current_frame_index, current_frame_name, using_image=using_image)
        after_fe = get_fe(after_frame_index, after_frame_name, using_image=using_image)
        yield before_fe, current_fe, after_fe



if __name__ == '__main__':
    pass
