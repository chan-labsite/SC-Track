import json
import math
import os.path

import numpy as np
import cv2


class DetectionParser:
    def __init__(self, fname: str = None, py_data: dict = None):
        if fname:
            if fname.endswith('.json'):
                with open(fname) as jf:
                    self.__data = json.load(jf)
        elif py_data:
            if type(py_data) is dict:
                self.__data = py_data
        else:
            raise KeyError(f"fname or py_data need once")

    def get_frame_names(self):
        frames = list(self.__data.keys())
        return frames

    def get_regions_by_frame(self, frame_name):
        return self.__data[frame_name]['regions']

    def get_region_attr(self, regions):
        """
        regions: extract from json files, one frame data, it cna be get from function [get_regions_by_frame()]
        Returns: all cells bounding box and cell_type list,
        like [((x_min, x_max, y_min, y_max), cell_type), ..., ((x_min, x_max, y_min, y_max), cell_type)]

        """
        attrs = []
        for i in regions:
            all_x = i['shape_attributes']['all_points_x']
            all_y = i['shape_attributes']['all_points_y']
            phase = i['region_attributes']['cell_type']
            x_min = int(np.min(all_x))
            x_max = math.ceil(np.max(all_x))
            y_min = int(np.min(all_y))
            y_max = math.ceil(np.max(all_y))
            attrs.append([(x_min, x_max, y_min, y_max), phase])
        return attrs


def convert_dtype(__image: np.ndarray) -> np.ndarray:
    min_16bit = np.min(__image)
    max_16bit = np.max(__image)
    image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def draw_bbox(image: np.ndarray, regions_bounding: list):
    if len(image.shape) > 2:
        im_rgb = image
    else:
        im_rgb = cv2.cvtColor(convert_dtype(image), cv2.COLOR_GRAY2RGB)
    _id = 0
    for i in regions_bounding:
        if i[1] == 'G1/G2':
            color = (0, 0, 255)
        elif i[1] == 'S':
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(im_rgb, (i[0][0], i[0][2]), (i[0][1], i[0][3]), color, 2)
        cv2.putText(im_rgb, str(_id), (i[0][0], i[0][2]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        _id += 1

    return im_rgb


def json2box(shapes: dict):
    pass


if __name__ == '__main__':
    pass
    # from tqdm import tqdm
    #
    # file = r"G:\20x_dataset\copy_of_xy_01\copy_of_1_xy01.json"
    # image = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0000.tif'
    # base = r'G:\20x_dataset\copy_of_xy_01\tif\mcy'
    # dp = DetectionParser(fname=file)
    # dp.get_frame_names()
    # # rb = dp.get_region_attr(dp.get_regions_by_frame('copy_of_1_xy01-0000.png'))
    # im = cv2.imread(image, -1)
    # bb_list = []
    # videoWriter = cv2.VideoWriter(r'G:\20x_dataset\copy_of_xy_01\development-dir\result-2-multi-bbox.mp4',
    #                               cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
    #                               5, (2048, 2048), True)
    # for i in tqdm(range(len(dp.get_frame_names()))):
    #     fname = dp.get_frame_names()
    #     img = cv2.imread(os.path.join(base, fname[i].replace('.png', '.tif')), -1)
    #     if i == 0:
    #         rb = dp.get_region_attr(dp.get_regions_by_frame(fname[i]))
    #         rb_before = rb
    #     else:
    #         rb = dp.get_region_attr(dp.get_regions_by_frame(fname[i]))
    #         rb_before = dp.get_region_attr(dp.get_regions_by_frame(fname[i - 1]))
    #     tmp = draw_bbox(img, rb_before)
    #     ret = draw_bbox(tmp, rb)
    #     videoWriter.write(ret)
    # videoWriter.release()
