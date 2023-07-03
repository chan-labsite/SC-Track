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
    每个细胞实例所包含的所有特征，包括以下内容(后续待更新)：
    - area: 细胞面积
    - bbox_area: bounding box area
    - shape: 细胞轮廓坐标序列
    - center：细胞中心坐标
    - vector: 细胞相对于原点的向量
    - bbox: 细胞的bounding box坐标[y_min, y_max, x_min, x_max]
    - dic_intensity: mask区域的dic灰度值强度 ?平均值，最大值最小值，像素强度分布范围，分布曲线
    - mcy_intensity: mask区域的mcy灰度值强度
    - phase 预测周期

    归一化0-255后，概率阈值：
        dic通道: 平均像素强度: M期>=120，G1/G2/S期<120, 方差: M期<=200，, G1/G2/S期>200
        mcy通道: 平均像素强度: M期<70, G1/G2<70, S>70  方差: G1/G2/M < 300, S>300

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

    def feature(self):
        """
        返回细胞特征
        :return:
        """
        pass


class FeatureExtractor(object):
    """提取单帧图像中每个细胞的可用特征"""
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

        Args:
            image_dic: np.ndarray dic图像信息 2048x2048
            image_mcy:  np.ndarray mcy图像信息 2048x2048
            annotation: dict 细胞的轮廓和周期等注释信息, json文件中的regions
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
                if config.USING_IMAGE_FOR_TRACKING is True: # config 配置using image，但是参数不合规，关闭此选项
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
        """根据轮廓坐标绘制mask, 如果只传入一组轮廓坐标值，请务必将其置于列表中传入函数，
        例如， coord = ([x1 x2 ... xn], [y1 y2 ... yn]),调用时请按照coordinate2mask([coord])调用
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
        # 基于向量叉乘计算多边形面积
        area = 0
        if len(points) < 3:
            raise Exception("error")

        for i in range(0, len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            triArea = (p1.x * p2.y - p2.x * p1.y) / 2
            area += triArea
        return abs(area)

    def sort_point(self, points):
        """将顶点进行排序"""
        pass

    def area(self, cell):
        """细胞面积"""
        _area = cv2.contourArea(self.coord2counter(cell.position))
        return _area

    def bbox(self, cell: Cell):
        """bounding box坐标"""
        x0 = math.floor(np.min(cell.position[0])) if math.floor(np.min(cell.position[0])) > 0 else 0
        x1 = math.ceil(np.max(cell.position[0]))
        y0 = math.floor(np.min(cell.position[1])) if math.floor(np.min(cell.position[1])) > 0 else 0
        y1 = math.ceil(np.max(cell.position[1]))
        return y0, y1, x0, x1

    def get_roi_from_coord(self, cell: Cell, image: np.ndarray):
        """
        利用细胞轮廓获取dic图像或者mcy图像，视传入的image参数不同而不同
        :param cell: Cell对象
        :param image: dic图像或者mcy图像，即参数self.mcy或者self.dic
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
        获取单帧图像中所有细胞
        """
        cell_list = []
        for region in self.annotation:
            try:
                all_x = region['shape_attributes']['all_points_x']
                all_y = region['shape_attributes']['all_points_y']
                all_x = [0 if i < 0 else i for i in all_x]
                all_y = [0 if j < 0 else j for j in all_y]
                phase = region['region_attributes'].get('phase')
                phase = None
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
                    phase = None
                    all_x, all_y = self.ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
                    cell = Cell(position=(all_x, all_y), phase=phase, frame_index=self.frame_index)
                    cell.set_region(region)
                    cell_list.append(cell)
                else:
                    print(region)
        return cell_list

    def get_cell_image(self, cell: Cell):
        """
        获取细胞的dic图像和mcy图像
        :param cell:
        :return:
        """
        if config.USING_IMAGE_FOR_TRACKING:
            dic = self.get_roi_from_coord(cell, self.dic)
            mcy = self.get_roi_from_coord(cell, self.mcy)
            return dic, mcy
        else:
            return None

    def set_cell_image(self, cell: Cell):
        """
        为细胞实例设置dic信息和mcy信息
        :param cell: Cell对象
        :return: 包含图像信息的Cell对象
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
        :param cell: Cell对象
        :return: Feature对象
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
        """将图像从uint16转化为uint8"""
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
    """逐帧返回FeatureExtractor实例，包含当前帧，前一帧，后一帧"""
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



def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def normal_distribution(x, y0, A, w, xc):
    # y = (1 / (np.sqrt(2 * np.pi) * sigma)) * (np.exp(-(((x - mu) ** 2) / (2 * sigma ** 2))))
    # return y
    y = y0 + (A / (w * np.sqrt(np.pi / 2))) * np.exp(-2 * ((x - xc) / w) ** 2)
    return y


if __name__ == '__main__':
    import csv
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error as mse

    from scipy.stats import norm
    from sklearn.mixture import GaussianMixture
    from sklearn.gaussian_process import GaussianProcessRegressor

    test_dic_image = r'G:\20x_dataset\copy_of_xy_01\tif\dic\copy_of_1_xy01-0000.tif'
    test_mcy_image = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0000.tif'
    test_dic_image2 = r'G:\20x_dataset\copy_of_xy_01\tif\dic\copy_of_1_xy01-0001.tif'
    test_mcy_image2 = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0001.tif'
    ann = r'G:\20x_dataset\copy_of_xy_01\copy_of_1_xy01.json'
    with open(ann) as f:
        data = json.load(f)
    regions = data['copy_of_1_xy01-0000.png']['regions']
    regions2 = data['copy_of_1_xy01-0001.png']['regions']
    dic = cv2.imread(test_dic_image, -1)
    mcy = cv2.imread(test_mcy_image, -1)
    dic2 = cv2.imread(test_dic_image2, -1)
    mcy2 = cv2.imread(test_mcy_image2, -1)
    fe = FeatureExtractor(image_dic=dic, image_mcy=mcy, annotation=regions)
    fe2 = FeatureExtractor(image_dic=dic, image_mcy=mcy, annotation=regions2)
    print(len(fe2.cells))
    fe2.add_cell(Cell(position=([1, 1], [2, 2])))
    print(len(fe2.cells))

    fe3 = FeatureExtractor(image_dic=dic, image_mcy=mcy, annotation=regions2)
    print(fe2 is fe3)
    print(len(fe3.cells))

    for i in fe.cells:
        print(i)
    print(fe.get_cell_list())
    print(fe2.get_cell_list())

    cell1 = None
    cell2 = None
    cell_ctrl = None
    for i in fe.get_cell_list():
        if i.phase == 'M':
            fe.set_cell_image(i)
            cell1 = i
            break
    for j in fe2.get_cell_list():
        if j.phase == 'S':
            fe.set_cell_image(j)
            cell2 = j
            break
    Ms = []
    for k in fe2.get_cell_list():
        if k.phase == 'M':
            fe2.set_cell_image(k)
            Ms.append(k)
    print(cell1)
    print(cell2)
    # fe.show(cell1.mcy)
    # fe.show(cell2.mcy)
    for m in Ms:
        fe.show(m.mcy)
    # print("M and S:  ", cv2.matchShapes(cell1.counter, cell2.counter, 1, 0.0))
    # for c in Ms:
    #     print(f"{c} M and M: ", cv2.matchShapes( c.counter, cell2.counter, 1, 0.0))
    # fi = 0
    #
    # g = []
    # m = []
    # s = []
    # img_rgb = cv2.cvtColor(fe.convert_dtype(mcy), cv2.COLOR_GRAY2RGB)
    # for i in fe.cells:
    #     # print(i.mcy)
    #     n, bins, patches = plt.hist(i.mcy.ravel(), 256)
    #     px = []
    #     hz = []
    #     for z in zip(n, bins):
    #         if int(z[0]) != 0:
    #             px.append(z[1])
    #             hz.append(z[0])
    #     z1 = np.polyfit(px, hz, 10)
    #     p1 = np.poly1d(z1)
    #     if i.phase == 'G1/G2':
    #         g.append(z1)
    #     elif i.phase == 'S':
    #         s.append(z1)
    #     else:
    #         m.append(z1)
    #
    #     plt.xlim(0, 256)
    #     plt.title(i.phase)
    #     plt.show()
    #     fe.show(i.mcy)
    #
    #     test = i.mcy.copy()
    #     test[:10, :15] = 0
    #     print(i.center)
    #     # cv2.arrowedLine(img_rgb, (0, 0), (int(i.center[0]), int(i.center[1])), (0, 0, 255), tipLength=0.01)
    #     cv2.putText(img_rgb, f"{int(fe.area(i))}", (int(i.center[0]), int(i.center[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
    #                 (255, 255, 255), 1)
    #     # fe.show(mcy)
    #     fi += 1
    #
    #     print(fe.area(i))
    #
    # cv2.imwrite(r"G:\20x_dataset\copy_of_xy_01\development-dir\arrow-cv2area.png", img_rgb)
    # if fi > 3:
    #     break

# if __name__ == '__main__':
#     mcy = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_mcy\M\00b0df672c2e9ad9ed02fe0a4b3c6693.tif'
#     mcy2 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_mcy\M\00b0df672c2e9ad9ed02fe0a4b3c6693-hflip.tif'
#     dic = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_dic\M\00b0df672c2e9ad9ed02fe0a4b3c6693.tif'
#     dic2 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_dic\M\00b0df672c2e9ad9ed02fe0a4b3c6693-hflip.tif'
#
#     mcy3 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_mcy\G\000cf1fdafddb1312bebb5e1febc2395.tif'
#     dic3 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_dic\G\000cf1fdafddb1312bebb5e1febc2395.tif'
#
#     mcy4 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_mcy\M\00b0df672c2e9ad9ed02fe0a4b3c6693-rotate15.tif'
#     dic4 = r'G:\20x_dataset\copy_of_xy_01\train_data_dev1.3.2\AAAaugment_dic\M\00b0df672c2e9ad9ed02fe0a4b3c6693-rotate15.tif'
#
#     test1 = get_test_data(mcy, dic)
#     test2 = get_test_data(mcy2, dic2)
#
#     test3 = get_test_data(mcy3, dic3)
#     test4 = get_test_data(mcy4, dic4)
#
#     model = resnet_50_for_feature()
#     model2 = resnet_50()
#     # model.build(input_shape=(None, 100, 100, 2))
#     model.load_weights(r'../../models/classify/20x/best/model')
#     model2.load_weights(r'../../models/classify/20x/best/model')
#     f1 = model.predict(test1)
#     f2 = model.predict(test2)
#     f3 = model.predict(test3)
#     f4 = model.predict(test4)
#     print(type(f1))
#     print(f1)
#     print(f2)
#     print("--" * 100)
#     print(model2.predict(test1))
#     print(model2.predict(test2))
#     print(model2.predict(test3))
#     print(model2.predict(test4))
#
#     print(mse(f1, f2))  # <0.1
#     print(mse(f1, f3))  # >0.1
#     print(mse(f2, f3))  # >0.1
#     print(mse(f2, f4))  # <0.1
#     print(mse(f3, f4))  # >0.1
#
#     print("余弦相似度 f1 f2: ", tf.keras.losses.cosine_similarity(f1, f2))
#     print("余弦相似度 f1 f3: ", tf.keras.losses.cosine_similarity(f1, f3))
#     print("余弦相似度 f2 f3: ", tf.keras.losses.cosine_similarity(f2, f3))
#     print("余弦相似度 f2 f4: ", tf.keras.losses.cosine_similarity(f2, f4))
#     print("余弦相似度 f3 f4: ", tf.keras.losses.cosine_similarity(f3, f4))
#
#     print("余弦相似度 f1 f2: ", np.mean(tf.keras.losses.cosine_similarity(test1, test2)))
#     print("余弦相似度 f1 f3: ", np.mean(tf.keras.losses.cosine_similarity(test1, test3)))
#     print("余弦相似度 f2 f3: ", np.mean(tf.keras.losses.cosine_similarity(test2, test3)))
#     print("余弦相似度 f2 f4: ", np.mean(tf.keras.losses.cosine_similarity(test2, test4)))
#     print("余弦相似度 f3 f4: ", np.mean(tf.keras.losses.cosine_similarity(test3, test4)))
#
#     # model.summary()
