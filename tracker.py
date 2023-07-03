"""
定义追踪树，追踪节点
每个追踪树root节点起始于第一帧的细胞
起始追踪初始化第一帧识别的细胞数量个树实例

定义树节点， 每个节点包含细胞的详细信息。
以及追踪信息，帧数，分裂信息等


"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import shapely
from libtiff import TIFF

import config

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

import dataclasses
import enum
from copy import deepcopy
from functools import wraps, lru_cache

import warnings
from typing import List
import heapq

import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2
import json

import treelib
from treelib import Tree, Node
from tqdm import tqdm

from utils import convert_dtype, readTif
from base import Cell, Rectangle, Vector, SingleInstance, CacheData, MatchStatus, TreeStatus, CellStatus
from t_error import InsertError, MitosisError, NodeExistError, ErrorMatchMitosis, StatusError
from feature import FeatureExtractor, feature_extract

TEST = False
TEST_INDEX = None
CELL_NUM = 0


class Filter(SingleInstance):
    """
    过滤一帧中距离较远的细胞，降低匹配候选数量
    基本操作数为帧
    过滤依据：bbox的坐标
    参数：
    """

    def __init__(self):
        super(Filter, self).__init__()

    def filter(self, current: Cell, cells: List[Cell]):
        """

        :param current: 待匹配的细胞
        :param cells: 候选匹配项
        :return: 筛选过的候选匹配项
        """
        return [cell for cell in cells if cell in current]


class Checker(object):
    """检查器，检查参与匹配的细胞是否能够计算"""
    _protocols = [None, 'calcIoU', 'calcCosDistance', 'calcCosSimilar', 'calcEuclideanDistance',
                  'compareDicSimilar', 'compareMcySimilar', 'compareShapeSimilar']

    def __init__(self, protocol=None):
        self.protocol = protocol

    def check(self):
        pass

    def __call__(self, method):

        if self.protocol not in self._protocols:
            warnings.warn(f"Don't support protocol: {self.protocol}, now just support {self._protocols}")
            _protocol = None
        else:
            _protocol = self.protocol

        @wraps(method)
        def wrapper(*args, **kwargs):
            """args[0]: object of func; args[1]: param 1 of func method; args[2]: param 2 of func method"""
            # print(_protocol)
            # print('args:', *args)
            if _protocol is None:
                return method(*args, **kwargs)
            elif _protocol == 'calcIoU':
                return method(*args, **kwargs)
            elif _protocol == 'calcCosDistance':
                return method(*args, **kwargs)
            elif _protocol == 'calcCosSimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareDicSimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareMcySimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareShapeSimilar':
                return method(*args, **kwargs)
            else:
                return method(*args, **kwargs)

        return wrapper


class CellNode(Node):
    """
    追踪节点，包含细胞的tracking ID，以及细胞自身的详细信息，和父子节点关系
    """
    _instance_ = {}
    STATUS = ['ACCURATE', 'ACCURATE-FL', 'INACCURATE', 'INACCURATE-MATCH', 'PREDICTED']

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instance_:
            cls._instance_[key] = super().__new__(cls)
            cls._instance_[key].status = None
            cls._instance_[key].track_id = None
            cls._instance_[key].__branch_id = None
            cls._instance_[key].parent = None
            cls._instance_[key].childs = []
            cls._instance_[key].add_tree = False  # 如果被添加到TrackingTree中，设置为True
            cls._instance_[key].life = 5  # 每个分支初始生命值为5，如果没有匹配上，或者利用缺省值填充匹配，则-1，如如果生命值为0，则该分支不再参与匹配
            cls._instance_[key]._init_flag = False
        return cls._instance_[key]

    def __init__(self, cell: Cell, node_type='cell', fill_gap_index=None):
        if not self._init_flag:
            self.cell = cell
            if node_type == 'gap':
                assert fill_gap_index is not None
            super().__init__(cell)
            self._init_flag = True

    @property
    def identifier(self):
        return str(id(self.cell))

    @property
    def nid(self):
        return self.identifier

    def _set_identifier(self, nid):
        if nid is None:
            self._identifier = str(id(self.cell))
        else:
            self._identifier = nid

    def get_status(self):
        return self.status

    def set_parent(self, parent: CellNode):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_childs(self, child: CellNode):
        self.childs.append(child)

    def get_childs(self):
        return self.childs

    def set_tree_status(self, status: TreeStatus):
        self.tree_status = status
        self.add_tree = True

    def get_tree_status(self):
        if self.add_tree:
            return self.tree_status
        return None

    def set_status(self, status):
        if status in self.STATUS:
            self.status = status
        else:
            raise ValueError(f"set error status: {status}")

    def get_branch_id(self):
        return self.__branch_id

    def set_branch_id(self, branch_id):
        self.__branch_id = branch_id
        self.cell.set_branch_id(branch_id)

    def get_track_id(self):
        if self.track_id is None:
            raise ValueError("Don't set the track_id")
        else:
            return self.track_id

    def set_track_id(self, track_id):
        self.track_id = track_id

    def __repr__(self):
        if self.add_tree:
            return f"Cell Node of {self.cell}, status: {self.get_tree_status()}"
        else:
            return f"Cell Node of {self.cell}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return int(id(self))


class TrackingTree(Tree):
    """追踪树，起始的时候初始化根节点，逐帧扫描的时候更新left子节点，发生分裂的时候添加right子节点"""

    def __init__(self, root: CellNode = None, track_id=None):
        super().__init__()
        self.root = root
        self.track_id = track_id
        self.mitosis_start_flag = False
        self.status = TreeStatus(self)
        self.__last_layer = []
        self._exist_branch_id = []
        self._available_branch_id = 1
        self.m_counter = 5

    def __contains__(self, item):
        return item.identifier in self._nodes

    def change_mitosis_flag(self, flag: bool):
        """当细胞首次进入mitosis的时候，self.mitosis_start_flag设置为True， 当细胞完成分裂的时候，重新设置为false"""
        self.mitosis_start_flag = flag

    def add_node(self, node: CellNode, parent: CellNode = None):
        node.set_parent(parent)
        if parent != None:
            parent.childs.append(node)
        super().add_node(node, parent)

    def get_parent(self, node: CellNode):
        return node.get_parent()

    def get_childs(self, node: CellNode):
        return node.get_childs()

    @property
    def last_layer(self):
        return self.__last_layer

    @property
    def last_layer_cell(self):
        """返回{叶节点：节点包含的细胞}字典"""
        cells = {}
        for node in self.leaves():
            cells[node.cell] = node
        return cells

    def update_last_layer(self, node_list: List[CellNode]):
        self.__last_layer = node_list

    def auto_update_last_layer(self):
        self.__last_layer = self.leaves()

    def branch_id_distributor(self):
        if self._available_branch_id not in self._exist_branch_id:
            self._exist_branch_id.append(self._available_branch_id)
            self._available_branch_id += 1
            return self._available_branch_id - 1
        else:
            i = 1
            while True:
                if self._available_branch_id + i not in self._exist_branch_id:
                    self._available_branch_id += (i + 1)
                    return self._available_branch_id + i
                i += 1


class Match(object):
    """
    匹配器，根据前后帧及当前帧来匹配目标并分配ID
    主要用来进行特征比对，计算出相似性
    操作单位：帧
    """
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def normalize(self, x, _range=(0, np.pi / 2)):
        """将值变换到区间[0, π/2]"""
        return _range[0] + (_range[1] - _range[0]) * x

    def calcIoU_roughly(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的交并比
        返回值范围: float(0-1)
        """
        rect1 = Rectangle(*cell_1.bbox)
        rect2 = Rectangle(*cell_2.bbox)

        if not rect1.isIntersect(rect2):
            return 0
        elif rect1.isInclude(rect2):
            return 1
        else:
            intersection = Rectangle(min(rect1.x_min, rect2.x_min), max(rect1.x_max, rect2.x_max),
                                     min(rect1.y_min, rect2.y_min), max(rect1.y_max, rect2.y_max))
            union = Rectangle(max(rect1.x_min, rect2.x_min), min(rect1.x_max, rect2.x_max),
                              max(rect1.y_min, rect2.y_min), min(rect1.y_max, rect2.y_max))
            return union.area / intersection.area

    def calcIoU(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的交并比
        返回值范围: float(0-1)
        """
        poly1 = cell_1.polygon
        poly2 = cell_2.polygon

        # 计算两个多边形的交集
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                intersection = poly1.intersection(poly2)
                # 计算两个多边形的并集
                union = poly1.union(poly2)
        except shapely.errors.GEOSException:
            return self.calcIoU_roughly(cell_1, cell_2)
        if union.area == 0:
            return 0
        elif poly1.contains(poly2) or poly2.contains(poly1):
            return 1
        else:
            return intersection.area / union.area

    def calcCosDistance(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的余弦距离
        返回值范围: float[0, 2]
        距离越小越相似，通过反正切函数缩放到[0, 1)
        返回值为归一化[0, π/2]之后的余弦值，返回值越小，表示相似度越低
        """
        dist = cell_1.vector.cosDistance(cell_2.vector)
        return np.cos(np.arctan(dist) / (np.pi / 2))

    def calcCosSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的中心点的余弦相似度
        返回值范围: float[0, 1]
        值越大越相似
        返回值为归一化[0, π/2]后的正弦值
        """
        score = cell_1.vector.cosSimilar(cell_2.vector)
        return np.sin(self.normalize(score))

    def calcAreaSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的面积相似度
        返回值范围: float[0, 1]
        """
        return min(cell_1.area, cell_2.area) / max(cell_1.area, cell_2.area)

    def calcEuclideanDistance(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞中心点的欧氏距离
        返回值范围: float(0,∞)
        距离越小越相似，通过反正切函数缩放到[0, 1)
        返回值为归一化[0, π/2]之后的余弦值，返回值越小，表示相似度越低
        """
        dist = cell_1.vector.EuclideanDistance(cell_2.vector)
        # return np.cos(np.arctan(dist) / (np.pi / 2))
        return dist

    def compareDicSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        比较dic相似度
        返回值范围: float(0, 1)
        """
        dic1 = Vector(cell_1.feature.dic_intensity, cell_1.feature.dic_variance)
        dic2 = Vector(cell_2.feature.dic_intensity, cell_2.feature.dic_variance)
        if dic1 and dic2:
            return np.sin(self.normalize(dic1.cosSimilar(dic2)))
        else:
            return 0

    def compareMcySimilar(self, cell_1: Cell, cell_2: Cell):
        """
        比较mcy相似度
        返回值范围: float(0, 1)
        """
        mcy1 = Vector(cell_1.feature.mcy_intensity, cell_1.feature.mcy_variance)
        mcy2 = Vector(cell_2.feature.mcy_intensity, cell_2.feature.mcy_variance)
        if mcy1 and mcy2:
            return np.sin(self.normalize(mcy1.cosSimilar(mcy2)))
        else:
            return 0

    def compareShapeSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的轮廓相似度
        返回值范围: float(0, 1)
        score值越小表示相似度越大，可进行取反操作
        返回值为归一化[0, π/2]后的余弦值
        """
        score = cv2.matchShapes(cell_1.contours, cell_2.contours, 1, 0.0)
        # return np.cos(self.normalize(score))
        return score

    def __str__(self):
        return f"Match object at {id(self)}"


class Matcher(object):
    """选定当前帧细胞，匹配前一帧最佳选项，根据下一帧匹配上一帧"""

    def __init__(self):
        self.matcher = Match()
        self.WEIGHT = {'IoU': 0.6, 'shape': 0.1, 'area': 0.3}

    def draw_bbox(self, bg1, bbox, track_id=None):
        if len(bg1.shape) > 2:
            im_rgb1 = bg1
        else:
            im_rgb1 = cv2.cvtColor(convert_dtype(bg1), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_rgb1, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      [255, 255, 0], 2)
        cv2.putText(im_rgb1, str(track_id), (bbox[3], bbox[1]), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 1)
        return im_rgb1

    def predict_next_position(self, parent: Cell):
        """根据速度预测子细胞可能出现的位置，利用预测的子细胞参与匹配"""
        # new_cell = parent.move(parent.move_speed, 1)
        new_cell = parent
        return new_cell

    def _filter(self, child: Cell, cells: List[Cell]):

        # filtered_candidates = [cell for cell in cells if cell in child]
        filtered_candidates = []
        for cell in cells:
            if cell in child and cell.is_accurate_matched is False:
                if self.matcher.calcEuclideanDistance(cell, child) < child.r_long or \
                        self.matcher.calcIoU(cell, child) > 0.2:
                    filtered_candidates.append(cell)

        return filtered_candidates

    def match_candidates(self, child: Cell, before_cell_list: List[Cell]):
        """匹配候选项"""
        return self._filter(child, before_cell_list)

    def calc_similar(self, parent, child_cell):
        similar = [self.matcher.calcIoU(parent, child_cell),
                   self.matcher.calcAreaSimilar(parent, child_cell),
                   self.matcher.compareShapeSimilar(parent, child_cell),
                   self.matcher.calcEuclideanDistance(parent, child_cell),
                   self.matcher.calcCosDistance(parent, child_cell),
                   self.matcher.calcCosSimilar(parent, child_cell),
                   self.matcher.compareDicSimilar(parent, child_cell),
                   self.matcher.compareMcySimilar(parent, child_cell)]
        np.set_printoptions(precision=6, floatmode='fixed')
        return np.array(similar)

    @lru_cache(maxsize=None)
    def match_similar(self, cell_1: Cell, cell_2: Cell):
        similar = {'IoU': self.matcher.calcIoU(cell_1, cell_2),
                   'shape': self.matcher.compareShapeSimilar(cell_1, cell_2),
                   'area': self.matcher.calcAreaSimilar(cell_1, cell_2),
                   'distance': self.matcher.calcEuclideanDistance(cell_1, cell_2)
                   }
        return similar

    def match_duplicate_child(self, parent, unmatched_child_list):
        """
        调用这个函数，意味着候选项中不止一个，此方法计算每个候选项的匹配度
        返回值为{Cell: similar_dict}形式的字典
        """
        matched = {}
        for i in unmatched_child_list:
            # if i.is_be_matched :
            #     if i.status.exist_mitosis_time < 50:
            #         continue
            similar = self.match_similar(parent, i)
            matched[i] = similar
        return matched

    def is_mitosis_start(self, pre_parent: Cell, last_leaves: List[Cell], area_size_t=1.5, iou_t=0.5):
        """判断细胞是否进入M期，依据是细胞进入Mitosis的时候，体积会变大
        :returns 如果成功进入M期，返回包含最后一帧的G2和第一帧M的字典信息， 否则，返回False
        """
        match_score = {}
        for i in last_leaves:
            if self.match_similar(pre_parent, i).get('IoU') >= iou_t:
                match_score[i] = self.match_similar(pre_parent, i)
        for child_cell in match_score:
            # if Rectangle(*parent.bbox).isInclude(Rectangle(*child_cell.bbox)) or (
            #         (child_cell.area / parent.area) >= area_size_t):
            if (child_cell.area / pre_parent.area) >= area_size_t:
                return {'last_G2': pre_parent, 'first_M': child_cell}
        return False

    def get_similar_sister(self, parent: Cell, matched_cells_dict: dict, area_t=0.7, shape_t=0.03, area_size_t=1.3,
                           iou_t=0.1):
        """在多个候选项中找到最相似的两个细胞作为子细胞"""
        cell_dict_keys = list(matched_cells_dict.keys())
        cell_dict_keys.sort(key=lambda cell: cell.area, reverse=True)

        for cell in cell_dict_keys:
            # if matched_cells_dict[cell].get('IoU') < iou_t:
            if matched_cells_dict[cell].get('IoU') == 0:
                cell_dict_keys.remove(cell)
        # if len(cell_dict_keys) > 2:
        #     if cell_dict_keys[0].area + cell_dict_keys[1].area > parent.area * area_size_t:
        #         cell_dict_keys.remove(cell_dict_keys[0])
        if len(cell_dict_keys) > 2:
            remove_dict_list = list(matched_cells_dict).sort(key=lambda x: x[1]['IoU'], reverse=True)[2:]
            for i in remove_dict_list:
                key = list(i.keys())[0]
                cell_dict_keys.remove(key)

        length = len(cell_dict_keys)
        match_result = {}
        for i in range(length - 1):
            cell_1 = cell_dict_keys.pop(0)
            for j in range(len(cell_dict_keys)):
                cell_2 = cell_dict_keys[j]
                score = self.match_similar(cell_1, cell_2)
                # if score.get('area') >= area_t and score.get('shape') <= shape_t:
                if score.get('area') >= area_t:
                    match_result[(cell_1, cell_2)] = score.get('area') + score.get('IoU')
        if match_result:
            max_score_cells = max(match_result, key=match_result.get)
            return max_score_cells
        else:
            raise MitosisError('cannot match the suitable daughter cells')

    def select_mitosis_cells(self, parent: Cell, candidates_child_list: List[Cell], area_t=0.7, shape_t=0.05,
                             area_size_t=1.3):
        """
        如果发生有丝分裂，选择两个子细胞， 调用这个方法的时候，确保大概率发生了分裂事件
        如果返回值为有丝分裂，则需要检查两个子细胞的面积是否正常，如果某一个子细胞的面积过大，则认为是误判

        :return ([cell_1, cell2], 'match status')
        """

        matched_candidates = self.match_duplicate_child(parent, candidates_child_list)
        checked_candidates = {}
        for i in matched_candidates:
            if i.is_be_matched:
                if i.status.exist_mitosis_time < 50:
                    continue
            checked_candidates[i] = matched_candidates[i]
        matched_cells_dict = self.check_iou(checked_candidates)
        # matched_cells_dict = self.check_iou(matched_candidates)
        if not matched_cells_dict:
            raise MitosisError('not enough candidates')
        else:
            if len(matched_cells_dict) == 2:
                cells = list(matched_cells_dict.keys())
                # if max([i.area for i in
                #         list(matched_cells_dict.keys())]) > parent.area * area_size_t:  # 如果母细胞太小了，不认为会发生有丝分裂，转向单项判断
                #     raise MitosisError('The cell is too small to have cell division !')
                # if self.matcher.calcAreaSimilar(cells[0], cells[1]) > area_t and self.matcher.compareShapeSimilar(
                #         cells[0], cells[1]) < shape_t:
                if self.matcher.calcAreaSimilar(cells[0], cells[1]) > area_t:
                    return cells, 'ACCURATE'
                else:
                    # raise MitosisError("not enough candidates, after matched.")
                    return cells, 'INACCURATE'
            else:
                # max_two = heapq.nlargest(2, [sum(sm) for sm in matched_cells_dict.values()])  # 找到匹配结果中最大的两个值
                try:
                    max_two = self.get_similar_sister(parent, matched_cells_dict)  # 找到匹配结果中最大的两个值
                except MitosisError:
                    raise MitosisError("not enough candidates, after matched.")
                better_sisters = []
                for i in max_two:
                    better_sisters.append(i)
                if self.matcher.calcAreaSimilar(better_sisters[0],
                                                better_sisters[1]) > area_t and self.matcher.compareShapeSimilar(
                    better_sisters[0], better_sisters[1]) < shape_t:
                    return better_sisters, 'ACCURATE'
                else:
                    return better_sisters, 'INACCURATE'

    def select_single_child(self, score_dict):
        """
        对于有多个IOU匹配的选项，选择相似度更大的那一个, 此处是为了区分发生重叠的细胞，而非发生有丝分裂.
        注意：这个方法匹配出来的结果不一定是准确的，有可能因为细胞交叉导致发生错配，需要在后面的流程中解决
        另外，如果一个细胞被精确匹配后，另一个细胞在没有匹配项的时候（即在识别过程中，下一帧没有识别上，可能会出现重复匹配）
        这种情况原本应该填充预测细胞。

        """

        def calc_weight(candidate_score_dict):
            result = {}
            # for cell in candidate_score_dict:
            #     score_dict = candidate_score_dict[cell]
            # value =  score_dict['IoU'] * self.WEIGHT.get('IoU') + \
            #        score_dict['shape'] * self.WEIGHT.get('shape') + score_dict['area'] * self.WEIGHT.get('area')
            selected_cell = None
            max_iou = 0.0
            min_distance = 100
            threshold = 0.5
            iou_above_threshold = False

            # 遍历match字典中的每个cell和score_dict
            for cell, score_dict in candidate_score_dict.items():
                iou = score_dict['IoU']
                distance = score_dict['distance']

                # 如果iou大于阈值，更新最大iou和选取的cell，并将iou_above_threshold设置为True
                if iou > threshold:
                    iou_above_threshold = True
                    if distance < min_distance:
                        selected_cell = cell
                        min_distance = distance
                        max_iou = iou
                # 如果iou不大于阈值，但是大于当前最大iou，则更新最大iou和选取的cell
                elif iou > max_iou:
                    selected_cell = cell
                    max_iou = iou
                    min_distance = distance

            # 如果没有任何一个score_dict的iou大于阈值，则选取iou值最大的那个cell
            if not iou_above_threshold:
                for cell, score_dict in candidate_score_dict.items():
                    iou = score_dict['IoU']
                    if iou > max_iou:
                        selected_cell = cell
                        max_iou = iou
            # 返回选取的cell
            return selected_cell
            # return max(result, key=result.get)

        candidates = {}
        for cell in score_dict:
            if score_dict[cell].get('IoU') > 0.5:
                candidates[cell] = score_dict[cell]
        if not candidates:
            # print("第二分支，重新选择")
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0.1:
                    candidates[cell] = score_dict[cell]
        if not candidates:
            # print("第二分支，重新选择")
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0.0:
                    candidates[cell] = score_dict[cell]

        if not candidates:
            # print("第三分支，重新选择")
            for cell in score_dict:
                candidates[cell] = sum(score_dict[cell].values())
            try:
                best = max(candidates, key=candidates.get)
            except ValueError:
                print(score_dict)
        else:
            best = calc_weight(candidates)
        return best

    def match_one(self, predict_child, candidates):
        if len(candidates) == 1:  # 只有一个候选项，判断应该为准确匹配
            # print('matched single:', self.calc_similar(parent, filtered_candidates[0]))
            score = self.calc_similar(predict_child, candidates[0])
            if score[0] > 0.5:
                return [(candidates[0], 'ACCURATE')]
            elif score[0] > 0:
                return [(candidates[0], 'INACCURATE')]
            else:
                return None
        else:
            return False

    def check_iou(self, similar_dict):
        """检查IoU，为判断进入有丝分裂提供依据，如果拥有IoU>0的匹配项少于2，返回False，否则，返回这两个细胞"""
        matched = {}
        for cell in similar_dict:
            score = similar_dict[cell]
            if score.get('IoU') > 0:
                matched[cell] = score
        if len(matched) < 2:
            return False
        else:
            return matched

    def _match(self, parent: Cell, filter_candidates_cells: List[Cell], cell_track_status: TreeStatus):
        """比较两个细胞的综合相似度
        """
        # predict_child = self.predict_next_position(parent)
        predict_child = parent
        # filtered_candidates = self.match_candidates(predict_child, no_filter_candidates_cells)
        filtered_candidates = [cell for cell in filter_candidates_cells if cell.is_accurate_matched is False]
        # print(filtered_candidates)
        if len(filtered_candidates) > 0:
            if not self.match_one(predict_child, filtered_candidates):  # 不只有一个选项
                if self.match_one(predict_child, filtered_candidates) is None:
                    return
                matched_candidates = self.match_duplicate_child(predict_child, filtered_candidates)
                if len(matched_candidates) > 1:
                    cell_track_status.enter_mitosis(parent.frame)
                if not cell_track_status.status.get('enter_mitosis'):
                    # if parent.phase != 'M':
                    return {'matched_cell': [(self.select_single_child(matched_candidates), 'INACCURATE')],
                            'status': cell_track_status}
                # elif not self.check_iou(matched_candidates):
                #     return {'matched_cell': [(self.select_single_child(matched_candidates), 'ACCURATE')],
                #             'status': cell_track_status}
                else:
                    matched_result = []
                    try:
                        sisters, status = self.select_mitosis_cells(parent, filtered_candidates)  # 此时细胞一分为二
                        cell_track_status.exit_mitosis(parent.frame + 1)
                        for sister in sisters:
                            matched_result.append((sister, status))

                    except MitosisError as M:  # 细胞可能仍然处于M期，但是已经完成分开，或者只是被误判为M期
                        matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                        # print(M)
                    except ErrorMatchMitosis as M2:
                        # 细胞可能不均等分裂
                        matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                        # print(M2)
                    finally:
                        return {'matched_cell': matched_result, 'status': cell_track_status}
            else:
                return {'matched_cell': self.match_one(predict_child, filtered_candidates), 'status': cell_track_status}

    def calc_sorted_value(self, parent: Cell, matched_cell):
        """计算Cell对象的排序值"""

        match_score = self.match_similar(parent, matched_cell)
        sort_value = match_score['IoU'] + 1 / (match_score['distance'] + 1e-5)
        return sort_value

    def add_child_node(self, tree, child_node: CellNode, parent_node: CellNode):
        try:
            tree.add_node(child_node, parent=parent_node)
            child_node.set_tree_status(tree.status)
        # except TypeError as E:
        #     print(E)
        # except NodeExistError as E2:
        #     print(E2)
        except treelib.exceptions.DuplicatedNodeIdError:
            pass

    def match_single_cell(self, tree: TrackingTree, current_frame: FeatureExtractor):
        """追踪单个细胞的变化情况"""
        cells = current_frame.cells
        parents = tree.last_layer_cell
        m_counter = 5
        if len(parents) > 1:
            parent_dict = {parent: parent.frame for parent in parents}
            keys = [p for p in parents]
            if parent_dict[keys[0]] != parent_dict[keys[1]]:
                min_parent = min(parent_dict, key=parent_dict.get)
                parents.pop(min_parent)
        for parent in parents:
            # 分裂后的两个细胞或者多个细胞
            # print(f'\nparent cell math status: {parent.is_be_matched}')
            tree.m_counter -= 1
            if not tree.m_counter:
                tree.m_counter = 5
                tree.status.reset_M_count()
            if parent.phase == 'M':
                tree.status.add_M_count()
            if tree.status.predict_M_len >= 2:
                tree.status.enter_mitosis(parent.frame - 3)

            # predict_child = self.predict_next_position(parent)
            predict_child = parent

            filtered_candidates = self.match_candidates(predict_child, cells)
            # filtered_candidates = list(candidates)
            before_parent = tree.get_parent(parents[parent])
            if before_parent:
                if self.is_mitosis_start(before_parent.cell, [predict_child]):
                    # if self.is_mitosis_start(predict_child, filtered_candidates):
                    tree.status.enter_mitosis(parent.frame)
            match_result = self._match(predict_child, filtered_candidates, tree.status)
            if match_result is not None:
                child_cells = match_result.get('matched_cell')
                for i in child_cells:
                    sort_value = self.calc_sorted_value(parent, i[0])
                    i[0].sort_value = sort_value
                parent_node = CellNode(parent)
            else:
                continue
            if len(child_cells) == 1:
                # if child_cells[0][1] == 'PREDICTED':
                #     current_frame.add_cell(child_cells[0][0])
                #     child_node = CellNode(child_cells[0][0])
                #     child_node.life -= 1
                if child_cells[0][1] == 'ACCURATE':
                    # candidates.remove(child_cells[0][0])
                    child_cells[0][0].is_accurate_matched = True
                    child_node = CellNode(child_cells[0][0])
                else:
                    child_node = CellNode(child_cells[0][0])
                    # child_cells[0][0].is_accurate_matched = True
                # child_node.set_branch_id(parent_node.get_branch_id())
                child_node.cell.set_branch_id(parent_node.cell.branch_id)
                child_node.cell.set_status(tree.status)
                child_node.cell.update_region(track_id=tree.track_id)
                child_node.cell.update_region(branch_id=parent_node.cell.branch_id)
                child_node.cell.set_match_status(child_cells[0][1])
                # if child_node.life > 0:
                self.add_child_node(tree, child_node, parent_node)
                # child_node.branch_id = parent_node.branch_id
            else:
                #     try:
                #         assert len(child_cells) == 2
                #     except AssertionError:
                #         # self._match(predict_child, filtered_candidates, tree.status)
                #         continue

                for cell in child_cells:
                    new_branch_id = tree.branch_id_distributor()
                    cell[0].set_branch_id(new_branch_id)
                    cell[0].update_region(track_id=tree.track_id)
                    cell[0].update_region(branch_id=new_branch_id)
                    cell[0].set_match_status(cell[1])
                    child_node = CellNode(cell[0])
                    if cell[1] == 'ACCURATE':
                        cell[0].is_accurate_matched = True
                        # candidates.remove(cell[0])
                    self.add_child_node(tree, child_node, parent_node)
        tree.status.add_exist_time()


class Tracker(object):
    """Tracker对象，负责逐帧扫描图像，进行匹配并分配track id，初始化并更新TrackingTree"""

    def __init__(self, annotation, mcy=None, dic=None):
        self.matcher = Matcher()
        self.fe_cache = deque(maxlen=5)
        self.trees: List[TrackingTree] = []
        self.mcy = mcy
        self.dic = dic
        self.annotation = annotation
        self._exist_tree_id = []
        self._available_id = 0
        self.init_flag = False
        self.feature_ext = feature_extract(mcy=self.mcy, dic=self.dic, jsonfile=self.annotation)
        self.tree_maps = {}
        self.init_tracking_tree(next(self.feature_ext)[0])
        self.nodes = set()
        self.count = 0
        self.parser_dict = None

    def id_distributor(self):
        if self._available_id not in self._exist_tree_id:
            self._exist_tree_id.append(self._available_id)
            self._available_id += 1
            return self._available_id - 1
        else:
            i = 1
            while True:
                if self._available_id + i not in self._exist_tree_id:
                    self._available_id += (i + 1)
                    return self._available_id + i
                i += 1

    def init_tracking_tree(self, fe: FeatureExtractor):
        """初始化Tracking Tree"""
        # trees = []
        for i in fe.cells:
            tree = TrackingTree(track_id=self.id_distributor())
            i.sort_value = 1
            i.set_match_status('ACCURATE')
            i.set_track_id(tree.track_id, 1)
            i.set_branch_id(0)
            i.set_cell_id(str(i.track_id) + '-' + str(i.branch_id))
            i.update_region(track_id=tree.track_id)
            i.update_region(branch_id=0)
            node = CellNode(i)
            node.set_branch_id(0)
            node.set_track_id(tree.track_id)
            node.status = 'ACCURATE'
            i.set_match_status('ACCURATE')
            i.is_accurate_matched = True
            node.cell.set_status(TreeStatus(tree))
            tree.add_node(node)
            self.trees.append(tree)
            self.tree_maps[i] = tree
        self.init_flag = True
        # self.trees = trees
        self.fe_cache.append(fe)

    def draw_bbox(self, bg1, cell: Cell, track_id, branch_id=None, phase=None):
        bbox = cell.bbox
        if len(bg1.shape) > 2:
            im_rgb1 = bg1
        else:
            im_rgb1 = cv2.cvtColor(convert_dtype(bg1), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_rgb1, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      [0, 255, 0], 2, )

        def get_str():
            raw = str(track_id)
            if branch_id is not None:
                raw += '-'
                raw += str(branch_id)
            if phase is not None:
                raw += '-'
                raw += str(phase)
            return raw

        text = get_str()
        cv2.putText(im_rgb1, text, (bbox[3], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 1)

        return im_rgb1

    @staticmethod
    def update_speed(parent: Cell, child: Cell, default: Vector = None):
        """更新细胞的移动速度"""
        if default:
            child.update_speed(Vector(0, 0))
        else:
            speed = (child.vector - parent.vector)
            child.update_speed(speed)

    def update_tree_map(self, cell_key, tree_value):
        """更新tree_map"""
        # 废弃的参数，不再依赖tree_map

    def get_current_tree(self, parent_cell: Cell):
        """获取当前母细胞存在的TrackingTree"""
        exist_trees = []  # 细胞可能存在的tree
        for tree in self.trees:
            if parent_cell in tree.last_layer_cell:
                exist_trees.append(tree)
        return exist_trees

    def add_node(self, child_node, parent_node, tree):
        if child_node not in tree:
            # tree.add_node(child_node, parent=parent_node.identifier)
            tree.add_node(child_node, parent=parent_node)
            child_node.cell.set_status(CellStatus(tree))
        else:
            raise NodeExistError(child_node)

    def track_near_frame(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """匹配临近帧"""
        cells = sorted(fe1.cells, key=lambda cell: cell.sort_value, reverse=True)
        for parent in cells:
            # print(parent)
            trees = self.get_current_tree(parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

    def track_near_frame_mult_thread(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """多线程测试版"""

        def work(__parent: Cell):
            trees = self.get_current_tree(__parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

        thread_pool_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="track_")
        cells = sorted(fe1.cells, key=lambda cell: cell.sort_value, reverse=True)
        # print([i.sort_value for i in cells])
        for parent in cells:
            thread_pool_executor.submit(work, parent)
        thread_pool_executor.shutdown(wait=True)

    def handle_duplicate_match(self, duplicate_match_cell):
        """解决一个细胞被多个细胞重复匹配"""
        child_node = CellNode(duplicate_match_cell)
        tmp = self.get_current_tree(duplicate_match_cell)
        parent0 = tmp[0].parent(child_node.nid)

        parent1 = tmp[1].parent(child_node.nid)
        # if not (parent0.is_root() and parent1.is_root()):
        #     parent00 = tmp[0].parent(parent0.nid)
        #     parent11 = tmp[1].parent(parent1.nid)
        #     tree_dict = {parent00: tmp[0], parent11: tmp[1]}
        #     sm0 = self.matcher.match_similar(duplicate_match_cell, parent00.cell)
        #     sm1 = self.matcher.match_similar(duplicate_match_cell, parent11.cell)
        #     # match_score = {parent00: sm0['IoU'] + 1 / (sm0['distance'] + 1e-5),
        #     #                parent11: sm1['IoU'] + 1 / (sm0['distance'] + 1e-5), }
        #     match_score = {parent00: sm0['distance'],
        #                    parent11: sm0['distance']}
        #     error_parent = max(match_score)
        # else:
        tree_dict = {parent0: tmp[0], parent1: tmp[1]}
        sm0 = self.matcher.match_similar(duplicate_match_cell, parent0.cell)
        sm1 = self.matcher.match_similar(duplicate_match_cell, parent1.cell)
        # match_score = {parent0: sm0['IoU'] + 1 / (sm0['distance'] + 1e-5),
        #                parent1: sm1['IoU'] + 1 / (sm0['distance'] + 1e-5), }
        match_score = {parent0: sm0['distance'],
                       parent1: sm1['distance']}
        # truth_parent = max(match_score)
        error_parent = min(match_score)
        # if len(tree_dict[truth_parent].nodes) < 3:
        #     error_parent = truth_parent
        tree_dict[error_parent].remove_node(child_node.nid)
        return {error_parent: tree_dict[error_parent]}

    def rematch(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """对于发生漏检的细胞，将其上下帧与缓存帧进行重新匹配"""
        unmatched_list = [cell for cell in fe1.cells if cell.is_be_matched is False]
        if not unmatched_list:
            return
        # for cell in fe1.cells:
        for cell in unmatched_list:
            current_frame = cell.frame
            if cell.is_be_matched is False:
                handle_flag = False
                trees = self.trees
                wait_dict = {}
                wait_tree_map = {}
                for tree in trees:
                    last_layer_cells = [cell for cell in tree.last_layer_cell]
                    for last_layer_cell in last_layer_cells:
                        if 1 < current_frame - last_layer_cell.frame < 4:
                            match_result = self.matcher.match_similar(last_layer_cell, cell)
                            if match_result['IoU'] > 0 or match_result['distance'] < cell.d_long:
                                wait_dict[last_layer_cell] = match_result['IoU'] + 10 / (
                                            match_result['distance'] + 1e-5)
                                wait_tree_map[last_layer_cell] = tree
                                handle_flag = True
                    if (not handle_flag):
                        for last_layer_cell in last_layer_cells:
                            if config.GAP_WINDOW_LEN:
                                if 1 < current_frame - last_layer_cell.frame < config.GAP_WINDOW_LEN:
                                    match_result = self.matcher.match_similar(last_layer_cell, cell)
                                    if match_result['distance'] < 50:
                                        wait_dict[last_layer_cell] = match_result['IoU'] + 10 / (
                                                    match_result['distance'] + 1e-5)
                                        wait_tree_map[last_layer_cell] = tree
                                        handle_flag = True
                            else:
                                if current_frame > last_layer_cell.frame:
                                    match_result = self.matcher.match_similar(last_layer_cell, cell)
                                    if match_result['distance'] < 50:
                                        wait_dict[last_layer_cell] = match_result['IoU'] + 10 / (
                                                    match_result['distance'] + 1e-5)
                                        wait_tree_map[last_layer_cell] = tree
                                        handle_flag = True
                if wait_dict:
                    matched_cell = max(wait_dict, key=wait_dict.get)
                    tree = wait_tree_map[matched_cell]
                    child_node = CellNode(cell)
                    parent_node = CellNode(matched_cell)
                    tree.add_node(child_node, parent_node)
                    cell.is_accurate_matched = True
                    cell.is_be_matched = True
                    cell.set_status(tree.status)
                    cell.set_match_status('ACCURATE')
                    cell.set_track_id(tree.track_id, 1)
                    cell.set_branch_id(matched_cell.branch_id)
                    self.matcher.match_single_cell(tree, fe2)
                if not handle_flag:
                    tree = TrackingTree(track_id=self.id_distributor())
                    cell.set_track_id(tree.track_id, 1)
                    cell.set_branch_id(0)
                    cell.set_cell_id(str(cell.track_id) + '-' + str(cell.branch_id))
                    cell.update_region(track_id=tree.track_id)
                    cell.update_region(branch_id=0)
                    node = CellNode(cell)
                    node.set_branch_id(0)
                    node.set_track_id(tree.track_id)
                    node.status = 'ACCURATE'
                    tree.add_node(node)
                    self.trees.append(tree)
                    self.tree_maps[cell] = tree
                    cell.set_match_status('ACCURATE')
                    cell.set_status(CellStatus(tree))
                    cell.is_accurate_matched = True
                    cell.is_be_matched = True
                    self.matcher.match_single_cell(tree, fe2)

    def calc_weight(self, candidate_score_dict):
        result = {}
        # for cell in candidate_score_dict:
        #     score_dict = candidate_score_dict[cell]
        # value =  score_dict['IoU'] * self.WEIGHT.get('IoU') + \
        #        score_dict['shape'] * self.WEIGHT.get('shape') + score_dict['area'] * self.WEIGHT.get('area')
        selected_cell = None
        max_iou = 0.0
        min_distance = 100
        threshold = 0.6
        iou_above_threshold = False
        # 遍历match字典中的每个cell和score_dict
        for cell, score_dict in candidate_score_dict.items():
            iou = score_dict['IoU']
            distance = score_dict['distance']
            # 如果iou大于阈值，更新最大iou和选取的cell，并将iou_above_threshold设置为True
            if iou > threshold:
                iou_above_threshold = True
                if distance < min_distance:
                    selected_cell = cell
                    min_distance = distance
                    max_iou = iou
            # 如果iou不大于阈值，但是大于当前最大iou，则更新最大iou和选取的cell
            elif iou > max_iou:
                selected_cell = cell
                max_iou = iou
                min_distance = distance
        # 如果没有任何一个score_dict的iou大于阈值，则选取iou值最大的那个cell
        if not iou_above_threshold:
            for cell, score_dict in candidate_score_dict.items():
                iou = score_dict['IoU']
                if iou > max_iou:
                    selected_cell = cell
                    max_iou = iou
        # 返回选取的cell
        return selected_cell
        # return max(result, key=result.get)

    def check_track(self, fe1: FeatureExtractor, fe2: FeatureExtractor, fe3: FeatureExtractor):
        """检查track结果，查看是否有错误匹配和遗漏， 同时更新匹配状态"""
        cells = sorted(fe2.cells, key=lambda cell: cell.sort_value, reverse=True)
        self.rematch(fe1, fe2)
        for cell in cells:
            tmp = self.get_current_tree(cell)
            if len(tmp) > 1:
                self.handle_duplicate_match(duplicate_match_cell=cell)

    def track(self, range=None, speed_filename=None):
        """顺序读取图像帧，开始追踪"""
        global writer, speed_f
        index = 0
        if speed_filename:
            speed_f = open(speed_filename, 'w', newline='')
            writer = csv.writer(speed_f)
            writer.writerow(['Iteration', 'Speed (it/s)'])
        for fe_before, fe_current, fe_next in tqdm(self.feature_ext, total=range, desc='tracking process'):
            # self.track_near_frame(fe_before, fe_current)
            start_time = time.time()
            self.track_near_frame(fe_before, fe_current)
            # self.track_near_frame_mult_thread(fe_before, fe_current)
            self.fe_cache.append(fe_before)
            self.check_track(fe_before, fe_current, fe_next)

            end_time = time.time()
            if speed_filename:
                writer.writerow([index, f"{1 / (end_time - start_time + 1e-10):.2f}"])
            if range:
                index += 1
                if index > range:
                    break
        if speed_filename:
            speed_f.close()

        # __filter_trees = [tree for tree in self.trees if len(tree.nodes) > 10]
        # del self.trees
        # self.trees = __filter_trees

    def track_tree_to_json(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        for i in self.trees:
            jsf = os.path.join(filepath, f'tree-{i.track_id}.json')
            if os.path.exists(jsf):
                os.remove(jsf)
            i.save2file(jsf)

    def visualize_single_tree(self, tree, background_filename_list, save_dir, xrange=None):
        bg_fname = background_filename_list[:xrange + 2] if xrange else background_filename_list
        print(bg_fname)
        images = list(map(lambda x: cv2.imread(x, -1), bg_fname))
        images_dict = dict(zip(list(range(len(bg_fname))), images))
        print(images_dict.keys())
        for node in tree.expand_tree():
            frame = tree.nodes.get(node).cell.frame
            bbox = tree.nodes.get(node).cell.bbox
            img_bg = images_dict[frame]
            phase = tree.nodes.get(node).cell.phase
            images_dict[frame] = self.draw_bbox(img_bg, tree.nodes.get(node).cell, tree.track_id,
                                                tree.get_node(node).cell.branch_id, phase)
        for i in zip(bg_fname, list(images_dict.values())):
            fname = os.path.join(save_dir, os.path.basename(i[0]).replace('.tif', '.png'))
            print(fname)
            cv2.imwrite(fname, i[1])

    def visualize_to_tif(self, background_mcy_image: str, output_tif_path, tree_list, xrange=None, single=False):
        def adjust_gamma(__image, gamma=1.0):
            image = convert_dtype(__image)
            brighter_image = np.array(np.power((image / 255), 1 / gamma) * 255, dtype=np.uint8)
            return brighter_image

        tif = readTif(background_mcy_image)
        images_dict = {}
        index = 0

        if xrange is not None:
            for img, _ in tif:
                if index >= xrange:
                    break
                img = adjust_gamma(img, gamma=1.5)
                images_dict[index] = img
                index += 1
        else:
            try:
                for img, _ in tif:
                    images_dict[index] = img
                    index += 1
            except KeyError:
                pass
        for i in tree_list:
            for node in i.expand_tree():
                frame = i.nodes.get(node).cell.frame
                if xrange:
                    if frame > xrange:
                        continue
                # bbox = i.nodes.get(node).cell.bbox
                img_bg = images_dict.get(frame)
                if img_bg is not None:
                    images_dict[frame] = self.draw_bbox(img_bg, i.nodes.get(node).cell, i.track_id,
                                                        i.get_node(node).cell.branch_id,
                                                        phase=i.get_node(node).cell.phase)
        if not single:
            if not (os.path.exists(output_tif_path) and os.path.isdir(output_tif_path)):
                os.mkdir(output_tif_path)
            for i in tqdm(range(index), desc="save tracking visualization"):
                fname = os.path.join(output_tif_path, f'{os.path.basename(output_tif_path)[:-4]}-{i:0>4d}.tif')
                tifffile.imwrite(fname, images_dict[i])
        else:
            with tifffile.TiffWriter(output_tif_path) as tif:
                for i in tqdm(range(index)):
                    if i > 300:
                        warnings.warn(
                            "the image is to big to save, and the tifffile cannot save the size >4GB tifffile, "
                            "so this image will be cut down.")
                        break
                    tif.write(images_dict[i])


def get_cell_line_from_tree(tree: TrackingTree, dic_path: str, mcy_path: str, savepath):
    """从track tree中获取完整的细胞序列，包括细胞图像，dic和mcy双通道，以及周期，生成的文件名以track_id-branch_id-frame-phase.tif命名"""
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save_mcy = os.path.join(savepath, 'mcy')
    save_dic = os.path.join(savepath, 'dic')
    if not os.path.exists(save_mcy):
        os.mkdir(save_mcy)
    if not os.path.exists(save_dic):
        os.mkdir(save_dic)
    mcy = tifffile.imread(mcy_path)
    dic = tifffile.imread(dic_path)
    for nid in tree.expand_tree():
        cell = tree.get_node(nid).cell
        y0, y1, x0, x1 = cell.bbox
        mcy_img = mcy[cell.frame][y0: y1, x0: x1]
        dic_img = dic[cell.frame][y0: y1, x0: x1]
        fname = str(tree.track_id) + '-' + str(cell.branch_id) + '-' + str(cell.frame) + '-' + str(
            cell.phase[0]) + '.tif'
        tifffile.imwrite(os.path.join(save_mcy, fname), convert_dtype(mcy_img))
        tifffile.imwrite(os.path.join(save_dic, fname), convert_dtype(dic_img))

        # break


if __name__ == '__main__':
    annotation = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\result-GT.json'
    mcy_img = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\mcy.tif'
    dic_img = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\dic.tif'
    tracker = Tracker(annotation)
    # tracker = Tracker(r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\result-GT.json')
    tracker.track(300)
    for i in enumerate(tracker.trees):
        get_cell_line_from_tree(i[1], dic_img, mcy_img,
                                fr'G:\20x_dataset\evaluate_data\copy_of_1_xy19\cell_lines\{i[0]}')
    # tracker.track_tree_to_json(r'G:\20x_dataset\copy_of_xy_01\development-dir\track_tree\tree5')
    # tracker.save_visualize(200)
    # for i in tracker.trees:
    #     print(i)
    #     print(i.nodes)
    #     node_r = i.nodes[list(i.nodes.keys())[0]]
    #     print(node_r)
    #     node_n = CellNode(i.nodes[list(i.nodes.keys())[0]].cell)
    #     print(node_n)
    #     break

    # track_jiaqi()
    # test()

    # c1 = CellNode(Cell(position=([1, 2], [3, 4])))
    # c2 = CellNode(Cell(position=([1, 3], [3, 4])))
    # c3 = CellNode(Cell(position=([1, 1], [3, 4])))
    # tree = TrackingTree()
    # tree.add_node(c1)
    # tree.add_node(c2, parent=c1)
    # tree.add_node(c3, parent=c1)
    # print(tree)
    # print(len(tree))
    # print(tree.nodes)
