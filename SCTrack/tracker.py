from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import shapely

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.setrecursionlimit(3000)

from functools import wraps, lru_cache
import warnings
from typing import List
import tifffile
import numpy as np
import cv2

import treelib
from treelib import Tree, Node
from tqdm import tqdm

from SCTrack.utils import convert_dtype, readTif
from SCTrack.base import Cell, Rectangle, Vector, TreeStatus, CellStatus
from SCTrack.t_error import InsertError, MitosisError, NodeExistError, ErrorMatchMitosis, StatusError
from SCTrack.feature import FeatureExtractor, feature_extract
from SCTrack import config

TEST = False
TEST_INDEX = None
CELL_NUM = 0


class Checker(object):
    """
    A checker that checks whether the cells participating in the match can calculate.
    """
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
    Tracking node, including the track_ID, cell_id, branch_id of the cell, and the detailed information of the cell,
    the parent-child node relationship
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
            cls._instance_[key].add_tree = False  # if adding to the TrackingTree，set as True
            cls._instance_[key].life = 5
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
        """The unique ID of a node within the scope of a TrackingTree"""
        return str(id(self.cell))

    @property
    def nid(self):
        """same as identifier, override treelib related methods"""
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

    def set_children(self, child: CellNode):
        self.childs.append(child)

    def get_children(self):
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
    """
    The core structure of tracking implement. All tracking results are stored in the instance of this class.
    Each TrackingTree instance represents the cell line of a cell, and the TrackingTree branch represents cell division.
    """

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
        """
        When the cell enters mitosis for the first time, self.mitosis_start_flag is set to True,
        and when the cell completes division, it is reset to false
        """
        self.mitosis_start_flag = flag

    def add_node(self, node: CellNode, parent: CellNode = None):
        """Add a new CellNode object to the TrackingTree"""
        node.set_parent(parent)
        if parent is not None:
            parent.childs.append(node)
        super().add_node(node, parent)

    def get_parent(self, node: CellNode):
        """Returns the parent CellNode of a node"""
        return node.get_parent()

    def get_childs(self, node: CellNode):
        """Returns the child CellNode of a node"""
        return node.get_children()

    @property
    def last_layer(self):
        """Return all CellNodes in the last layer of TrackingTree"""
        return self.__last_layer

    @property
    def last_layer_cell(self):
        """Return a dict of {leaf node: cells contained in the node}."""
        cells = {}
        for node in self.leaves():
            cells[node.cell] = node
        return cells

    def update_last_layer(self, node_list: List[CellNode]):
        self.__last_layer = node_list

    def auto_update_last_layer(self):
        self.__last_layer = self.leaves()

    def branch_id_distributor(self):
        """
        Used to assign a branch_id to each branch in a TrackingTree object.
        The branch_id is incremented in a non-reversible and non-reusable manner.
        """
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
    Matcher, matches targets based on previous and next frames and assigns IDs in the current frame.
    Mainly used for feature matching and calculating similarity.
    The unit of operation is frames.
    """
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def normalize(self, x, _range=(0, np.pi / 2)):
        """Transforming the value to the range of [0, π/2]."""
        return _range[0] + (_range[1] - _range[0]) * x

    def calcIoU_roughly(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate IoU of two cell bounding boxes.
        return value range: float(0-1)
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
        Calculate IoU of two cells.
        First, try to calculate the IoU of the contour. If it is not possible to calculate,
        then change to calculating the IoU of the bounding box.
        return value range: float(0-1)
        """
        poly1 = cell_1.polygon
        poly2 = cell_2.polygon

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                intersection = poly1.intersection(poly2)
                union = poly1.union(poly2)
                if union.area == 0:
                    return 0
                elif poly1.contains(poly2) or poly2.contains(poly1):
                    return 1
                else:
                    return intersection.area / union.area
        except shapely.errors.GEOSException:
            return self.calcIoU_roughly(cell_1, cell_2)

    def calcCosDistance(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate the cosine distance of two cells
        Return value range: float[0, 2]
        The smaller the distance, the more similar it is, scaled to [0, 1) by the arctangent function
        The return value is the cosine value after normalization [0, π/2].
        The smaller the return value, the lower the similarity
        """
        dist = cell_1.vector.cosDistance(cell_2.vector)
        return np.cos(np.arctan(dist) / (np.pi / 2))

    def calcCosSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate the cosine similarity of the center points of two cells
        Return value range: float[0, 1]
        The larger the value, the more similar
        The return value is the sine value after normalization [0, π/2]
        """
        score = cell_1.vector.cosSimilar(cell_2.vector)
        return np.sin(self.normalize(score))

    def calcAreaSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate the area similarity of two cells
        Return value range: float[0, 1]
        """
        return min(cell_1.area, cell_2.area) / max(cell_1.area, cell_2.area)

    def calcEuclideanDistance(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate the Euclidean distance between two cell center points
        Return value range: float(0,∞)
        The smaller the distance, the more similar it is, scaled to [0, 1) by the arctangent function
        The return value is the cosine value after normalization [0, π/2]. The smaller the return value, the lower the similarity
        """
        dist = cell_1.vector.EuclideanDistance(cell_2.vector)
        # return np.cos(np.arctan(dist) / (np.pi / 2))
        return dist

    def compareDicSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        Compare dic similarity
        Return value range: float(0, 1)
        """
        dic1 = Vector(cell_1.feature.dic_intensity, cell_1.feature.dic_variance)
        dic2 = Vector(cell_2.feature.dic_intensity, cell_2.feature.dic_variance)
        if dic1 and dic2:
            return np.sin(self.normalize(dic1.cosSimilar(dic2)))
        else:
            return 0

    def compareMcySimilar(self, cell_1: Cell, cell_2: Cell):
        """
        Compare mcy similarity
        Return value range: float(0, 1)
        """
        mcy1 = Vector(cell_1.feature.mcy_intensity, cell_1.feature.mcy_variance)
        mcy2 = Vector(cell_2.feature.mcy_intensity, cell_2.feature.mcy_variance)
        if mcy1 and mcy2:
            return np.sin(self.normalize(mcy1.cosSimilar(mcy2)))
        else:
            return 0

    def compareShapeSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        Calculate the contour similarity of two cells
        Return value range: float(0, 1)
        The smaller the score value, the greater the similarity, and the inversion operation can be performed.
        (The return value is the cosine value after normalization [0, π/2], no longer do this change, directly return
        the score.)
        """
        score = cv2.matchShapes(cell_1.contours, cell_2.contours, 1, 0.0)
        # return np.cos(self.normalize(score))
        return score

    def __str__(self):
        return f"Match object at {id(self)}"


class Matcher(object):
    """
    A matcher that realizes the association of front and back frame objects.
    """

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
        """
        According to the speed, the possible position of the daughter cell is predicted, and the predicted daughter cell
        is used to participate in the matching. speed=0 is equivalent to not enabling prediction.
        """
        # new_cell = parent.move(parent.move_speed, 1)
        new_cell = parent
        return new_cell

    def _filter(self, child: Cell, cells: List[Cell]):
        """filter candidates"""

        # filtered_candidates = [cell for cell in cells if cell in child]
        filtered_candidates = []
        for cell in cells:
            if cell in child and cell.is_accurate_matched is False:
                if self.matcher.calcEuclideanDistance(cell, child) < child.r_long or \
                        self.matcher.calcIoU(cell, child) > 0:
                    filtered_candidates.append(cell)
        if not filtered_candidates:
            for cell in cells:
                if cell in child and cell.is_accurate_matched is False:
                    filtered_candidates.append(cell)

        return filtered_candidates

    def match_candidates(self, child: Cell, before_cell_list: List[Cell]):
        """match candidates"""
        return self._filter(child, before_cell_list)

    def calc_similar(self, parent, child_cell):
        """Calculate the similarity of two cells"""
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
        """Calculate the similarity of two cells, do not include the image information."""
        similar = {'IoU': self.matcher.calcIoU(cell_1, cell_2),
                   'shape': self.matcher.compareShapeSimilar(cell_1, cell_2),
                   'area': self.matcher.calcAreaSimilar(cell_1, cell_2),
                   'distance': self.matcher.calcEuclideanDistance(cell_1, cell_2)
                   }
        return similar

    def match_duplicate_child(self, parent, unmatched_child_list):
        """
        Match multiple candidates. Calling this function means that there is more than one candidate, and this method
        calculates the matching degree of each candidate
        The return value is a dictionary in the form of {Cell: similar_dict}
        """
        matched = {}
        for i in unmatched_child_list:
            similar = self.match_similar(parent, i)
            if similar['IoU'] > 0:
                matched[i] = similar
        return matched

    def is_mitosis_start(self, pre_parent: Cell, last_leaves: List[Cell], area_size_t=1.5, iou_t=0.3):
        """
        Determine whether a cell enters the M cell_type. The basic criterion is that when a cell enters mitosis, its volume
        increases and the number of candidate regions increases. If the cell successfully enters the M cell_type, return a
        dict containing information about the last frame of G2 and the first frame of M. Otherwise, return False.
         """
        match_score = {}
        for i in last_leaves:
            if self.match_similar(pre_parent, i).get('IoU') >= iou_t:
                match_score[i] = self.match_similar(pre_parent, i)
        for child_cell in match_score:
            if (child_cell.area / pre_parent.area) >= area_size_t:
                return {'last_G2': pre_parent, 'first_M': child_cell}
        return False

    def get_similar_sister(self, parent: Cell, matched_cells_dict: dict, area_t=0.7, shape_t=0.03, area_size_t=1.3,
                           iou_t=0.1):
        """Find the two most similar cells among multiple candidates as the daughter cells."""
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

    def select_mitosis_cells(self, parent: Cell, candidates_child_list: List[Cell], area_t=0.5, shape_t=0.05,
                             area_size_t=1.3):
        """
        If cell division occurs, select two daughter cells. When calling this method, make sure that cell division is
        highly likely to occur. If the return value is cell division, then it is necessary to check whether the areas
        of the two daughter cells are normal. If the area of one daughter cell is too large, it is considered a FP.

        :return ([cell_1, cell2], 'match status')

        """

        matched_candidates = self.match_duplicate_child(parent, candidates_child_list)
        checked_candidates = {}
        for i in matched_candidates:
            if i.is_be_matched:
                if i.status.exit_mitosis_time < config.ENTER_DIVISION_THRESHOLD:
                    continue
            checked_candidates[i] = matched_candidates[i]
        matched_cells_dict = self.check_iou(checked_candidates)
        if not matched_cells_dict:
            raise MitosisError('not enough candidates')
        else:
            if len(matched_cells_dict) == 2:
                cells = list(matched_cells_dict.keys())
                if self.matcher.calcAreaSimilar(cells[0], cells[1]) > area_t:
                    return cells, 'ACCURATE'
                else:
                    return cells, 'INACCURATE'
            else:
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
        For multiple IOU matching options, choose the one with a higher similarity. This is to distinguish overlapping
        cells rather than cell division. Note: the results of this method are not necessarily accurate and may result
        in mismatches due to cell crossing, which needs to be resolved in subsequent steps. In addition, if one cell is
        accurately matched, and the other cell has no match (i.e., not detected in the next frame during recognition),
        it should be filled as a predicted cell.

        """
        def calc_weight(candidate_score_dict):
            """Compute and dynamically update matching weights and matching results."""
            # for cell in candidate_score_dict:
            #     score_dict = candidate_score_dict[cell]
            # value =  score_dict['IoU'] * self.WEIGHT.get('IoU') + \
            #        score_dict['shape'] * self.WEIGHT.get('shape') + score_dict['area'] * self.WEIGHT.get('area')
            selected_cell = None
            max_iou = 0.0
            min_distance = 100
            threshold = 0.5
            iou_above_threshold = False

            for cell, score_dict in candidate_score_dict.items():
                iou = score_dict['IoU']
                distance = score_dict['distance']

                # If iou is greater than the threshold, update the maximum iou and the selected cell,
                # and set iou_above_threshold to True
                if iou > threshold:
                    iou_above_threshold = True
                    if distance < min_distance:
                        selected_cell = cell
                        min_distance = distance
                        max_iou = iou

                # If the iou is not greater than the threshold, but greater than the current maximum iou,
                # update the maximum iou and the selected cell
                elif iou > max_iou:
                    selected_cell = cell
                    max_iou = iou
                    min_distance = distance

            # If the iou is not greater than the threshold, but greater than the current maximum iou,
            # update the maximum iou and the selected cell
            if not iou_above_threshold:
                for cell, score_dict in candidate_score_dict.items():
                    iou = score_dict['IoU']
                    if iou > max_iou:
                        selected_cell = cell
                        max_iou = iou
            return selected_cell

        candidates = {}
        for cell in score_dict:
            if score_dict[cell].get('IoU') > 0.5:
                candidates[cell] = score_dict[cell]
        if not candidates:
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0.1:
                    candidates[cell] = score_dict[cell]
        if not candidates:
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0.0:
                    candidates[cell] = score_dict[cell]
        if not candidates:
            if not score_dict:
                return None
            for cell in score_dict:
                candidates[cell] = sum(score_dict[cell].values())
            best = max(candidates, key=candidates.get)
            return best
        else:
            best = calc_weight(candidates)
            return best

    def match_one(self, predict_child, candidates):
        if len(candidates) == 1:
            score = self.matcher.calcIoU(predict_child, candidates[0])
            if score > 0.5:
                return [(candidates[0], 'ACCURATE')]
            elif score > 0:
                return [(candidates[0], 'INACCURATE')]
            else:
                return [(candidates[0], 'INACCURATE')]
        else:
            return False

    def check_iou(self, similar_dict):
        """
        Check the IoU to provide a basis for determining cell division. If there are less than 2 matching options
        with IoU > 0, return False. Otherwise, return these two cells.
        """
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
        """
        Compare the overall similarity of two cells.
        """
        predict_child = parent
        # filtered_candidates = self.match_candidates(predict_child, no_filter_candidates_cells)
        filtered_candidates = [cell for cell in filter_candidates_cells if cell.is_accurate_matched is False]
        if len(filtered_candidates) > 0:
            if not self.match_one(predict_child, filtered_candidates):  # 不只有一个选项
                if self.match_one(predict_child, filtered_candidates) is None:
                    return
                matched_candidates = self.match_duplicate_child(predict_child, filtered_candidates)
                if len(matched_candidates) > 1 and cell_track_status.exit_mitosis_time > 25:
                    if config.INCLUDE_CELL_CYCLE is False:
                        cell_track_status.enter_mitosis(parent.frame)
                if not cell_track_status.status.get('enter_mitosis'):
                    if self.select_single_child(matched_candidates) is None:
                        return
                    return {'matched_cell': [(self.select_single_child(matched_candidates), 'INACCURATE')],
                            'status': cell_track_status}
                else:
                    matched_result = []
                    try:
                        sisters, status = self.select_mitosis_cells(parent, filtered_candidates)  # 此时细胞一分为二
                        cell_track_status.exit_mitosis(parent.frame + 1)
                        for sister in sisters:
                            matched_result.append((sister, status))

                    except MitosisError as M:  # 细胞可能仍然处于M期，但是已经完成分开，或者只是被误判为M期
                        if self.select_single_child(matched_candidates):
                            matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                    except ErrorMatchMitosis as M2:
                        # 细胞可能不均等分裂
                        if self.select_single_child(matched_candidates):
                            matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                    finally:
                        if matched_result:
                            return {'matched_cell': matched_result, 'status': cell_track_status}
                        return None
            else:
                return {'matched_cell': self.match_one(predict_child, filtered_candidates), 'status': cell_track_status}

    def calc_sorted_value(self, parent: Cell, matched_cell):
        """Calculate the sorting value of a Cell object."""

        match_score = self.match_similar(parent, matched_cell)
        sort_value = match_score['IoU'] + 1 / (match_score['distance'] + 1e-5)
        return sort_value

    def add_child_node(self, tree, child_node: CellNode, parent_node: CellNode):
        try:
            tree.add_node(child_node, parent=parent_node)
            child_node.set_tree_status(tree.status)
        except treelib.exceptions.DuplicatedNodeIdError:
            pass

    def match_single_cell(self, tree: TrackingTree, current_frame: FeatureExtractor):
        """The implementation logic for tracking a single cell."""

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
            # Two or more cells after cell division.
            tree.m_counter -= 1
            if not tree.m_counter:
                tree.m_counter = 5
                tree.status.reset_M_count()
            if parent.cell_type == 'M':
                tree.status.add_M_count()
            if tree.status.predict_M_len >= 3:
                tree.status.enter_mitosis(parent.frame - 3)
            predict_child = parent
            filtered_candidates = self.match_candidates(predict_child, cells)
            before_parent = tree.get_parent(parents[parent])
            if before_parent:
                if self.is_mitosis_start(before_parent.cell, [predict_child]):
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
                if child_cells[0][1] == 'ACCURATE':
                    child_cells[0][0].is_accurate_matched = True
                    child_node = CellNode(child_cells[0][0])
                else:
                    child_node = CellNode(child_cells[0][0])
                child_node.cell.set_branch_id(parent_node.cell.branch_id)
                child_node.cell.set_status(tree.status)
                child_node.cell.update_region(track_id=tree.track_id)
                child_node.cell.update_region(branch_id=parent_node.cell.branch_id)
                child_node.cell.set_match_status(child_cells[0][1])
                self.add_child_node(tree, child_node, parent_node)
            else:
                for cell in child_cells:
                    new_branch_id = tree.branch_id_distributor()
                    cell[0].set_branch_id(new_branch_id)
                    cell[0].update_region(track_id=tree.track_id)
                    cell[0].update_region(branch_id=new_branch_id)
                    cell[0].set_match_status(cell[1])
                    child_node = CellNode(cell[0])
                    if cell[1] == 'ACCURATE':
                        cell[0].is_accurate_matched = True
                    self.add_child_node(tree, child_node, parent_node)
        tree.status.add_exist_time()


class Tracker(object):
    """
    Tracker object, which is the controller of the tracking process. It reads images frame by frame,
    initializes and updates the TrackingTree, performs matching and assigns various IDs.
    """

    def __init__(self, annotation, mcy=None, dic=None):
        self.matcher = Matcher()
        self.fe_cache = deque(maxlen=5)
        self.trees: List[TrackingTree] = []
        self.mcy = mcy
        self.dic = dic
        self.annotation = annotation
        self._exist_tree_id = []
        self._available_id = 1
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
        """Initialize TrackingTree"""
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
        """Update cell movement speed"""
        if default:
            child.update_speed(Vector(0, 0))
        else:
            speed = (child.vector - parent.vector)
            child.update_speed(speed)

    def get_current_tree(self, parent_cell: Cell):
        """Get the TrackingTree where the current parent cell is located"""
        exist_trees = []  # Possible trees include the cells
        for tree in self.trees:
            if parent_cell in tree.last_layer_cell:
                exist_trees.append(tree)
        return exist_trees

    def add_node(self, child_node, parent_node, tree):
        if child_node not in tree:
            tree.add_node(child_node, parent=parent_node)
            child_node.cell.set_status(CellStatus(tree))
        else:
            raise NodeExistError(child_node)

    def track_near_frame(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """match adjacent frames"""
        cells = sorted(fe1.cells, key=lambda cell: cell.sort_value, reverse=True)
        for parent in cells:
            trees = self.get_current_tree(parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

    def track_near_frame_mult_thread(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """Match Adjacent Frames, Multithreaded Beta"""
        def work(__parent: Cell):
            trees = self.get_current_tree(__parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

        thread_pool_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="track_")
        cells = sorted(fe1.cells, key=lambda cell: cell.sort_value, reverse=True)
        for parent in cells:
            thread_pool_executor.submit(work, parent)
        thread_pool_executor.shutdown(wait=True)

    def handle_duplicate_match(self, duplicate_match_cell):
        """Solve a cell is repeatedly matched by multiple cells"""
        child_node = CellNode(duplicate_match_cell)
        tmp = self.get_current_tree(duplicate_match_cell)
        parent0 = tmp[0].parent(child_node.nid)
        parent1 = tmp[1].parent(child_node.nid)
        tree_dict = {parent0: tmp[0], parent1: tmp[1]}
        sm0 = self.matcher.match_similar(duplicate_match_cell, parent0.cell)
        sm1 = self.matcher.match_similar(duplicate_match_cell, parent1.cell)
        match_score = {parent0: sm0['distance'],
                       parent1: sm1['distance']}
        error_parent = min(match_score)
        tree_dict[error_parent].remove_node(child_node.nid)
        return {error_parent: tree_dict[error_parent]}

    def rematch(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """For the loss detection cells, re-match the upper and lower frames with the cache frame"""
        unmatched_list = [cell for cell in fe1.cells if cell.is_be_matched is False]
        if not unmatched_list:
            return
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

    def check_track(self, fe1: FeatureExtractor, fe2: FeatureExtractor, fe3: FeatureExtractor):
        """
        Check the track results to see if there are any wrong matches and omissions,
        and update the matching status at the same time
        """
        cells = sorted(fe2.cells, key=lambda cell: cell.sort_value, reverse=True)
        self.rematch(fe1, fe2)
        for cell in cells:
            tmp = self.get_current_tree(cell)
            if len(tmp) > 1:
                self.handle_duplicate_match(duplicate_match_cell=cell)

    def track(self, range=None, speed_filename=None):
        """Read image frames sequentially and start tracking"""
        global writer, speed_f
        index = 0
        if speed_filename:
            speed_f = open(speed_filename, 'w', newline='')
            writer = csv.writer(speed_f)
            writer.writerow(['Iteration', 'Speed (it/s)'])
        for fe_before, fe_current, fe_next in tqdm(self.feature_ext, total=range, desc='tracking process'):
            start_time = time.time()
            self.track_near_frame(fe_before, fe_current)
            # self.track_near_frame_mult_thread(fe_before, fe_current)
            self.fe_cache.append(fe_before)
            self.check_track(fe_before, fe_current, fe_next)
            del fe_before

            end_time = time.time()
            if speed_filename:
                writer.writerow([index, f"{1 / (end_time - start_time + 1e-10):.2f}"])
            if range:
                index += 1
                if index > range:
                    break
        if speed_filename:
            speed_f.close()

        __filter_trees = [tree for tree in self.trees if len(tree.nodes) > config.FILTER_THRESHOLD]
        del self.trees
        self.trees = __filter_trees

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
        images = list(map(lambda x: cv2.imread(x, -1), bg_fname))
        images_dict = dict(zip(list(range(len(bg_fname))), images))
        for node in tree.expand_tree():
            frame = tree.nodes.get(node).cell.frame
            bbox = tree.nodes.get(node).cell.bbox
            img_bg = images_dict[frame]
            phase = tree.nodes.get(node).cell.cell_type
            images_dict[frame] = self.draw_bbox(img_bg, tree.nodes.get(node).cell, tree.track_id,
                                                tree.get_node(node).cell.branch_id, phase)
        for i in zip(bg_fname, list(images_dict.values())):
            fname = os.path.join(save_dir, os.path.basename(i[0]).replace('.tif', '.png'))
            cv2.imwrite(fname, i[1])

    def visualize_to_tif(self, background_mcy_image: str, output_tif_path, tree_list, xrange=None, single=False):
        """
        Visualize the tracking results, you can choose to save as a single tif file or multiple tif sequences
        :param background_mcy_image: background image for visualization
        :param output_tif_path: output file path
        :param tree_list: TrackingTree list to visualize
        :param xrange: visualization range
        :param single: Whether to choose to save as a single file
        """

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
                                                        phase=i.get_node(node).cell.cell_type)
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
    """
    Obtain a complete cell sequence from the track tree, including cell images, dic and mcy dual channels, and cycles,
    and the generated file name is named track_id-branch_id-frame-cell_type.tif"""
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
            cell.cell_type[0]) + '.tif'
        tifffile.imwrite(os.path.join(save_mcy, fname), convert_dtype(mcy_img))
        tifffile.imwrite(os.path.join(save_dic, fname), convert_dtype(dic_img))



