import logging
import os.path
from collections import Counter
from copy import deepcopy
from typing import List
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from imagesize import imagesize

from SCTrack.base import Cell
from SCTrack.tracker import Tracker, CellNode, TrackingTree
from SCTrack.config import RAW_INPUT_IMAGE_SIZE
from SCTrack.config import N_CLASS, CLASS_NAME
from SCTrack import config


class TreeParser(object):
    """parse TrackingTree"""

    def __init__(self, track_tree: TrackingTree):
        self.tree = track_tree

        self.lineage_dict = {}
        # {cells: [children_list], branch_id: ''， 'G1_start': '', 'S_start': '', 'G2_start': '', 'M1_start': '', 'M2_start': ''}

        self.root_parent_list = [self.tree.nodes.get(self.tree.root)]
        self.parse_root_flag = False
        self.parse_mitosis_flag = {}
        self.parse_s_flag = {}
        self.smooth_flag = {}

        self.__division_count = 0

    def record_cell_division_count(self):
        self.__division_count += 1
        return self.__division_count

    def get_child(self, parent: CellNode) -> List[CellNode]:
        """Return all child nodes according to the parent node, including only direct child nodes"""
        return self.tree.children(parent.identifier)

    def search_root_node(self):
        """
        Find the root node of each generation of cells, that is, the first node of each branch of TrackingTree
        This method is to deal with the situation where mitosis occurs, that is, the tree begins to branch
        When the cell begins to undergo mitosis, the tree will split, record the node at this time, and backtrack
        as a generation of cells
        """
        branch_id = 0
        root_node = self.tree.nodes.get(self.tree.root)
        root_node.cell.set_cell_id(str(self.tree.track_id) + '_' + str(branch_id))
        self.lineage_dict[root_node] = {'cells': [root_node], 'branch_id': root_node.cell.branch_id, 'parent': None}
        if self.tree.root is None:
            return
        else:
            loop_queue = [root_node]
            while loop_queue:
                current = loop_queue.pop(0)
                ch = self.get_child(current)
                for next_node in ch:
                    loop_queue.append(next_node)
                if len(ch) > 1:
                    self.record_cell_division_count()
                    # if current not in self.lineage_dict or current in self.lineage_dict:  # not useful branch
                    for child_cell in ch:
                        self.lineage_dict[child_cell] = {'cells': [child_cell],
                                                         'branch_id': child_cell.cell.branch_id,
                                                         'branch_start': child_cell.cell.frame, 'parent': current}
                        if child_cell not in self.root_parent_list:
                            branch_id += 1
                            child_cell.cell.set_cell_id(str(self.tree.track_id) + '_' + str(branch_id))
                            self.root_parent_list.append(child_cell)
        self.parse_root_flag = True

    def parse_lineage(self, root_node):
        """Get the cell node sequence contained in each branch"""
        loop_queue = [root_node]
        last_node = None
        while loop_queue:
            current_node = loop_queue.pop(0)
            last_node = current_node
            ch = self.get_child(current_node)
            for loop_node in ch:
                loop_queue.append(loop_node)
            if current_node not in self.lineage_dict[root_node].get('cells'):
                self.lineage_dict[root_node].get('cells').append(current_node)
            if len(ch) > 1:
                break
        self.lineage_dict[root_node]['branch_end'] = last_node.cell.frame + 1

    def get_lineage_dict(self):
        if not self.parse_root_flag:
            self.search_root_node()
        for root in self.root_parent_list:
            self.parse_lineage(root)
            self.parse_mitosis_flag[root] = False
        return self.lineage_dict

    def bfs(self):
        """Traverse TrackingTree according to breadth first"""
        root_node = self.tree.nodes.get(self.tree.root)
        if self.tree.root is None:
            return
        else:
            loop_queue = [root_node]
            while loop_queue:
                current = loop_queue.pop(0)
                ch = self.get_child(current)
                for loop_node in ch:
                    loop_queue.append(loop_node)
                yield ch

    @staticmethod
    def check_mitosis_start(start_index, lineage_cell, area_size_t=0.9, mitosis_gap=20, m_predict_threshold=5):
        """
        Check the entry of the M period, if the area meets the conditions, check the next frame, if the area of the next
        frame is too small, it is not considered to have entered, and it is judged as a segmentation misjudgment.
        If the check is passed, check the next 6 frames. If the number of M phases is predicted to be greater than or
        equal to the threshold, it is judged to enter the M cell_type to pass, otherwise, the judgment fails.
        If the interval from the last entry to M is too short, it is also considered a misjudgment
        """
        predict_enter_cell = lineage_cell[start_index]
        next_cell = lineage_cell[start_index + 1]
        if lineage_cell[0].cell.cell_type == 'M' and start_index < mitosis_gap:
            return False
        elif next_cell.cell.area / predict_enter_cell.cell.area < area_size_t:
            return False
        else:
            predict_m_count = 0
            if len(lineage_cell) - start_index < 5:
                return True
            for i in range(min(8, len(lineage_cell) - start_index)):
                if lineage_cell[start_index + i].cell.cell_type == 'M':
                    predict_m_count += 1
            if predict_m_count >= m_predict_threshold:
                return True
            else:
                return False

    @staticmethod
    def check_ctype_start(type_name, start_index, linage_cell, threshold=config.PROB_THRESHOLD * config.SMOOTH_WINDOW_LEN):
        """
        Check the entry of cell_type, if the cell is predicted to be in indicate cell_type, check 10 frames later,
        If the number of remaining frames is less than 10 frames, check all the remaining frames.
        If the cumulative predicted indicate cell type is greater than the threshold,
        If the judgment succeeds, otherwise, the judgment fails
        """
        count = 0
        if linage_cell[start_index].cell.cell_type == type_name:
            for i in range(min(config.SMOOTH_WINDOW_LEN, len(linage_cell) - start_index)):
                if linage_cell[start_index + i].cell.cell_type == type_name:
                    count += 1
        if count >= threshold:
            return True
        return False

    @staticmethod
    def check_ctype_exit(type_name, end_index, linage_cell, threshold=config.PROB_THRESHOLD * config.SMOOTH_WINDOW_LEN):
        """
        To judge the exit of cell_type, the judgment principle is the same as entering  cell_type,
        if the cells start to exit cell_type, check later
        """
        non_type_count = 0
        if linage_cell[end_index].cell.cell_type != type_name:
            for i in range(min(config.SMOOTH_WINDOW_LEN, len(linage_cell) - end_index)):
                if linage_cell[end_index + i].cell.cell_type != type_name:
                    non_type_count += 1
            if non_type_count >= threshold:
                return True
            return False

    def parse_mitosis(self, lineage: dict, root: CellNode, lineage_index=None):
        """
        Parse entry and exit of mitosis
        """
        area_size_t = 1.4
        cell_node_line = lineage.get('cells')
        mitosis_start_index = None
        exist_m_frame = 0
        for i in range(len(cell_node_line) - 1):
            before_cell_node = cell_node_line[i]
            current_cell_node = cell_node_line[i + 1]
            # print(f'{current_cell_node.cell.area / before_cell_node.cell.area:.2f}')
            if before_cell_node.cell.cell_type == 'M':
                exist_m_frame += 1
            if current_cell_node.cell.area / before_cell_node.cell.area >= area_size_t:
                if self.check_mitosis_start(i, cell_node_line, m_predict_threshold=config.PROB_THRESHOLD * config.SMOOTH_WINDOW_LEN):
                    mitosis_start_index = i + 1
                    break
            elif before_cell_node.cell.cell_type == 'M' and current_cell_node.cell.area / before_cell_node.cell.area < area_size_t:
                if self.check_mitosis_start(i, cell_node_line, m_predict_threshold=config.PROB_THRESHOLD * config.SMOOTH_WINDOW_LEN):
                    mitosis_start_index = i
                    break
        if mitosis_start_index is None:
            if len(cell_node_line) < 5:
                if exist_m_frame >= 3:
                    for cell_node in cell_node_line:
                        cell_node.cell.cell_type = 'M'
                    lineage['m2_start'] = 0
            else:
                if lineage_index != 0:
                    for cell_node in cell_node_line[: 3]:
                        cell_node.cell.cell_type = 'M'
                    lineage['m1_start'] = 0
        else:
            for m_index in range(mitosis_start_index, len(cell_node_line)):
                cell_node_line[m_index].cell.cell_type = 'M'
            lineage['m2_start'] = mitosis_start_index
        self.parse_mitosis_flag[root] = True

    def parse_s(self, lineage: dict, root: CellNode, lineage_index=None):
        """To judge the entry of S cell_type"""
        cell_node_line = lineage.get('cells')
        s_start_index = None
        s_exit_index = None
        if not self.parse_mitosis_flag[root]:
            self.parse_mitosis(lineage, root, lineage_index=lineage_index)
        if lineage.get('m1_start') is not None:
            check_start = lineage.get('m1_start')
        else:
            check_start = 0
        for cell_node_index in range(check_start, len(cell_node_line)):
            if cell_node_line[cell_node_index].cell.cell_type == 'S':
                if self.check_ctype_start('S', cell_node_index, cell_node_line):
                    lineage['s_start'] = s_start_index = cell_node_index
                    break
        if s_start_index is not None:
            for cell_node_index_2 in range(s_start_index, len(cell_node_line)):
                if cell_node_line[cell_node_index_2].cell.cell_type != 'S':
                    if self.check_ctype_exit('S', cell_node_index_2, cell_node_line):
                        lineage['s_exit'] = s_exit_index = cell_node_index_2
                        break

        if s_start_index is not None:
            if s_exit_index is not None:
                end = s_exit_index
            else:
                end = len(cell_node_line)
            for cell_node_index_s in range(s_start_index, end):
                cell_node_line[cell_node_index_s].cell.cell_type = 'S'
        self.parse_s_flag[root] = True

    def parse_g1_g2(self, lineage: dict, root: CellNode, lineage_index=None):
        """Accurately distinguish G1/G2 into G1, G2"""
        cell_node_line = lineage.get('cells')
        g1_start_index = None
        g1_exit_index = None
        g2_start_index = None
        g2_exit_index = None
        m1_start = lineage.get('m1_start')
        m2_start = lineage.get('m2_start')
        if not self.parse_s_flag[root]:
            self.parse_s(lineage, root)
        if lineage.get('s_start') is not None:
            # 1. track starts from  S
            # 2. track starts from G1 cell_type
            # 3. The track starts from the M1
            g1_exit_index = lineage.get('s_start')
            if m1_start is not None:
                g1_start_index = 3
            else:
                g1_start_index = 0
            if lineage.get('s_exit') is not None:
                g2_start_index = lineage.get('s_exit')
                if m2_start is not None:
                    g2_exit_index = m2_start
                else:
                    g2_exit_index = len(cell_node_line)

        else:
            # cells are not in S cell_type
            # 1. The track starts from the M1 period and does not enter the S cell_type
            # 2. The track starts from the G1 cell_type and does not enter the S cell_type
            # 3. The track starts from the G2 period and enters the M2 cell_type
            # 3. The track starts from the G2 period and does not enter the M2 cell_type
            if m2_start is not None:
                if len(cell_node_line) > 5:
                    g2_start_index = 0
                    g2_exit_index = m2_start
            elif m1_start is not None:
                g1_start_index = 3
                g1_exit_index = len(cell_node_line)
            else:
                if len(cell_node_line) > 10:
                    if lineage_index == 0:
                        g1_start_index = 0
                        g1_exit_index = len(cell_node_line)
                    else:
                        g2_start_index = 0
                        g2_exit_index = len(cell_node_line)
                else:
                    g1_start_index = 0
                    g1_exit_index = len(cell_node_line)

        if g1_start_index is not None:
            lineage['g1_start'] = g1_start_index
            if g1_exit_index is not None:
                end = g1_exit_index
            else:
                end = len(cell_node_line)
            for cell_node_index_g1 in range(g1_start_index, end):
                cell_node_line[cell_node_index_g1].cell.cell_type = 'G1'
        if g2_start_index is not None:
            lineage['g2_start'] = g2_start_index
            if g2_exit_index is not None:
                end_2 = g2_exit_index
            else:
                end_2 = len(cell_node_line)
            for cell_node_index_g2 in range(g2_start_index, end_2):
                cell_node_line[cell_node_index_g2].cell.cell_type = 'G2'

    def parse_mitosis_error(self, lineage: dict, root: CellNode, lineage_index=None):
        """
        If the two daughter cells after division are not matched, all subsequent cells will be in the M cell_type.
        At this time, it should be corrected to G1 within a certain period of time.
        """
        cell_node_line = lineage.get('cells')
        m1_start = lineage.get('m1_start')
        m2_start = lineage.get('m2_start')
        m_count = 30
        if m1_start is not None:
            m_start = m1_start
        elif m2_start is not None:
            m_start = m2_start
        else:
            return
        for index in range(m_start, len(cell_node_line)):
            if cell_node_line[index].cell.cell_type == 'M':
                m_count -= 1
                if m_count < 0:
                    cell_node_line[index].cell.cell_type = 'G1'

    def set_cell_id(self, lineage: dict, root: CellNode, lineage_index):
        cell_node_line = lineage.get('cells')
        branch_id = lineage_index
        cell_id = str(self.tree.track_id) + '_' + str(branch_id)
        if lineage_index == 0:
            root.cell.set_cell_id(cell_id)
            root.cell.set_branch_id(branch_id)
            lineage['parent'] = root
        else:
            parent = lineage['parent']
            parent.cell.set_cell_id(parent.cell.cell_id)
            # parent.cell.set_cell_id(cell_id)
            # parent.cell.set_branch_id(branch_id)
        for cell_node in cell_node_line:
            cell_node.cell.set_cell_id(cell_id)
            cell_node.cell.set_branch_id(branch_id)
            cell_node.cell.set_track_id(self.tree.track_id, 1)

    def parse_lineage_phase(self, lineage: dict, root: CellNode, linage_index):
        root = root
        self.parse_mitosis(lineage, root, linage_index)
        self.parse_s(lineage, root, linage_index)
        self.parse_g1_g2(lineage, root, linage_index)
        self.parse_mitosis_error(lineage, root, linage_index)
        self.set_cell_id(lineage, root, linage_index)

    def parse_lineage_branch_id(self, lineage, branch_id):
        pass

    def smooth_type(self, cell_lineage, root, lineage_index):
        if N_CLASS and CLASS_NAME:
            if len(CLASS_NAME) != N_CLASS:
                logging.error('The number of category names and category numbers is not equal, '
                              'try to change N_CLASS OR CLASS_NAME')
                return
        if CLASS_NAME:
            if CLASS_NAME == ['G1', 'S', 'G2', 'M']:
                self.parse_lineage_phase(cell_lineage, root, lineage_index)
                return
            else:
                class_name = CLASS_NAME
        elif N_CLASS:
            class_name = set()
            for cell in cell_lineage.get('cells'):
                class_name.add(cell.cell.cell_type)
        else:
            logging.info('N_CLASS and CLASS_NAME not provided, ignore the whole process')
            return
        resolved_index_map = {}
        for ctype in class_name:
            cell_node_line = cell_lineage.get('cells')
            check_start = 0
            start_index = None
            exit_index = None
            for cell_node_index in range(check_start, len(cell_node_line)):
                if cell_node_line[cell_node_index].cell.cell_type == ctype:
                    if self.check_ctype_start(ctype, cell_node_index, cell_node_line):
                        cell_lineage[f'{ctype}_start'] = start_index = cell_node_index
                        break
            if start_index is not None:
                for cell_node_index_2 in range(start_index, len(cell_node_line)):
                    if cell_node_line[cell_node_index_2].cell.cell_type != ctype:
                        if self.check_ctype_exit(ctype, cell_node_index_2, cell_node_line):
                            cell_lineage[f'{ctype}_exit'] = exit_index = cell_node_index_2 + 1
                            break
            if start_index is not None:
                if exit_index is not None:
                    end = exit_index
                else:
                    end = len(cell_node_line)
                resolved_index_map[ctype] = start_index
                for cell_node_index in range(start_index, end):
                    cell_node_line[cell_node_index].cell.cell_type = ctype
        if 0 not in resolved_index_map.values():
            if not resolved_index_map:
                type_count = []
                for cell in cell_lineage.get('cells'):
                    type_count.append(cell.cell.cell_type)
                counter = Counter(type_count)
                resolved_type = counter.most_common(1)[0][0]
                for cell in cell_lineage.get('cells'):
                    cell.cell.cell_type = resolved_type
            else:
                resolved_type = min(resolved_index_map, key=resolved_index_map.get)
                for cell in cell_lineage.get('cells')[:min(resolved_index_map.values())]:
                    cell.cell.cell_type = resolved_type
        self.smooth_flag[root] = True


def pares_single_tree(tree: TrackingTree):
    parser = TreeParser(tree)
    parser.search_root_node()
    parser.get_lineage_dict()
    for node_index in range(len(parser.root_parent_list)):
        cell_lineage = parser.lineage_dict.get(parser.root_parent_list[node_index])
        if tree.get_node(tree.root).cell.cell_type is None:
            parser.set_cell_id(cell_lineage, root=parser.root_parent_list[node_index], lineage_index=node_index)
        else:
            # parser.parse_lineage_phase(cell_lineage, root=parser.root_parent_list[node_index], linage_index=node_index)
            parser.smooth_type(cell_lineage, root=parser.root_parent_list[node_index], lineage_index=node_index)
            parser.set_cell_id(cell_lineage, root=parser.root_parent_list[node_index], lineage_index=node_index)
            # print(cell_lineage)
    return parser


def run_track(annotation, track_range=None, dic=None, mcy=None, speed_filename=None):
    tracker = Tracker(annotation, dic=dic, mcy=mcy)
    if track_range:
        tracker.track(range=track_range, speed_filename=speed_filename)
    else:
        tracker.track()
    parser_dict = {}
    for tree in tracker.trees:
        parser = pares_single_tree(tree)
        parser_dict[tree] = parser
    tracker.parser_dict = parser_dict
    return tracker


def track_tree_to_table(tracker: Tracker, filepath):
    """Export track result to table"""
    track_detail_columns = ['frame_index', 'track_id', 'cell_id', 'parent_id', 'center_x', 'center_y', 'cell_type',
                            'mask_of_x_points', 'mask_of_y_points']
    track_detail_dataframe = pd.DataFrame(columns=track_detail_columns)

    def generate_series(cell_lineage):
        cell_nodes = cell_lineage.get('cells')
        parent = cell_lineage.get('parent')
        series_list = []
        new_nodes = []
        before_index = 0
        current_index = 1
        if len(cell_nodes) == 1:
            new_nodes.append(cell_nodes[0])
        for cell_node_index in range(len(cell_nodes) - 1):
            before_node = cell_nodes[before_index]
            current_node = cell_nodes[current_index]
            new_nodes.append(before_node)
            if current_node.cell.frame - before_node.cell.frame != 1:
                for frame in range(before_node.cell.frame + 1, current_node.cell.frame):
                    # gap_cell = deepcopy(before_node.cell)
                    # gap_cell.frame = frame
                    gap_cell = Cell(position=before_node.cell.position, cell_type=before_node.cell.cell_type,
                                    frame_index=frame)
                    gap_cell.set_track_id(before_node.cell.track_id, 1)
                    gap_cell.set_cell_id(before_node.cell.cell_id)
                    gap_cell.set_branch_id(before_node.cell.branch_id)
                    gap_node = CellNode(gap_cell)
                    new_nodes.append(gap_node)
            elif current_index == len(cell_nodes) - 1:
                new_nodes.append(current_node)
            else:
                pass
            before_index += 1
            current_index += 1

        # for node in cell_nodes:
        # print(new_nodes)
        for node in new_nodes:
            col = [node.cell.frame, node.cell.track_id,
                   node.cell.cell_id, parent.cell.cell_id,
                   node.cell.center[0], node.cell.center[1],
                   node.cell.cell_type,
                   node.cell.position[0], node.cell.position[1]]
            s = pd.Series(dict(zip(track_detail_columns, col)))
            series_list.append(s)
        return series_list, new_nodes

    parser_dict = tracker.parser_dict
    for tree in parser_dict:
        if tree.track_id == 4:
            p = parser_dict[tree]
        parser = parser_dict[tree]
        for node_index in parser.root_parent_list:
            cell_lineage = parser.lineage_dict.get(node_index)
            series_list, new_node_list = generate_series(cell_lineage)
            for series in series_list:
                track_detail_dataframe = track_detail_dataframe._append(series, ignore_index=True)
    fname = filepath
    track_detail_dataframe.to_csv(fname, index=False)


def track_trees_to_json(tracker: Tracker, output_fname, xrange, basename=None):
    """Export track result to json file"""
    if basename is None:
        prefix = 'mcy'
    else:
        prefix = basename

    def update_region(node):
        # print(type(node))
        region = node.cell.region
        phase = node.cell.cell_type
        track_id = node.cell.track_id
        cell_id = node.cell.cell_id
        region['region_attributes']['cell_type'] = phase
        region['region_attributes']['track_id'] = track_id
        region['region_attributes']['cell_id'] = cell_id
        region['region_attributes']['id'] = track_id
        node.cell.set_region(region)

    result = {}
    frame_map = {}

    for frame in range(xrange):
        image_name = prefix + '-' + str(frame).zfill(4) + '.png'
        tmp = {
            image_name: {
                "filename": image_name,
                "size": int(RAW_INPUT_IMAGE_SIZE[0] * RAW_INPUT_IMAGE_SIZE[1]),
                "regions": [],
                "file_attributes": {}
            }
        }
        result[image_name] = deepcopy(tmp[image_name])
        frame_map[frame] = image_name
        del tmp
    for tree in tracker.trees:
        for node in tree.all_nodes():
            frame = node.cell.frame
            update_region(node)
            region = node.cell.region
            image_name = frame_map.get(frame)
            if image_name is not None:
                tmp_frame = result.get(image_name)
                tmp_frame['regions'].append(region)
    with open(output_fname, 'w') as f:
        json.dump(result, f)


def track_tree_to_TRA(tracker: Tracker, output_fname=None, xrange=None, basename=None):
    """Export track result as TRA format"""

    TRA = []
    parser_dict = tracker.parser_dict
    for tree in parser_dict:
        parser = parser_dict[tree]
        for node_index in parser.root_parent_list:
            cell_lineage = parser.lineage_dict.get(node_index)
            cells = cell_lineage.get('cells')
            parent = cell_lineage['parent']
            branch_id = cell_lineage['branch_id']
            L = int(cells[0].cell.cell_id.replace('_', ''))
            B = cells[0].cell.frame
            E = cells[-1].cell.frame
            if branch_id == 0:
                P = 0
            else:
                P = int(parent.cell.cell_id.replace('_', ''))
            line = f'{L} {B} {E} {P}\n'
            TRA.append(line)
    if output_fname:
        with open(output_fname, 'w') as f:
            f.writelines(TRA)


def track_tree_to_mask(tracker, width, height, output_dir):
    parser_dict = tracker.parser_dict
    cell_each_frame = {}  # {frame: cell_list}
    for tree in parser_dict:
        parser = parser_dict[tree]
        for node_index in parser.root_parent_list:
            cell_lineage = parser.lineage_dict.get(node_index)
            cells = cell_lineage.get('cells')
            for cell in cells:
                if cell.cell.frame not in cell_each_frame:
                    cell_each_frame[cell.cell.frame] = {cell}
                else:
                    cell_each_frame[cell.cell.frame].add(cell)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for frame in cell_each_frame.keys():
        fname = os.path.join(output_dir, f"mask" + f"{frame}".zfill(3) + ".tif")
        mask_arr = np.zeros((height, width), dtype=np.uint16)
        for cell in cell_each_frame[frame]:
            contours = cell.cell.contours
            L = int(cell.cell.cell_id.replace('_', ''))
            cv2.fillConvexPoly(mask_arr, contours, (L))
        tifffile.imwrite(fname, mask_arr)



def run(annotation, output_dir, basename, track_range=None, save_visualize=True, visualize_background_image=None,
        dic=None, mcy=None, track_to_json=True):
    if track_range is None:
        if type(annotation) is str:
            with open(annotation, encoding='utf-8') as f:
                data = json.load(f)
        elif type(annotation) is dict:
            data = annotation
        else:
            raise ValueError(f"annotation type error {type(annotation)}")
        xrange = len(data)
    else:
        xrange = track_range + 2
    if config.RECORD_SPEED:
        speed_output_filename = os.path.join(output_dir, 'track_speed.csv')
    else:
        speed_output_filename = None
    tracker = run_track(annotation, track_range=xrange - 2, dic=dic, mcy=mcy, speed_filename=speed_output_filename)
    track_table_fname = os.path.join(output_dir, 'track.csv')
    track_visualization_fname = os.path.join(output_dir, 'track_visualization.tif')
    track_json_fname = os.path.join(output_dir, 'result_with_track.json')
    tracktree_save_path = os.path.join(output_dir, 'TrackTree')
    track_tree_to_table(tracker, track_table_fname)
    tracker.track_tree_to_json(tracktree_save_path)
    track_tree_to_TRA(tracker, os.path.join(output_dir, 'TRA.txt'))
    image_width, image_height = imagesize.get(mcy)
    track_tree_to_mask(tracker, image_width, image_height, os.path.join(output_dir, 'mask'))
    if track_to_json:
        track_trees_to_json(tracker, track_json_fname, xrange=xrange, basename=basename)
    if save_visualize:
        tracker.visualize_to_tif(visualize_background_image, track_visualization_fname, tracker.trees, xrange=xrange)


if __name__ == '__main__':
    # annotation = r'G:\杂项\example\example-annotation.json'
    # mcy_img = r'G:\杂项\example\example-image.tif'
    # dic_img = r'G:\杂项\example\example-bf.tif'
    annotation = r"G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\test.json"
    mcy_img = r"G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\SEG.tif"
    dic_img = r"G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST\SEG.tif"
    # tracker = run_track(r'G:\paper\evaluate_data\copy_of_1_xy10\result-GT.json', track_range=10)
    # tracker = run_track(annotation, track_range=30)
    # background_filename_list = [os.path.join(r'G:\paper\evaluate_data\copy_of_1_xy10\tif-seq', i) for i in
    #                             os.listdir(r'G:\paper\evaluate_data\copy_of_1_xy10\tif-seq')]
    # print(background_filename_list)
    # for i in tracker.trees:
    #     print(i)
    # tracker.visualize_single_tree(tree=tracker.trees[51],
    #                               save_dir=r'G:\paper\evaluate_data\copy_of_1_xy10\single_tree_visualize',
    #                               background_filename_list=background_filename_list, xrange=10)

    run(annotation=annotation, output_dir=r"G:\CTC dataset\Fluo-N2DL-HeLa\Fluo-N2DL-HeLa\01_ST",
        track_range=None, dic=None, mcy=mcy_img,
        save_visualize=True, visualize_background_image=mcy_img,
        track_to_json=True, basename=r'man_seg')
