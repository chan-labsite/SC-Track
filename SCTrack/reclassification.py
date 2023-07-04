import logging
import os.path
import sys
from copy import deepcopy
from typing import List
import json
import random
import pickle
import pandas as pd
from SCTrack.base import Cell
from SCTrack.tracker import Tracker, CellNode, TrackingTree
from SCTrack.config import RAW_INPUT_IMAGE_SIZE


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
                    if current not in self.lineage_dict:
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
        equal to the threshold, it is judged to enter the M phase to pass, otherwise, the judgment fails.
        If the interval from the last entry to M is too short, it is also considered a misjudgment
        """
        predict_enter_cell = lineage_cell[start_index]
        next_cell = lineage_cell[start_index + 1]
        if lineage_cell[0].cell.phase == 'M' and start_index < mitosis_gap:
            return False
        elif next_cell.cell.area / predict_enter_cell.cell.area < area_size_t:
            return False
        else:
            predict_m_count = 0
            if len(lineage_cell) - start_index < 5:
                return True
            for i in range(min(8, len(lineage_cell) - start_index)):
                if lineage_cell[start_index + i].cell.phase == 'M':
                    predict_m_count += 1
            if predict_m_count >= m_predict_threshold:
                return True
            else:
                return False

    @staticmethod
    def check_s_start(start_index, linage_cell, threshold=6):
        """
        Check the entry of S phase, if the cell is predicted to be in S phase, check 10 frames later at random,
        If the number of remaining frames is less than 10 frames, check all the remaining frames.
        If the cumulative predicted S period is greater than the threshold,
        If the judgment succeeds, otherwise, the judgment fails
        """
        s_count = 0
        if linage_cell[start_index].cell.phase == 'S':
            for i in range(min(10, len(linage_cell) - start_index)):
                if linage_cell[start_index + i].cell.phase == 'S':
                    s_count += 1
        if s_count >= threshold:
            return True
        return False

    @staticmethod
    def check_s_exit(end_index, linage_cell, threshold=6):
        """
        To judge the exit of S phase, the judgment principle is the same as entering S phase,
        if the cells start to exit S phase, check later
        """
        non_s_count = 0
        if linage_cell[end_index].cell.phase != 'S':
            for i in range(min(10, len(linage_cell) - end_index)):
                if linage_cell[end_index + i].cell.phase != 'S':
                    non_s_count += 1
            if non_s_count >= threshold:
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
            if before_cell_node.cell.phase == 'M':
                exist_m_frame += 1
            if current_cell_node.cell.area / before_cell_node.cell.area >= area_size_t:
                if self.check_mitosis_start(i, cell_node_line):
                    mitosis_start_index = i + 1
                    break
            elif before_cell_node.cell.phase == 'M' and current_cell_node.cell.area / before_cell_node.cell.area < area_size_t:
                if self.check_mitosis_start(i, cell_node_line, m_predict_threshold=5):
                    mitosis_start_index = i
                    break
        if mitosis_start_index is None:
            if len(cell_node_line) < 5:
                if exist_m_frame >= 3:
                    for cell_node in cell_node_line:
                        cell_node.cell.phase = 'M'
                    lineage['m2_start'] = 0
            else:
                if lineage_index != 0:
                    for cell_node in cell_node_line[: 3]:
                        cell_node.cell.phase = 'M'
                    lineage['m1_start'] = 0
        else:
            for m_index in range(mitosis_start_index, len(cell_node_line)):
                cell_node_line[m_index].cell.phase = 'M'
            lineage['m2_start'] = mitosis_start_index
        self.parse_mitosis_flag[root] = True

    def parse_s(self, lineage: dict, root: CellNode, lineage_index=None):
        """To judge the entry of S phase"""
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
            if cell_node_line[cell_node_index].cell.phase == 'S':
                if self.check_s_start(cell_node_index, cell_node_line):
                    lineage['s_start'] = s_start_index = cell_node_index
                    break
        if s_start_index is not None:
            for cell_node_index_2 in range(s_start_index, len(cell_node_line)):
                if cell_node_line[cell_node_index_2].cell.phase != 'S':
                    if self.check_s_exit(cell_node_index_2, cell_node_line):
                        lineage['s_exit'] = s_exit_index = cell_node_index_2
                        break

        if s_start_index is not None:
            if s_exit_index is not None:
                end = s_exit_index
            else:
                end = len(cell_node_line)
            for cell_node_index_s in range(s_start_index, end):
                cell_node_line[cell_node_index_s].cell.phase = 'S'
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
            # 2. track starts from G1 phase
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
            # cells are not in S phase
            # 1. The track starts from the M1 period and does not enter the S phase
            # 2. The track starts from the G1 phase and does not enter the S phase
            # 3. The track starts from the G2 period and enters the M2 phase
            # 3. The track starts from the G2 period and does not enter the M2 phase
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
                cell_node_line[cell_node_index_g1].cell.phase = 'G1'
        if g2_start_index is not None:
            lineage['g2_start'] = g2_start_index
            if g2_exit_index is not None:
                end_2 = g2_exit_index
            else:
                end_2 = len(cell_node_line)
            for cell_node_index_g2 in range(g2_start_index, end_2):
                cell_node_line[cell_node_index_g2].cell.phase = 'G2'

    def parse_mitosis_error(self, lineage: dict, root: CellNode, lineage_index=None):
        """
        If the two daughter cells after division are not matched, all subsequent cells will be in the M phase.
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
            if cell_node_line[index].cell.phase == 'M':
                m_count -= 1
                if m_count < 0:
                    cell_node_line[index].cell.phase = 'G1'

    def set_cell_id(self, lineage: dict, root: CellNode, linage_index):
        cell_node_line = lineage.get('cells')
        branch_id = linage_index
        cell_id = str(self.tree.track_id) + '_' + str(branch_id)
        if linage_index == 0:
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


def pares_single_tree(tree: TrackingTree):
    parser = TreeParser(tree)
    parser.search_root_node()
    parser.get_lineage_dict()

    for node_index in range(len(parser.root_parent_list)):
        cell_lineage = parser.lineage_dict.get(parser.root_parent_list[node_index])
        if tree.get_node(tree.root).cell.phase is None:
            parser.set_cell_id(cell_lineage, root=parser.root_parent_list[node_index], linage_index=node_index)
        else:
            parser.parse_lineage_phase(cell_lineage, root=parser.root_parent_list[node_index], linage_index=node_index)
    return parser


def run_track(annotation, track_range=None, dic=None, mcy=None, speed_filename=None):
    tracker = Tracker(annotation, dic=dic, mcy=mcy)
    if track_range:
        tracker.track(range=track_range, speed_filename=speed_filename)
    else:
        tracker.track()
    parser_dict = {}
    for tree in tracker.trees:
        try:
            parser = pares_single_tree(tree)
            parser_dict[tree] = parser
        except:
            continue
    tracker.parser_dict = parser_dict
    return tracker


def track_tree_to_table(tracker: Tracker, filepath):
    """Export track result to table"""
    track_detail_columns = ['frame_index', 'track_id', 'cell_id', 'parent_id', 'center_x', 'center_y', 'phase',
                            'mask_of_x_points', 'mask_of_y_points']
    track_detail_dataframe = pd.DataFrame(columns=track_detail_columns)

    def generate_series(cell_lineage):
        cell_nodes = cell_lineage.get('cells')
        parent = cell_lineage.get('parent')
        series_list = []
        new_nodes = []
        before_index = 0
        current_index = 1
        for cell_node_index in range(len(cell_nodes) - 1):
            before_node = cell_nodes[before_index]
            current_node = cell_nodes[current_index]
            new_nodes.append(before_node)
            if current_node.cell.frame - before_node.cell.frame != 1:
                for frame in range(before_node.cell.frame + 1, current_node.cell.frame):
                    # gap_cell = deepcopy(before_node.cell)
                    # gap_cell.frame = frame
                    gap_cell = Cell(position=before_node.cell.position, phase=before_node.cell.phase, frame_index=frame)
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
                   node.cell.phase,
                   node.cell.position[0], node.cell.position[1]]
            s = pd.Series(dict(zip(track_detail_columns, col)))
            series_list.append(s)
        return series_list, new_nodes

    parser_dict = tracker.parser_dict
    for tree in parser_dict:
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
        phase = node.cell.phase
        track_id = node.cell.track_id
        cell_id = node.cell.cell_id
        region['region_attributes']['phase'] = phase
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


def track_tree_to_MOT(tracker: Tracker, output_fname, xrange, basename=None):
    """Export track result as MOT format"""
    parser_dict = tracker.parser_dict
    for tree in parser_dict:
        parser = parser_dict[tree]
        for node_index in parser.root_parent_list:
            cell_lineage = parser.lineage_dict.get(node_index)
            pass


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
    speed_output_filename = os.path.join(output_dir, 'track_speed.csv')
    tracker = run_track(annotation, track_range=xrange - 2, dic=dic, mcy=mcy, speed_filename=speed_output_filename)
    track_table_fname = os.path.join(output_dir, 'track.csv')
    track_visualization_fname = os.path.join(output_dir, 'track_visualization.tif')
    track_json_fname = os.path.join(output_dir, 'result_with_track.json')
    tracktree_save_path = os.path.join(output_dir, 'tracktree')
    track_tree_to_table(tracker, track_table_fname)
    tracker.track_tree_to_json(tracktree_save_path)
    if track_to_json:
        track_trees_to_json(tracker, track_json_fname, xrange=xrange, basename=basename)
    if save_visualize:
        tracker.visualize_to_tif(visualize_background_image, track_visualization_fname, tracker.trees, xrange=xrange)


if __name__ == '__main__':
    annotation = r'G:\paper\evaluate_data\copy_of_1_xy10\result-GT.json'
    mcy_img = r'G:\paper\evaluate_data\copy_of_1_xy10\mcy.tif'
    dic_img = r'G:\paper\evaluate_data\copy_of_1_xy10\dic.tif'
    table = r'G:\paper\evaluate_data\copy_of_1_xy10\track-table-test.csv'
    visual = r'G:\paper\evaluate_data\copy_of_1_xy10\tracking_visualize-test.tif'
    tracker = run_track(r'G:\paper\evaluate_data\copy_of_1_xy10\result-GT.json', track_range=300)
    background_filename_list = [os.path.join(r'G:\paper\evaluate_data\copy_of_1_xy10\tif-seq', i) for i in
                                os.listdir(r'G:\paper\evaluate_data\copy_of_1_xy10\tif-seq')]
    # print(background_filename_list)
    print(tracker.trees[9])
    tracker.visualize_single_tree(tree=tracker.trees[51],
                                  save_dir=r'G:\paper\evaluate_data\copy_of_1_xy10\single_tree_visualize',
                                  background_filename_list=background_filename_list, xrange=300)

    # run(annotation=fjson, output_dir=result_save_path, track_range=track_range, dic=fbf, mcy=fpcna,
    #                 save_visualize=export_visualization, visualize_background_image=fpcna,
    #                 track_to_json=track_to_json, basename=basename)
