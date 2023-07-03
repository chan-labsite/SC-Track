#  第一步：进行目标检测，利用生成的json文件，可以很方便的给出每一帧每个细胞的位置


# 第二步：将所有目标框中对应的目标抠出来，进行特征提取


# 第三步：进行相似度计算，计算前后两帧目标之间的匹配程度


# 第四步：数据关联，为每个对象分配目标的 ID


# Done  实现json向bounding box的转化 finished

# Done  实现特征提取 finished

# Done  实现前后帧特征匹配 finished

# Done 处理遮挡后匹配


from __future__ import annotations

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

# import base
# import feature
# import tracker
# import check_phase
# import t_error
# import prepare
# import track

# frame1 = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0000.tif'
# frame2 = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0001.tif'

# image1 = cv2.imread(frame1, -1)
# image2 = cv2.imread(frame2, -1)


# from base import Cell




# x =np.arange(-2* np.pi, 2*np.pi, 0.01)
# y = np.cos(x)

# from base import Vector

# a = Vector(1, 1)

# b = Vector(2, 2)

# c = Vector(1, 0)

# print(a.cosDistance(b))
# print(a.cosDistance(c))
# print(a.cosSimilar(b))
# print(a.cosSimilar(c)

# from base import Rectangle

# a = Rectangle(20, 50, 10, 40)

# b = Rectangle(30, 70, 20, 60)
# b = Rectangle(60, 80, 20, 60)
# b = Rectangle(10, 60, 20, 30)
# b = Rectangle(30, 40, 20, 30)

# bg = np.zeros(shape=(100, 100))
# a.draw(background=bg)
#
# print(a.isIntersect(b))
# print(b.isIntersect(a))
# print(b.isIntersect(b))
#
# print(b.isInclude(a))
# print(a.isInclude(b))

# b.draw(background=bg,isShow=True)

# c = Rectangle(-10, 20, -20, 40)
# d = Rectangle(5, 15, 10, 20)
# d.draw(bg)
# c.draw(bg, isShow=True)


# from base import Cell
#
# bg = np.zeros(shape=(1024, 1024))
# Cell1 = Cell(position=([545, 640], [576, 670]))
# Cell2 = Cell(position=([565, 660], [596, 690]))
# Cell3 = Cell(position=([800, 860], [900, 990]))
#
#
# c1 = Rectangle(Cell1.bbox[0], Cell1.bbox[1], Cell1.bbox[2], Cell1.bbox[3])
# c2 = Rectangle(*Cell2.bbox)
# c3 = Rectangle(*Cell3.bbox)
#
# print(Cell1 in Cell2)
# print(Cell2 in Cell1)
#
# print(Cell2 in Cell2)
#
# print(Cell1 in Cell3)
# print(Cell3 in Cell1)
#
# c1.draw(bg)
# Cell1.available_range.draw(bg)
# c2.draw(bg)
# Cell2.available_range.draw(bg)
# c3.draw(bg)
# Cell3.available_range.draw(bg, isShow=True)

# s = [rf" python .\main.py -p F:\wangjiaqi\src\s{i}\mcy.tif -bf F:\wangjiaqi\src\s{i}\dic.tif -o F:\wangjiaqi\src\s{i}\ret.json -t" for i in range(1, 12)]
# for i in s:
#     print(i)


# def loop(matched_cells_dict: dict):
#     cell_dict_keys = list(matched_cells_dict.keys())
#     length = len(cell_dict_keys)
#     match_result = {}
#     for i in range(length - 1):
#         cell_1 = matched_cells_dict[cell_dict_keys.pop(0)]
#         for j in range(len(cell_dict_keys)):
#             cell_2 = matched_cells_dict[cell_dict_keys[j]]
#             print((cell_1, cell_2))


# class CellStatus(object):
#     """记录细胞当前状态，包括周期情况，分裂情况，匹配情况
#     周期情况：是否进入了有丝分裂期， 哪一帧进入的有丝分裂
#     分裂情况：有无发生有丝分裂事件
#     匹配情况：此细胞有无丢失匹配
#     """
#
#     __status_types = ['enter_mitosis', 'enter_mitosis_frame', 'division_event_happen',
#                       'division_count', 'exit_mitosis', 'exit_mitosis_frame']
#
#     def __init__(self):
#         self.__enter_mitosis: bool = False
#         self.__enter_mitosis_frame: int | None = None
#         self.__division_event_happen: bool = False       # 此值记录表示细胞至少发生了一次有丝分裂
#         self.__division_count: int = 0
#         self.__exit_mitosis: bool = False
#         self.__exit_mitosis_frame: int | None = None
#
#     @property
#     def status(self):
#         return dict(zip(CellStatus.__status_types,
#                         (self.__enter_mitosis, self.__enter_mitosis_frame, self.__division_event_happen,
#                          self.__division_count,self.__exit_mitosis, self.__exit_mitosis_frame)))
#
#     def get_status(self, status_type):
#         return self.status.get(status_type)
#
#     def enter_mitosis(self, frame):
#         self.__enter_mitosis = True
#         self.__exit_mitosis = False
#         self.__enter_mitosis_frame = frame
#
#     def exit_mitosis(self, frame):
#         self.__enter_mitosis = False
#         self.__exit_mitosis = True
#         self.__exit_mitosis_frame = frame
#         self.__division_event_happen = True
#         self.__division_count += 1
#
# class CellNode(Node):
#     """
#     追踪节点，包含细胞的tracking ID，以及细胞自身的详细信息，和父子节点关系
#     """
#     _instance_ = {}
#     STATUS = ['ACCURATE', 'ACCURATE-FL', 'INACCURATE', 'INACCURATE-MATCH', 'PREDICTED']
#
#     def __new__(cls, *args, **kwargs):
#         key = str(args[0]) + str(kwargs)
#         if key not in cls._instance_:
#             cls._instance_[key] = super().__new__(cls)
#             cls._instance_[key].status = None
#             cls._instance_[key].track_id = None
#             cls._instance_[key].parent = None
#             cls._instance_[key].childs = []
#             cls._instance_[key].add_tree = False  # 如果被添加到TrackingTree中，设置为True
#             cls._instance_[key].life = 5  # 每个分支初始生命值为5，如果匹配成功则+1，如果没有匹配上，或者利用缺省值填充匹配，则-1，如如果生命值为0，则该分支不再参与匹配
#             cls._instance_[key]._init_flag = False
#         return cls._instance_[key]
#
#     def __init__(self, cell: Cell, branch_id=None, node_type='cell', fill_gap_index=None):
#         if not self._init_flag:
#             self.cell = cell
#             self.branch_id = branch_id
#             if node_type == 'gap':
#                 assert fill_gap_index is not None
#             super().__init__((cell, self.branch_id))
#             self._init_flag = True
#
#
#     @property
#     def identifier(self):
#         return str(id(self.cell))
#
#     def _set_identifier(self, nid):
#         if nid is None:
#             self._identifier = str(id(self.cell))
#         else:
#             self._identifier = nid
#
#     def get_status(self):
#         return self.status
#
#     def set_parent(self, parent: CellNode):
#         self.parent = parent
#
#     def get_parent(self):
#         return self.parent
#
#     def set_childs(self, child: CellNode):
#         self.childs.append(child)
#
#     def get_childs(self):
#         return self.childs
#
#     def set_tree_status(self, status: CellStatus):
#         self.tree_status = status
#         self.add_tree = True
#
#     def get_tree_status(self):
#         if self.add_tree:
#             return self.tree_status
#         return None
#
#     def set_status(self, status):
#         if status in self.STATUS:
#             self.status = status
#         else:
#             raise ValueError(f"set error status: {status}")
#
#     def get_branch_id(self):
#         if self.branch_id is None:
#             raise ValueError("Don't set the branch_id")
#         else:
#             return self.branch_id
#
#     def set_branch_id(self, branch_id):
#         self.branch_id = branch_id
#
#     def get_track_id(self):
#         if self.track_id is None:
#             raise ValueError("Don't set the track_id")
#         else:
#             return self.track_id
#
#     def set_track_id(self, track_id):
#         self.track_id = track_id
#
#     def __repr__(self):
#         if self.add_tree:
#             return f"Cell Node of {self.cell}, branch {self.branch_id}, status: {self.get_tree_status()}"
#         else:
#             return f"Cell Node of {self.cell}, branch {self.branch_id}"
#
#     def __str__(self):
#         return self.__repr__()
#
#     def __hash__(self):
#         return int(id(self))
#
#

# region = {'shape_attributes':[], 'region_attributes': {}}
# cell = base.Cell(position=[(1, 2, 3, 4), (2, 4, 6, 8)])
# cell2 = base.Cell(position=[(1, 2, 3, 5), (2, 4, 6, 10)])
#
# cell.set_branch_id(0)
#
# print(cell)
#
# cell_t = base.Cell(position=[(1, 2, 3, 4), (2, 4, 6, 8)])
# print(cell_t)
#
# print(cell_t is cell)
# #
# node = CellNode(cell)
# print(node)
# node.set_branch_id(0)
# print(node)
#
# node2 = CellNode(cell)
# print(node2)
#
# node3 = CellNode(cell2)
# print(node3)

