from __future__ import annotations

import math
import warnings
from enum import Enum
from shapely.geometry import Polygon
import numpy as np
from functools import lru_cache
from utils import convert_dtype
import config


def NoneTypeFilter(func):
    def _filter(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        cells = []
        for i in ret:
            if i.mcy.size != 0:
                cells.append(i)
        return cells

    return _filter


def warningFilter(func):
    warnings.filterwarnings("ignore")

    def _warn_filter(*args, **kwargs):
        return func(*args, **kwargs)

    return _warn_filter


class MatchStatus(Enum):
    """
    Matching status enumeration object,
    including matched, unmatched, and missing matches,
    is the status value of TrackingTree.
    """

    Matched = 0
    Unmatched = 1
    LossMatch = 2


class TreeStatus(object):
    """
    Record the state of the tracked cells, including cell phase status, division status, and matching status
    Phase: whether Mitosis has entered, and which frame has entered Mitosis
    Division: whether there is Mitosis
    Matching situation: Is there any loss of matching in this cell
    """

    # TreeStatus all Status properties
    __status_types = ['enter_mitosis', 'enter_mitosis_frame', 'division_event_happen',
                      'division_count', 'exit_mitosis', 'exit_mitosis_frame']

    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key].__tracking_tree = None
            cls._instances[key].__init_flag = False
            cls._instances[key].enter_mitosis_threshold = 50

            # The cell division window stage, with an initial value of 20, will exit division matching when no cell
            # division is matched or cell division is completed during this window period. After entering the split
            # window period, this value will decrease every frame forward.
            cls._instances[key].division_windows_len = 20

            # Starting from the completion of splitting and exiting mitosis, counting, no further entry into mitosis
            # is allowed within 10 frames, that is, when __exit_mitosis_time < 10, self.__ enter_ mitosis cannot be true
            cls._instances[key].__exit_mitosis_time = cls._instances[key].enter_mitosis_threshold

        return cls._instances[key]

    def __init__(self, tree: 'TrackingTree | None'):
        if not self.__init_flag:
            self.__tracking_tree = tree
            self.__enter_mitosis: bool = False
            self.__enter_mitosis_frame: int | None = None
            self.__division_event_happen: bool = False  # True indicates that the cell has at least one M    itosis
            self.__division_count: int = 0
            self.__exit_mitosis: bool = False
            self.__exit_mitosis_frame: int | None = None
            self.__match_status = MatchStatus.Unmatched

            # This value records the number of predicted M phase. If > 3, it is considered to have entered M period,
            # and at this point, enter needs to be called externally_ Mitosis()
            self.__predict_M_count = 0
            self.__init_flag = True

    @property
    def status(self):
        return dict(zip(TreeStatus.__status_types,
                        (self.__enter_mitosis, self.__enter_mitosis_frame, self.__division_event_happen,
                         self.__division_count, self.__exit_mitosis, self.__exit_mitosis_frame)))

    def get_status(self, status_type: 'str in  __status_types'):
        """
        :param status_type: __status_types member
        :return: member status
        """
        return self.status.get(status_type)

    def check_division_window(self):
        """
        Check if the cell is in the division window stage
        """
        if self.division_windows_len > 0:
            return True
        return False

    def reset_division_window(self):
        """
        reset the division_windows_len
        """
        self.division_windows_len = 20

    def sub_division_window(self):
        """
        When the cell is in the division window stage,  continuously decreases the division_windows_len
        """
        if self.division_windows_len > 0:
            self.division_windows_len -= 1

    def is_in_division_window(self):
        if self.division_windows_len > 0:
            return True
        return False

    def enter_mitosis(self, frame):
        if self.__exit_mitosis_time >= self.enter_mitosis_threshold:
            self.__enter_mitosis = True
            self.__exit_mitosis = False
            self.__exit_mitosis_time = 0
            self.__enter_mitosis_frame = frame
            self.__exit_mitosis_frame = None
            self.reset_division_window()

    def exit_mitosis(self, frame):
        self.__enter_mitosis = False
        self.__exit_mitosis = True
        self.__exit_mitosis_frame = frame
        self.__division_event_happen = True
        self.__division_count += 1

    def set_matched_status(self, value):
        if value == 'matched':
            self.__match_status = MatchStatus.Matched
        elif value == 'loss':
            self.__match_status = MatchStatus.LossMatch
        else:
            pass

    def add_M_count(self):
        self.__predict_M_count += 1

    def reset_M_count(self):
        self.__predict_M_count = 0

    @property
    def predict_M_len(self):
        return self.__predict_M_count

    def add_exist_time(self):
        if self.__exit_mitosis:
            self.__exit_mitosis_time += 1

    @property
    def is_mitosis_enter(self):
        return self.__enter_mitosis

    @property
    def exist_mitosis_time(self):
        return self.__exit_mitosis_time

    def __str__(self):
        return str(self.status)

    def __repr__(self):
        return self.__str__()


class CellStatus(TreeStatus):
    """
    Record the status of cells and exclude them from other matching candidates if they have participated in precise matching.
    If the cell has undergone Mitosis in a short time, it will not participate in Mitosis matching.
    """
    pass


class SingleInstance(object):
    """Singleton pattern base class. If the parameters are the same, only one instance object will be instantiated"""
    _instances = {}
    init_flag = False

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            SingleInstance.__init__(SingleInstance._instances[key], *args, **kwargs)
        return cls._instances[key]

    def __init__(self, *args, **kwargs):
        pass


class Rectangle(object):
    """
    Rectangular class , used to record the bounding box and available range of cells
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @property
    def area(self):
        """
        Return the Rectangle area,
        """
        return abs((self.x_max - self.x_min) * (self.y_max - self.y_min))

    def _intersectX(self, other):
        """
        Determine whether two rectangles intersect on the X-axis
        """
        if max(self.x_min, other.x_min) - min(self.x_max, other.x_max) >= 0:
            return False
        else:
            return True

    def _intersectY(self, other):
        """Determine whether two rectangles intersect on the Y-axis"""
        if max(self.y_min, other.y_min) - min(self.y_max, other.y_max) >= 0:
            return False
        else:
            return True

    def _include(self, other, self_max, other_max, self_min, other_min):
        """
        Determine if it contains, incoming x value represents in the x-axis direction, incoming y value represents in the y-axis direction
        """
        if self_min >= other_min:
            if self_max <= other_max:
                flag = other
                return True, flag  # indicate longer instance
            else:
                return False
        else:
            if self_min <= other_min:
                if self_max >= other_max:
                    flag = self
                    return True, flag
                else:
                    return False

    def _includeX(self, other):
        """
        Determine if there is an inclusion relationship between the X-axis of two rectangles
        """
        return self._include(other, self.x_max, other.x_max, self.x_min, other.x_min)

    def _includeY(self, other):
        """
        Determine if there is an inclusion relationship between the Y-axis of two rectangles
        """
        return self._include(other, self.y_max, other.y_max, self.y_min, other.y_min)

    def isIntersect(self, other):
        """
        Determine whether two rectangles intersect
        """
        if self._intersectX(other) and self._intersectY(other):
            return True
        else:
            return False

    def isInclude(self, other):
        """
        Determine whether two rectangles contain
        """
        if not self._includeX(other):
            return False
        else:
            if not self._includeY(other):
                return False
            else:
                if self._includeX(other)[1] != self._includeY(other)[1]:
                    return False
                else:
                    return True

    def draw(self, background=None, isShow=False, color=(255, 0, 0)):
        """
        draw the rectangle
        """
        import cv2
        if background is not None:
            _background = background
        else:
            _background = np.zeros(shape=(100, 100))

        if len(_background.shape) == 2:
            _background = cv2.cvtColor(convert_dtype(_background), cv2.COLOR_GRAY2RGB)
            # (bbox[2], bbox[0]), (bbox[3], bbox[1])
        cv2.rectangle(_background, (self.y_min, self.x_min), (self.y_max, self.x_max), color, 2)
        if isShow:
            cv2.imshow('rec', _background)
            cv2.waitKey()
        return _background


class Vector(np.ndarray):
    """
    2D plane vector class
    """

    def __new__(cls, x=None, y=None, shape=(2,), dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(cls, shape, dtype,
                              buffer, offset, strides, order)
        if x is None and y is None:
            obj.xy = None
            obj.x = x
            obj.y = y
        elif x is None and y:
            obj.xy = [0, y]
            obj.x = 0
            obj.y = y
        elif x and y is None:
            obj.xy = [x, 0]
            obj.x = x
            obj.y = 0
        else:
            obj.xy = [x, y]
            obj.x = x
            obj.y = y
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'xy', None)
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)

    @property
    def module(self):
        """
        Return the modulus of vector
        """
        if self.xy is None:
            return 0
        return np.linalg.norm([self.x, self.y])

    @property
    def cos(self):
        """
        Return the cosine value of vector
        """
        if not self.xy:
            return None
        return self.x / self.module

    def cosSimilar(self, other):
        """
        Compare the cosine similarity of two vectors, with a range of values [-1, 1]
        """
        if self.xy is None:
            if other.xy is None:
                return None
            elif other == Vector(0, 0):
                return 0
            else:
                return other.cos
        if self == other:
            return 1
        if other == Vector(0, 0):
            return 0
        if self == Vector(0, 0):
            if other.yx is None:
                return 0
            elif other == Vector(0, 0):
                return 1
            else:
                return other.cos
        return np.dot(self.xy, other.xy) / (self.module * other.module)

    def cosDistance(self, other):
        """
        Cosine distance, numerically equal to 1 minus cosine similarity, range of values [0,2]
        """
        return 1 - self.cosSimilar(other)

    def EuclideanDistance(self, other):
        """
        Return the Euclidean distance between two vectors
        """
        return np.sqrt((abs(self.x - other.x) ** 2 + abs(self.y - other.y) ** 2))

    def __len__(self):
        if self.xy is None:
            return 0
        return len(self.xy)

    def __str__(self):
        if self.xy is None:
            return "None Vector"
        return str(self.xy)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.module < other.module

    def __bool__(self):
        if not self.xy or self == Vector(0, 0):
            return False
        return True

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __mul__(self, other):
        if hasattr(other, '__float__'):
            return Vector(self.x * other, self.y * other)
        elif isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            raise TypeError(f'{type(other)} are not support the Multiplication operation with type Vector!')


class Cell(object):
    """
    Define the cell class, which is the core class of SC-Track. The smallest unit of all operations is the Cell instance,
    and the Cell is implemented as a conditional singleton mode: that is, different objects are generated according to
    the parameters passed in. If the parameters passed in are the same , only one object is instantiated. When defining
    the Cell object, Cell class can be guaranteed that there is only one Cell instance for the same cell in one frame,
    and differentCell instances are generated in different frames.
    If you want to instantiate a Cell object, at least its position information needs to be passed in, that is, all the
    xy coordinate points that constitute the outline of the cell.
    """
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super(Cell, cls).__new__(cls)
            cls._instances[key].__feature = None
            cls._instances[key].sort_value = 0
            cls._instances[key].__feature_flag = False
            cls._instances[key].__track_id = None
            cls._instances[key].__branch_id = None
            cls._instances[key].__inaccurate_track_id = None
            cls._instances[key].__is_track_id_changeable = True
            cls._instances[key].mitosis_start_flag = False
            cls._instances[key].__region = None
            cls._instances[key].__status = None
            cls._instances[key].__is_accurate_matched = False
            cls._instances[key].__match_status = False  # 匹配状态，如果参与匹配则设置为匹配状态，从未匹配则设置为False
        return cls._instances[key]

    def __init__(self, position=None, mcy=None, dic=None, phase=None, frame_index=None, flag=None):
        # if  Cell.init_flag is False:
        self.position = position  # [(x1, x2 ... xn), (y1, y2 ... yn)]
        self.mcy = mcy
        self.dic = dic
        self.phase = phase
        self.__id = None
        self.frame = frame_index
        self.__parent = None  # If a cell divides, record the __id of the cell's parent
        self.__move_speed = Vector(0, 0)
        self.polygon = Polygon([xy for xy in zip(*self.position)])

        if flag is None:
            self.flag = 'cell'
        else:
            self.flag = 'gap'
        Cell.init_flag = True
        # else:
        #     return

    def change_mitosis_flag(self, flag: bool):
        """
        When the cell enters mitosis for the first time, self.mitosis_start_flag is set to True, and when the cell
         completes division, it is reset to false
        :param flag: mitosis flag
        :return: None
        """
        self.mitosis_start_flag = flag

    @property
    @lru_cache(maxsize=None)
    def contours(self):
        """
        Convert the list of xy coordinate points into contour points list
        :return: if successful, return contours, else return None
        """
        points = []
        if self.position:
            for j in range(len(self.position[0])):
                x = int(self.position[0][j])
                y = int(self.position[1][j])
                points.append((x, y))
            contours = np.array(points)
            return contours
        else:
            return None

    @property
    def move_speed(self) -> Vector:
        """
        :return:  moving speed of the cell, the Vector object type
        """
        return self.__move_speed

    def update_speed(self, speed: Vector):
        """
        :param speed: new speed of the cell, Vector instance.
        :return: None
        """
        self.__move_speed = speed

    @staticmethod
    def polygon_centroid(vertex_coordinates):
        """
        Calculate the physical center of gravity of the cell counters.
        :param vertex_coordinates: list of cell outline coordinate points, the format is [[x1, x2,...xn], [y1, y2,..yn]]
        :return: physical center of gravity
        """
        x_coords = vertex_coordinates[0]
        y_coords = vertex_coordinates[1]
        n = len(x_coords)
        area = 0.0
        centroid_x = 0.0
        centroid_y = 0.0

        for i in range(n):
            j = (i + 1) % n
            cross_product = x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]
            area += cross_product
            centroid_x += (x_coords[i] + x_coords[j]) * cross_product
            centroid_y += (y_coords[i] + y_coords[j]) * cross_product

        area /= 2.0
        centroid_x /= 6.0 * area
        centroid_y /= 6.0 * area

        return centroid_x, centroid_y

    @property
    @lru_cache(maxsize=None)
    def center(self):
        """
        :return: cell physical center
        """
        # return np.mean(self.position[0]), np.mean(self.position[1])
        return self.polygon_centroid(self.position)

    @property
    @lru_cache(maxsize=None)
    def available_range(self) -> Rectangle:
        """
        Define the matchable range of the two frames before and after, the default is twice the horizontal and vertical
        coordinates of the cell
        :return: Rectangle instance
        """
        mult = config.AVAILABLE_RANGE_COEFFICIENT

        x_len = self.bbox[3] - self.bbox[2]
        y_len = self.bbox[1] - self.bbox[0]
        x_min_expand = self.bbox[2] - mult * x_len
        x_max_expand = self.bbox[3] + mult * x_len
        y_min_expand = self.bbox[0] - mult * y_len
        y_max_expand = self.bbox[1] + mult * y_len
        return Rectangle(y_min_expand, y_max_expand, x_min_expand, x_max_expand)

    @property
    def r_long(self):
        """
        :return: Cell bounding box long side radius
        """
        return max((self.bbox[3] - self.bbox[2]) / 2, (self.bbox[1] - self.bbox[0]) / 2)

    @property
    def r_short(self):
        """
        :return: Cell bounding box short side radius
        """
        return min((self.bbox[3] - self.bbox[2]) / 2, (self.bbox[1] - self.bbox[0]) / 2)

    @property
    def d_long(self):
        """
        :return: Long side diameter of cell bounding box
        """
        return 2 * self.r_long

    @property
    def d_short(self):
        """
        :return: Short side diameter of cell bounding box
        """
        return 2 * self.r_short

    @staticmethod
    @lru_cache(maxsize=None)
    def polygon_area(x, y):
        """
        Calculate cell area

        :param x: A list of x-coordinates of all points of the cell counters
        :param y: A list of y-coordinates of all points of the cell counters
        :return: Area of cell
        """
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @property
    @lru_cache(maxsize=None)
    def vector(self) -> Vector:
        """
        Returns the cell center point vector, starting from the upper left corner of the image as the origin.
        :return: Vector instance
        """
        return Vector(*self.center)

    @property
    def area(self):
        """
        :return: Area of a cell
        """
        return self.polygon_area(tuple(self.position[0]), tuple(self.position[1]))

    def set_region(self, region):
        self.__region = region

    def set_status(self, status: TreeStatus | CellStatus):
        self.__status = status

    @property
    def status(self):
        return self.__status

    @property
    def is_accurate_matched(self):
        return self.__is_accurate_matched

    @is_accurate_matched.setter
    def is_accurate_matched(self, value: bool):
        self.__is_accurate_matched = value

    @property
    def region(self):
        return self.__region

    @property
    @lru_cache(maxsize=None)
    def bbox(self):
        """bounding box coordinates"""
        x0 = math.floor(np.min(self.position[0])) if math.floor(np.min(self.position[0])) > 0 else 0
        x1 = math.ceil(np.max(self.position[0]))
        y0 = math.floor(np.min(self.position[1])) if math.floor(np.min(self.position[1])) > 0 else 0
        y1 = math.ceil(np.max(self.position[1]))
        return y0, y1, x0, x1

    def move(self, speed: Vector, time: int = 1):
        """
        :param speed: move speed， Vector object instance
        :param time: move time，frame
        :return: New Cell instance after moving
        """
        new_position = [tuple([i + speed.x * time for i in self.position[0]]),
                        tuple([j + speed.y * time for j in self.position[1]])]
        new_cell = Cell(position=new_position, mcy=self.mcy, dic=self.dic, phase=self.phase, frame_index=self.frame)
        new_cell.set_track_id(self.__track_id, 0)
        return new_cell

    def set_feature(self, feature):
        """set Feature object for cell"""
        self.__feature = feature
        self.__feature_flag = True

    @property
    def feature(self):
        if self.__feature_flag:
            return self.__feature
        else:
            raise ValueError("No available feature! ")

    def set_track_id(self, __track_id, status: 0 | 1):
        """Set the track_id for the cell"""
        if self.__is_track_id_changeable:
            if status == 1:
                self.__track_id = __track_id
                self.__is_track_id_changeable = False
            elif status == 0:
                self.__inaccurate_track_id = __track_id
            else:
                raise ValueError(f'status {status} is invalid!')
        else:
            # warnings.warn('cannot change the accurate track_id')
            pass

    def set_match_status(self, status: bool | str):
        self.__match_status = status

    @property
    def is_be_matched(self):
        """I
        f participated in the match, return match status, otherwise, False
        """
        return self.__match_status

    @is_be_matched.setter
    def is_be_matched(self, value):
        self.__match_status = value

    def set_parent_id(self, __parent_id):
        self.__parent = __parent_id

    def set_cell_id(self, cell_id):
        self.__id = cell_id

    def set_branch_id(self, branch_id):
        self.__branch_id = branch_id

    def update_region(self, **kwargs):
        """update annotation region information ,add the tracking results."""
        new_region = self.region
        if new_region:
            if 'branch_id' in kwargs:
                new_region['region_attributes']['branch_id'] = kwargs['branch_id']
            if 'track_id' in kwargs:
                new_region['region_attributes']['track_id'] = kwargs['track_id']
            self.set_region(new_region)

    @property
    def branch_id(self):
        return self.__branch_id

    @property
    def cell_id(self):
        """Cell id, parent cell and daughter cell have different value"""
        return self.__id

    @property
    def parent(self):
        return self.__parent

    @property
    def track_id(self):
        return self.__track_id

    def draw(self, background=None, isShow=False, color=(255, 0, 0)):
        import cv2
        if background is not None:
            _background = background
        else:
            _background = np.ones(shape=(2048, 2048, 3), dtype=np.uint8)

        if len(_background.shape) == 2:
            _background = cv2.cvtColor(convert_dtype(_background), cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(_background, self.contours, -1, color, 3)
        cv2.rectangle(_background, (self.bbox[2], self.bbox[0]), (self.bbox[3], self.bbox[1]), color, 5)
        if isShow:
            cv2.imshow('rec', _background)
            cv2.resizeWindow('rec', 500, 500)
            cv2.waitKey()
        return _background

    def __contains__(self, item):
        return True if self.available_range.isIntersect(Rectangle(*item.bbox)) else False

    def __str__(self):
        if self.position:
            return f" Cell at ({self.center[0]: .2f},{self.center[1]: .2f}), frame {self.frame}, {self.phase}, branch {self.__branch_id}"
        else:
            return "Object Cell"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.position and other.position:
            self_module = np.linalg.norm([np.mean(self.position[0]), np.mean(self.position[1])])
            other_module = np.linalg.norm([np.mean(other.position[0]), np.mean(other.position[1])])
            return self_module < other_module
        else:
            raise ValueError("exist None object of ")

    def __eq__(self, other):
        return self.position == other.position and self.frame == other.frame

    def __hash__(self):
        return id(self)
