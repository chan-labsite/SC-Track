from __future__ import annotations

import json

import numpy as np
import cv2
from typing import Tuple, List
import matplotlib.pyplot as plt
import openpyxl as px
import tifffile
from SCTrack.prepare import convert_dtype


class Stack(object):
    def __init__(self):
        self.start_frame = None
        self.end_frame = None
        self.center_info = None
        self.id = None

    def __len__(self):
        if self.start_frame is None:
            return 0
        else:
            # return self.end_frame - self.start_frame + 1
            return len(self.center_info)

    def __str__(self):
        return f"start frame: {self.start_frame}, end frame: {self.end_frame}, total: " \
               f"{self.end_frame - self.start_frame + 1}, len: {len(self.center_info)} " \
               f"\ncenter info: {self.center_info}"

    def __repr__(self):
        return self.__str__()


class RefinedParser(object):
    def __init__(self, path):
        wb = px.load_workbook(path)
        sheet = wb[wb.sheetnames[0]]
        self.frame_details = sheet['A'][1:]
        self.id_details = sheet['B'][1:]
        self.lineage_details = sheet['C'][1:]
        self.parent_id_details = sheet['D'][1:]
        self.phase_details = sheet['S'][1:]
        self.center0 = sheet['F'][1:]
        self.center1 = sheet['E'][1:]

    def parse_id(self):
        id_record = []
        id_info = []
        current_index = 0
        for i in range(len(self.id_details)):
            if self.id_details[i].value not in id_record:
                id_record.append(self.id_details[i].value)
        for _id in id_record:
            length = 0
            start = current_index
            for j in self.id_details[current_index:]:
                if j.value == _id:
                    length += 1
                else:
                    current_index += length
                    break
            end = start + length - 1
            id_info.append({'id': _id, 'start': start, 'end': end, 'continue': length})
            # print({'id': _id, 'start': start, 'end': end, 'continue': length})
        return id_info

    def parse_position(self):
        id_info = self.parse_id()
        position = []
        for i in id_info:
            _id = i['id']
            start_index = i['start']
            end_index = i['end']
            # print(_id, start_index, end_index, self.frame_details[start_index].value,
            #       self.frame_details[end_index].value)
            position.append({'id': _id, 'start_index': start_index, 'end_index': end_index,
                             'start_frame': self.frame_details[start_index].value,
                             'end_frame': self.frame_details[end_index].value})
        return position

    def get_stack(self):
        positions = self.parse_position()
        stacks = []
        for i in positions:
            stack = Stack()
            stack.start_frame = i.get('start_frame')
            stack.end_frame = i.get('end_frame')
            stack.center_info = []
            stack.id = i.get('id')
            for index in range(i.get('start_index'), i.get('end_index') + 1):
                stack.center_info.append((round(self.center0[index].value), round(self.center1[index].value)))
            stacks.append(stack)
        return stacks


class JsonParser(object):
    def __init__(self, file):
        with open(file) as f:
            self.data = json.load(f)
        self.index_map = {}
        __index = 0
        for frame in self.data:
            self.index_map[__index] = frame
            __index += 1

    def __len__(self):
        return len(self.index_map)

    def parse_json(self):
        all_coords = {}
        for frame in self.data:
            regions = self.data[frame]['regions']
            coords = []
            for j in regions:
                all_x = j['shape_attributes']['all_points_x']
                all_y = j['shape_attributes']['all_points_y']
                coords.append((all_x, all_y))
            all_coords[frame] = coords
        return all_coords

    def get_coords_by_frame_name(self, frame_name):
        regions = self.data[frame_name]['regions']
        coords = []
        for j in regions:
            all_x = j['shape_attributes']['all_points_x']
            all_y = j['shape_attributes']['all_points_y']
            coords.append((all_x, all_y))
        return coords

    def get_coords_by_frame_index(self, frame_index=None):
        frame_name = self.index_map[frame_index]
        return self.get_coords_by_frame_name(frame_name)

    def get_frame_name_by_index(self, index):
        return self.index_map[index]


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


def coordinate2mask(coords: np.ndarray | list | tuple, value: int = 255, image_size: Tuple[int, int] = None) -> \
        List[Mask]:
    results = []
    for coord in coords:
        if image_size is None:
            mask = np.zeros((2048, 2048), dtype=np.uint8)
        else:
            mask = np.zeros(image_size, dtype=np.uint8)
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


def coord2counter(coord):
    points = []
    for j in range(len(coord[0])):
        x = int(coord[0][j])
        y = int(coord[1][j])
        points.append((x, y))
    contours = np.array(points)
    return contours


def link(json_file, refined_file):
    ref_file = refined_file
    refined = RefinedParser(ref_file)
    stacks = refined.get_stack()
    json_parser = JsonParser(json_file)
    for stack in stacks:
        linked_masks = []
        centers = stack.center_info
        start_frame = stack.start_frame
        for index in range(len(stack.center_info)):
            center = centers[index]
            coords = json_parser.get_coords_by_frame_index(index + start_frame)
            masks = coordinate2mask(coords)
            for m in masks:
                if (abs(center[0] - m.center[0]) <= 10) and (abs(center[1] - m.center[1]) <= 10):
                    # print(f"id index: {index} detected! at frame: {index + start_frame}.")
                    m.frame_index = index + start_frame
                    linked_masks.append(m)
                    break
        yield linked_masks


def extractRoiFromImg(images: str | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract the area in the original image according to the mask. Note that it can only be a single-channel image.
    If it is rgb, please convert it to a grayscale image first.
    :param images: original images
    :param mask: mask file
    :return: single cell image data
    """
    if type(images) is str:
        src = cv2.imread(images, -1)
    else:
        src = images
    new_src = convert_dtype(src)
    dst = np.zeros_like(new_src, dtype=np.uint8)
    cv2.copyTo(new_src, mask, dst)
    return dst


def csv2mask(jsonfile, excelfile, mask_filename):
    jp = JsonParser(jsonfile)
    rp = RefinedParser(excelfile)
    stacks = rp.get_stack()
    frame_info = {}
    for index in range(len(jp)):
        frame_info[index] = set()
    print(frame_info)
    for stack in stacks:
        start = stack.start_frame
        for i in range(len(stack)):
            frame_info[start + i].add((stack.center_info[i], stack.id))
    all_masks = []
    for i in range(len(frame_info)):
        coords = jp.get_coords_by_frame_index(i)
        masks = np.zeros((2048, 2048), dtype=np.uint8)
        match_count = 0
        matched_center = []
        for j in frame_info[i]:
            for coord in coords:
                if (abs(j[0][0] - round(float(np.mean(coord[0])))) < 8 and
                        abs(j[0][1] - round(float(np.mean(coord[1])))) < 8):
                    ret = coordinate2mask([coord], j[1])
                    matched_center.append((round(float(np.mean(coord[0]))), round(float(np.mean(coord[1])))))
                    match_count += 1
                    masks += ret[0].mask
        for coord in coords:
            if (round(float(np.mean(coord[0]))), round(float(np.mean(coord[1])))) not in matched_center:
                e_ret = coordinate2mask([coord], 255)
                masks += e_ret[0].mask
        plt.imshow(masks, cmap='gray')
        plt.show()
        print(match_count)
        all_masks.append(masks)
    all_masks = np.array(all_masks)
    tifffile.imsave(mask_filename.replace('.tif', '-1.tif'), all_masks[:all_masks.shape[0] // 2])
    tifffile.imsave(mask_filename.replace('.tif', '-2.tif'), all_masks[all_masks.shape[0] // 2:])


if __name__ == '__main__':
    pass
