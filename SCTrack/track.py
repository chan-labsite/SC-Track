# coding=utf-8

from __future__ import annotations

import logging
import os.path
from SCTrack import reclassification
from SCTrack.utils import mask_to_json


def start_track(fannotation: str | dict, fout, basename, track_range=None, fimage=None, fbf=None,
                export_visualization=True,
                track_to_json=True):
    """
     :param track_range: Track frame number range
     :param visualize_background_image: track background image
     :param basename:
     :param fannotation: segmentation output result, json file or dict
     :param fout: Tracking output folder path
     :param fimage: raw image path, can be empty
     :param fbf: Bright field image path, can be empty
     :param export_visualization: Whether to export the tracking visualization file, if yes, it will export a multi-frame tif file
     :param track_to_json: Whether to write the tracking result into fjson, if yes, a new json file will be generated
     :return: None
    """

    if type(fannotation) is str:
        if not fannotation.endswith('.json'):
            logging.info('convert mask to annotation file...')
            annotation = mask_to_json(fannotation, xrange=track_range)
        else:
            annotation = fannotation
    else:
        annotation = fannotation

    result_save_path = os.path.join(fout, 'tracking_output')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    reclassification.run(annotation=annotation, output_dir=result_save_path, track_range=track_range, dic=fbf,
                         mcy=fimage,
                         save_visualize=export_visualization, visualize_background_image=fimage,
                         track_to_json=track_to_json, basename=basename)


if __name__ == '__main__':
    i = 8
    annotation = rf"G:\paper\test\Data{i}\SEG.tif"
    mcy_img = rf"G:\paper\test\Data{i}\01.tif"
    start_track(annotation, rf"G:\paper\test\Data{i}", 'mcy', 1000,
                mcy_img)

    # annotation = r"G:\杂项\example\example-annotation.json"
    # mcy_img = r"G:\杂项\example\example-image.tif"
    # dic_img = r'G:\杂项\example\example-bf.tif'
    # start_track(annotation, r"G:\杂项\example", 'mcy', 30,
    #             mcy_img)
