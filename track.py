from __future__ import annotations

import logging
import os.path
import reclassification
from utils import mask_to_json


def start_track(fannotation: str | dict, fout, basename, track_range=None, fimage=None, fbf=None,
                export_visualization=True,
                track_to_json=True):
    """

    :param track_range: Track帧数范围
    :param visualize_background_image: track背景图
    :param basename:
    :param fannotation: 分割输出结果，json文件或者dict
    :param fout:  tracking输出文件夹路径
    :param fimage: pcna图像路径， 可为空
    :param fbf:  明场图像路径，可为空
    :param export_visualization: 是否导出tracking可视化文件，如果是，会导出一个多帧tif文件
    :param track_to_json:  是否将tracking结果写入到fjson中，如果是，会生成一个新的json文件
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
#
# start_track(r'G:\20x_dataset\evaluate_data\src01\result-GT.json',r'G:\20x_dataset\evaluate_data\src01', 'mcy', 40, fpcna=r'G:\20x_dataset\evaluate_data\src01\mcy.tif')
