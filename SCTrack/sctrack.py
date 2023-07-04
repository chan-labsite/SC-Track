#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: main.py
# @Author: Li Chengxin 
# @Time: 2023/7/2 19:51


import argparse
import track
import logging
import os
import sys
import time
import imagesize


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%M-%d %H:%M:%S", level=logging.INFO)

parser = argparse.ArgumentParser(description="", add_help=False)
help_content = """
    Welcome to use SC-Track!
    using this script to auto tracking the single cell images and identify each cell's type.\n
    usage:
        python main.py -image <image image filepath>  -bf <bf image filepath> -o [optional] <output result filepath> 
        -t [optional]
"""

parser.add_argument("-h", "--help", action="help", help=help_content)
parser.add_argument('-i', "--image", default=False, help="input image filepath for nuclear")
parser.add_argument('-o', "--output", default=False, help='output json file path')
parser.add_argument('-bf', "--bf", default=False, help='input image filepath of bright field')
parser.add_argument('-ot', "--ot", default=False, help='tracking output result saved dir')
parser.add_argument('-a', "--annotation", default=False, help='annotation file path, json file or tiff mask file.')
parser.add_argument('-r', "--range", default=False,
                    help='tracking frame range, default is None, means tracking whole timelapse')

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(0)

if args.image is False:
    logging.warning("Raw image not be given, so that the visualization result will not be support!")
    visualization = False
else:
    image = args.image
    visualization = True
    image_width, image_height = imagesize.get(image)
    bf = args.bf

if args.annotation is False:
    logging.error('You must provide segmentation annotation file for tracking, support json format or mask tif format.')
    sys.exit(-1)
else:
    annotation = args.annotation

if args.output is False:
    output = os.path.join(os.path.dirname(annotation), 'output.json')
    logging.warning(f"-o  not provided, using the default output file name: {output}")
else:
    if not args.output.endswith('.json'):
        output = os.path.join(os.path.dirname(args.image), 'output.json')
    else:
        output = args.output

start_time = time.time()

if args.ot:
    track_output = args.ot
else:
    if args.range is False:
        xrange = None
    else:
        try:
            xrange = int(args.range)
        except ValueError:
            logging.error(f'param <-r/--range >={args.range}, the value must be int!')
            sys.exit(-1)
    track_output = os.path.dirname(output)
    logging.info(f"Tracking result will saved to {track_output}\\tracking_output.")
    logging.info('start tracking ...')
    track.start_track(fannotation=annotation, fimage=args.image, fbf=None, fout=track_output, track_range=xrange,
                      export_visualization=visualization, basename=os.path.splitext(os.path.basename(args.annotation))[0])

end_time = time.time()

logging.info(f'tracking cost time: {end_time- start_time:.4f}')