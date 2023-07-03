#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: main.py
# @Author: Li Chengxin 
# @Time: 2023/7/2 19:51


import argparse
import json
import logging
import os
import sys
import time
import imagesize


logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%M-%d %H:%M:%S", level=logging.INFO)

parser = argparse.ArgumentParser(description="Welcome to use CCDeep!", add_help=False)
help_content = """
    using this script to auto segment the cell images and identify each cell's  cycle phase.
    usage:
        python main.py -pcna <pcna image filepath>  -bf <bf image filepath> -o [optional] <output result filepath> 
        -t [optional]
"""

parser.add_argument('-t', "--track", action='store_true', help='Optional parameter, track or not')
parser.add_argument("-h", "--help", action="help", help=help_content)
parser.add_argument('-p', "--pcna", default=False, help="input image filepath of pcna")
parser.add_argument('-o', "--output", default=False, help='output json file path')
parser.add_argument('-bf', "--bf", default=False, help='input image filepath of bright field')
parser.add_argument('-ot', "--ot", default=False, help='tracking output result saved dir')
parser.add_argument('-js', "--js", default=False, help='annotation json file  path')
parser.add_argument('-r', "--range", default=False,
                    help='tracking frame range, default is None, means tracking whole timelapse')

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(0)

if args.pcna is False:
    logging.error("pcna image must be given!")
    sys.exit(-1)
else:
    pcna = args.pcna
    image_width, image_height = imagesize.get(pcna)
    bf = args.bf
if args.output is False:
    output = os.path.join(os.path.dirname(pcna), 'output.json')
    logging.warning(f"-o  not provided, using the default output file name: {output}")
else:
    if not args.output.endswith('.json'):
        output = os.path.join(os.path.dirname(args.pcna), 'output.json')
    else:
        output = args.output

if args.track is True and args.js is False:
    logging.error("If you just want to do tracking, please give the `-js` parameter")
    sys.exit(-1)

if args.js:
    jsons = args.js
else:
    jsons = None

start_time = time.time()

if args.track:
    import track

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
        track.start_track(fjson=jsons, fpcna=args.pcna, fbf=None, fout=track_output, track_range=xrange,
                          export_visualization=True, basename=os.path.basename(args.pcna).replace('.tif', ''))

end_time = time.time()
if args.track:
    logging.info(f'tracking cost time: {end_time- start_time:.4f}')