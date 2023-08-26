

# <div align="center" style="text-align: center; font-size: 32px;"> <b><a href=https://github.com/chan-labsite/SC-Track>SC-Track : A Biologically Inspired Algorithm for Generating Accurate Single Cell Lineages</a></b></div>

<div align="center"> <img src="docs/icon/license.svg" width = 220 /> <img src="docs/icon/wheel.svg" width = 70 />  <img src="docs/icon/docs.svg" width = 80 /> <img src="docs/icon/Python-version.svg" width = 200 /> </div> 

## What's SC-Track?

SC-Track is an efficient multi-object cell tracking algorithm that can generate accurate single cell linages from the segmented nucleus of timelapse microscopy images. It employs a probabilistic cache-cascade matching model that can tolerate noisy segmentation and classification outputs, such as randomly missing segmentations and false detections/classifications from deep learning models. It also has a built-in cell division detection module that can assign mother-daughter relationships using the segmentation masks of cells.

SC-Track allows users to use two different segmentation results as input. It can either take:
1) A greyscale Multi-TIFF image, where every segmented instance of cells is given a unique pixel value and the background set as 0.
2) A VGG image annotator (VIA2) compatible JSON file, containing the segmented instances and class infromation of every segmented cell. 

SC-Track will output the tracking results in a track table for downstream analysis. If SC-track is run from command prompt, it will also produce a png image folder containing the labelled cell linages, VIA2 combatible JSON file containing the tracking information and a collection of TrackingTree files to aid visualisation and analysis of the generated single cell tracks. 

----------


## Why use SC-Track?

1) It is generally accepted that segmented instances of objects detected from state-of-the-art deep learning convolution neural networks such as U-net and Mask RCNN will produce low instances of segmentation and classification errors. The most common errors are false detections, failed detections or wrong classification of detected objects. These inaccuracies often confound commonly used tracking algorithms. SC-Track works well with noisy outputs from these deep learning models to generate single-cell linages from these segmented timelapse microscopy images.
2) SC-Track is compatible with the output results from manually segmentated images as well as from most existing mainstream segmentation models, such as Cellpose, DeepCell, and StarDist. 
3) SC-Track probabilistic cache-cascade matching model can efficiently track multiple targets between frames and therefore can be used for real-time tracking.
4) SC-Track is implemented in Python, making it easy to be integrated into any deep learning based segmentation pipelines to be used to generate single-cell tracks from timelapse microscopy images.



-------

## How to use SC-Track?

If you have a single-channel segmentation result, the segmentation results should be in a grayscale 2D+t tiff file with each segmented cell instance containing a unique pixel value with the beackground given a pixel value of 0. You will need to run SC-Track from the folder containing the image and mask TIFF files. The command to call SC-Track is as follows:
```
sctrack -p image.tif -a mask.tif
```

When the segmentation results are contained in a VIA2 compatible JSON file, you will need to run SC-Track from the folder containing the image and JSON files. The command to call SC-Track is as follows: 
```
sctrack -p image.tif -a annotation.json
```
The file "image.tif" corresponds to the microscopy timelapse image stack, "mask.tif" represents the greyscale segmented cell instances and "annotation.json" is the VIA2 compatible JSON annotation files. SC-Track can run without the corresponding "image.tif" file. In this case, SC-Track will output the tracking results without the png image folder containing the labelled cell linages.


----------

## Installation

SC-Track can be installed using the following methods listed below. It will automatically install all the required dependencies.
Requirement: Python >= 3.7

For windows based installations:

```
pip install SC-Track
```

For Linux/macOS:
```
pip3 install SC-Track
```

Note： On `Windows`, the package `pylibtiff `cannot be directly installed by pip, please install `pylibtiff` with the following command:

```
conda install libtiff
```

Alternatively, you can download the wheel package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff), and then use the command `pip install pylibtiff.whl` to install.



-----------------------

## Usage

To automate batch processing of a large number of files, please refer to our source code documentation. It's basic implementation is:

```python
from SCTrack.track import start_track

image = 'path/to/image.tif'

# using mask annotation
annotation_mask = '/path/to/annotation.tif'
start_track(fannotation=annotation_mask, fimage=image, basename='image', track_range=None, fout='/path/to/dir')

# using json file annotation
annotation_json = '/path/to/annotation.json'
start_track(fannotation=annotation_json, fimage=image, basename='image', track_range=None, fout='/path/to/dir')
```



------

## Using guidance

To see the using guidance, please refer our [quick-start](./notebook/quick-start.ipynb).

---------

## API  Documentation

For more information, please see the [reference documents](https://htmlpreview.github.io/?https://github.com/frozenleaves/SC-Track/blob/master/docs/build/html/index.html).


## Reference

Please cite our paper if you found this package useful. 
```
SC-Track: A Biologically Inspired Algorithm for Generating Accurate Single Cell Lineages. 
Chengxin Li，Shuang Shuang Xie，Jiaqi Wang，Septavera Sharvia，Kuan Yoow Chan
```
