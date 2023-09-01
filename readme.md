

## <div align="center" style="text-align: center; font-size: 32px;"> <b><a href=https://github.com/frozenleaves/SC-Track>SC-Track :  Tracking  for  Single  Cell</a></b></div>

<div align="center"> <img src="docs/icon/license.svg" width = 220 /> <img src="docs/icon/wheel.svg" width = 70 />  <img src="docs/icon/docs.svg" width = 80 /> <img src="docs/icon/Python-version.svg" width = 200 /> </div> 

- ## What's SC-Track?

    SC-Track is an efficient multi-object cell tracking algorithm that can generate accurate single cell linages from the segmented nucleus of timelapse microscopy images. It employs a probabilistic cache-cascade matching model that can tolerate noisy segmentation and classification outputs, such as randomly missing segmentations and false detections/classifications from deep learning models. It also has a built-in cell division detection module that can assign mother-daughter relationships using the segmentation masks of cells.

    SC-Track allows users to use two different segmentation results as input. It can either take:

    1) A greyscale Multi-TIFF image, where every segmented instance of cells is given a unique pixel value and the background set as 0.
    2) A VGG image annotator (VIA2) compatible JSON file, containing the segmented instances and class infromation of every segmented cell.


    ----------


    ## Why use SC-Track?

    1) It is generally accepted that segmented instances of objects detected from state-of-the-art deep learning convolution neural networks such as U-net and Mask RCNN will produce low instances of segmentation and classification errors. The most common errors are false detections, failed detections or wrong classification of detected objects. These inaccuracies often confound commonly used tracking algorithms. SC-Track works well with noisy outputs from these deep learning models to generate single-cell linages from these segmented timelapse microscopy images.
    2) SC-Track is compatible with the output results from manually segmentated images as well as from most existing mainstream segmentation models, such as Cellpose, DeepCell, and StarDist. 
    3) SC-Track probabilistic cache-cascade matching model can efficiently track multiple targets between frames and therefore can be used for real-time tracking.
    4) SC-Track is implemented in Python, making it easy to be integrated into any deep learning based segmentation pipelines to be used to generate single-cell tracks from timelapse microscopy images.



-------

### How to use SC-Track?

````markdown
If you have a single-channel segmentation result, the segmentation results should be in a grayscale 2D+t tiff file with
each segmented cell instance containing a unique pixel value with the beackground given a pixel value of 0. You will need
to run SC-Track from the folder containing the image and mask TIFF files. The command to call SC-Track is as follows:
```
sctrack -i /path/to/image.tif -a /path/to/mask.tif
```

When the segmentation results are contained in a VIA2 compatible JSON file, you will need to run SC-Track from the folder
containing the image and JSON files. The command to call SC-Track is as follows: 
```
sctrack -i /path/to/image.tif -a /path/to/annotation.json
```
The file "image.tif" corresponds to the microscopy timelapse image stack, "mask.tif" represents the greyscale segmented
cell instances and "annotation.json" is the VIA2 compatible JSON annotation files. SC-Track can run without the
corresponding "image.tif" file. In this case, SC-Track will output the tracking results without a corresponding
png image folder containing the labelled cell linages.


If you run with this command, and solve the related dependency problem, you should be able to get the following output:

|__image.tif
|__annotation.json
|__tracking_output\
   ├─TrackTree\
   └─track_visualization.tif\
   └─track.csv
   └─result_with_track.json
   
The `TrackTree` folder contains the detailed information of each TrackTree built during the tracking process,
 the `track_visualization.tif` folder contains the visual image information of the tracking(if you close the
visualization option, this content will not be generated), and `track.csv` is a detailed table of the tracking results.
For specific information about the track.csv, see https://github.com/chan-labsite/SC-Track/blob/master/notebook/quick-start.ipynb.
The content of `result_with_track.json` is consistent with the structure of annotation.json, but result_with_track.json
contains track information and smoothed classification information (if exists).

Normally, when the visualization results are exported, running our demo data set will take no more than `3` minutes
from start to finish; if the visualization results are not exported, the running time will not exceed `2` minutes.
````

For the demo dataset, you can visit [zenodo](https://zenodo.org/record/8284987)  to download. 



----------

### Installation

```
Requirement: Python >= 3.7

Windows: pip isntall SC-Track
Linux/Macos: pip3 isntall SC-Track

For details on dependencies, you can view https://github.com/chan-labsite/SC-Track/blob/master/requirements.txt
```

-   Note： On `Windows`, the requirement package `pylibtiff `cannot directly install by pip, please install with this command:

    `conda install libtiff`

    or you can download the wheel package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff), and then using `pip install pylibtiff.whl` to install.

    On `Linux` or `Macos`, just using `pip install pylibtiff` to install, all requirement packages will be automatically installed.

    In the case of a good network, the installation process usually takes no more than 5 minutes.



-----------------------

### Usage

```python
We provide a command line tool, you only need to run the sctrack tool on the command line. To automate 
batch processing of a large number of files, please refer to our source code documentation.
Its basic usage is:
    
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

### Using guidance

To see the using guidance, please refer our [quick-start](./notebook/quick-start.ipynb).

---------

### API  Documentation

For more information, please see the [reference documents](https://htmlpreview.github.io/?https://github.com/frozenleaves/SC-Track/blob/master/docs/build/html/index.html).



