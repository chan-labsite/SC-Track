

<div align="center" style="text-align: center; font-size: xx-large;"> <b><a href=https://github.com/frozenleaves/SC-Track>SC-Track :  Tracking  for  Single  Cell</a></b></div>

<div align="center"> <img src="docs/icon/license.svg" width = 200 /> <img src="docs/icon/wheel.svg" width = 70 />  <img src="docs/icon/docs.svg" width = 80 /> <img src="docs/icon/Python-version.svg" width = 200 /> </div> 

### What's  SC-Track?

SC-Track is an efficient algorithm for dynamic tracking of single cells on different time-lapse microscope images. 
It can use the segmentation results of various models to efficiently track single cells and reconstruct cell lines. 
It can track multi-generational cell division events without any additional information, only using the outline information of cells; 
and can reduce the noise of the segmentation, so as to use the noise segmentation results to generate accurate cell lineages. 
Its cascade-caching model can efficiently deal with segmentation loss, and its TPS algorithm can perform accurate reclassification 
for users with cell classification needs (such as classification of different cell cycle phases). 
SC-Track allows users to use different segmentation results as input, including the JSON annotation file format supported by VGG image annotator, 
and the common mask grayscale image format. The export results include track table, visualized labeled image, 
JSON file containing tracking information (which can be imported into VGG image annotator for viewing), 
and a collection of TrackingTree structure tree files. Users can perform more detailed downstream analysis on the track table, 
view the tracking results through visualized results, and modify the track table or track json file to manually correct tracking errors. 
SC-Track is not only suitable for small timelapse analysis, but also suitable for long time and high cell density timelapse analysis of thousands of frames.




### Why using  SC-Track?

-   The current mainstream methods for image segmentation all use deep learning, and the output results contain noises of varying intensities. SC-Track is currently the only algorithm that can use these noise data for accurate single-cell tracking and lineage reconstruction.
- SC-Track is compatible with the output results of most of the existing mainstream segmentation models, as well as manual segmentation results, including Cellpose, DeepCell, Stardist, etc. Users can choose a more advanced and suitable segmentation model according to the cell type to split.
- SC-Track can efficiently track multiple targets between frames without relying on global information, and can be used for real-time tracking.
- SC-Track is implemented in Python, which has strong scalability, convenient and quick installation, and low dependency.



### How to use SC-Track?

```
To use SC-Track, please follow the Installation steps first. It does not require too many settings during its use. When you only have a single-channel segmentation result, we require that your segmentation result must be a mask grayscale file in the form of 2D+t in tiff format. The cells in each mask need to guarantee their pixel values. is unique; or a JSON comment file. The specific format can refer to our example.

When the segmentation result is a mask, please run: sctrack -p image.tif -a mask.tif.
When the segmentation result is an annotation json file, please run: sctrack -p image.tif -a annotation.json.
Where image.tif is the original image, mask.tif, and annotation.json are annotation files. The original image may not be provided, but if the original image is not provided, the visualization result cannot be output.
```



### Installation

```
Requirement: Python >= 3.7

Windows: pip isntall SC-Track
Linux/Macos: pip3 isntall SC-Track

```

-   Note： On `Windows`, the requirement package `pylibtiff `cannot directly install by pip, please install with this command:

    `conda install libtiff`

    or you can download the wheel package from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff), and then using `pip install pylibtiff.whl` to install.

    On `Linux` or `Macos`, just using `pip install pylibtiff` to install.





### Usage

```python
We provide a command line tool, you only need to run the sctrack tool on the command line. To automate batch processing of a large number of files, please refer to our source code documentation.
Its basic usage is:
    
from SCTrack import strat_track

image = 'path/to/image.tif'

# using mask annotation
annotation_mask = '/path/to/annotation.tif'
start_track(fannotation=annotation_mask, fimage=image)

# using json file annotation
annotation_json = '/path/to/annotation.json'
start_track(fannotation=annotation_json, fimage=image)
```

