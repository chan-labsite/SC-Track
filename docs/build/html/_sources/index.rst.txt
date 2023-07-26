.. SC-Track documentation master file, created by
   sphinx-quickstart on Tue Jul  4 12:08:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SC-Track API documentation!
====================================

SC-Track is an efficient algorithm for dynamic tracking of single cells on different time-lapse microscope images. It can use the segmentation results of various models to efficiently track single cells and reconstruct cell lines. It can track multi-generational cell division events without any additional information, only using the outline information of cells; and can reduce the noise of the segmentation, so as to use the noise segmentation results to generate accurate cell lineages. Its cascade-caching model can efficiently deal with segmentation loss, and its TCS algorithm can perform accurate reclassification for users with cell classification needs (such as classification of different cell cycle phases). SC-Track allows users to use different segmentation results as input, including the JSON annotation file format supported by VGG image annotator, and the common mask grayscale image format. The export results include track table, visualized labeled image, JSON file containing tracking information (which can be imported into VGG image annotator for viewing), and a collection of TrackingTree structure tree files. Users can perform more detailed downstream analysis on the track table, view the tracking results through visualized results, and modify the track table or track json file to manually correct tracking errors. SC-Track is not only suitable for small timelapse analysis, but also suitable for long time and high cell density timelapse analysis of thousands of frames.


.. toctree::
   :maxdepth: 10
   :caption: Contents:

   modules.rst

.. toctree::
   :maxdepth: 10
   :caption: API References:

   python_apis/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
