

### What's  SC-Track?

```
SC-Track是一款用于对不同延时显微镜图像进行单细胞动态追踪的一种高效解决方案，它能够利用各种模型的分割结果对单细胞进行高效追踪，并重建细胞系。它可以在没有任何额外信息的情况下，仅仅利用细胞的轮廓信息，追踪多代细胞分裂事件；并能够对分割的噪音进行降噪，从而利用噪音分割结果，产生精确的细胞谱系。它的级联-缓存模型可以高效应对分割丢失的情况，它的TPS算法，可以针对有细胞分类需求的用户（例如分类不同细胞周期），进行准确的重分类。SC-Track允许用户使用不同的分割结果作为输入，包括VGG image annotator所支持的JSON注释文件格式，以及常见的mask灰度图格式。导出结果包括track table，可视化后的标注图像，包含tracking信息的JSON文件（可导入到VGG image annotator中查看），以及TrackingTree结构树文件合集。用户可以针对track table进行更细致的下游分析，可以通过可视化结果查看追踪结果，可以修改track table或者track json file从而手动修正追踪错误。SC-Track不仅适用于小型timelapse分析，也适用于上千帧的长时间，高细胞密度timelapse分析。
```



### Why using  SC-Track?

-   目前用于图像分割的的主流方法都是采用深度学习，其输出结果均包含强度不一的噪音，SC-Track是目前唯一能够利用这些噪音数据进行精准单细胞追踪和谱系重建的算法。
-   SC-Track能够兼容现有的绝大多数主流分割模型的输出结果，以及手动分割结, 包括Cellpose， DeepCell， Stardist等等，用户可以根据细胞类型选择一个更为先进，更为合适的分割模型来进行分割。
-   SC-Track能够高效地进行帧与帧之间的多目标追踪，而不依赖全局信息，能够被用来做实时追踪。
-   SC-Track采用Python实现，其扩展性强，安装方便快捷，依赖性低。



### How to use SC-Track?

```
要使用SC-Track，首先请按照Installation步骤安装。其使用过程无需过多设置，当您只有单通道分割结果时，我们要求您的分割结果必须是2D+t形式的mask灰度图文件，tiff格式，每个mask中的细胞需要保证其像素值是唯一的；或者是JSON注释文件。具体格式可参照我们的example。
当分割结果为mask时，运行 sctrack -i image.tif -a mask.tif即可。
当分割结果为annotation json file时，运行 sctrack -i image.tif -a annotation.json即可。
其中image.tif为原始图像，mask.tif，annotation.json为注释文件，原始图像可以不提供，但是不提供原始图像则无法输出可视化结果。
```



### Installation

```
Requirement: Python >= 3.7

Windows: pip isntall SC-Track
Linux/Macos: pip3 isntall SC-Track
```







### Usage

```python
我们提供了命令行工具，只需要在命令行中运行sctrack工具即可，要自动化批处理大量文件，请参阅我们的源码文档。
其基本用法是：
from SCTrack import strat_track

image = 'path/to/image.tif'

# using mask annotation
annotation_mask = '/path/to/annotation.tif'
start_track(fannotation=annotation_mask, fimage=image)

# using json file annotation
annotation_json = '/path/to/annotation.json'
start_track(fannotation=annotation_json, fimage=image)
```

