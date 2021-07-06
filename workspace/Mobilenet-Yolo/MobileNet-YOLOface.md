  
# 1. yoloface-500k-v2 


## 1.1. 

cd /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0



./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights 


 不迁移学习

./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg 



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/backup/yoloface-500k-v2_last.weights







./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights \
/home/jcq/chunxia.jpeg -thresh 0.1
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.1








## 1.2、、测试图像


./darknet detector  test  data/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yolov3_tiny_quant_channelwise-test.cfg \
/media/jcq/Soft/Pytorch/YOLO/YOLO_quantiztion/yolo_quantization/backup/yolov3_tiny_quant_channelwise-train_100000.weights \
 data/dog.jpg  -thresh 0.55





##1.3、断点续训

 ./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yolov3_tiny_quant_channelwise-train.cfg \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yoloface-500k-v2/voc-20210105/cfg/yoloface-500k-v2.conv.15





./darknet detector train cfg/voc.data cfg/yoloface-500k-v2.cfg backup/yoloface-500k-v2.backup





# 二 、yolov3  进行训练


./darknet partial ./Yolo-Fastest/COCO/yolo-fastest.cfg ./Yolo-Fastest/COCO/yolo-fastest.weights ./Yolo-Fastest/COCO/yolo-fastest.conv.109 109




./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3/yolov3-face.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/darknet53.conv.74


./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3/yolov3-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3/darknet53.conv.74






1) 标准模型

/media/jcq/Soft/NCNN/download/ncnn-master-20210531/build-host-gcc-linux/tools/darknet/darknet2ncnn \
/media/jcq/BACKUP/Darknet/darknet/cfg/yolov3.cfg \
/media/jcq/BACKUP/Darknet/darknet/yolov3.weights \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/zoo/yolov3.param  \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/zoo/yolov3.bin 






# 3 、yolov3-tiny  进行训练


## 3.1 、env

cd /media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0



 > 修改环境为：


## 3.2 . 训练权重

./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15





./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg 



## 3.3  . test

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.55



./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.55
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
   13 conv    256  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 256  0.089 BFLOPs
   14 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
   15 conv     18  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x  18  0.003 BFLOPs
   16 yolo
   17 route  13
   18 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128  0.011 BFLOPs
   19 upsample            2x    13 x  13 x 128   ->    26 x  26 x 128
   20 route  19 8
   21 conv    256  3 x 3 / 1    26 x  26 x 384   ->    26 x  26 x 256  1.196 BFLOPs
   22 conv     18  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x  18  0.006 BFLOPs
   23 yolo
Loading weights from /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights...Done!
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png: Predicted in 0.003093 seconds.
Face: 98%
Face: 90%
Face: 70%



## 3.4、 模型转换


1) 标准模型

/media/jcq/Soft/NCNN/download/ncnn-master-20210531/build-host-gcc-linux/tools/darknet/darknet2ncnn \
/media/jcq/BACKUP/Darknet/darknet/cfg/yolov3-tiny.cfg \
/media/jcq/BACKUP/Darknet/toos/darknet2onnx/weights/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/zoo/yolov3-tiny.param  \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/zoo/yolov3-tiny.bin 

2) yolov3-tiny-face 模型

/media/jcq/Soft/NCNN/download/ncnn-master-20210531/build-host-gcc-linux/tools/darknet/darknet2ncnn \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/face/yolov3-tiny.param  \
/media/jcq/Soft/NCNN/download/ncnn-master-20210531/tools/darknet/example/face/yolov3-tiny.bin 

3) 转化为 ncnn 模型

在线进行转换

https://convertmodel.com/







# 四 、yolo  进行训练


## 3.1 、env

cd /media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0


## 3.2 . 训练权重

./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15




./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15




./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg 


## 3.3  . test

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.55



# 4. yoloface-500k-v1


## 4.1. env

cd /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0



./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v1/yoloface-500k.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v1/yoloface-500k.weights 


 不迁移学习

./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v1/yoloface-500k.cfg \ 
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v1/yoloface-500k.weights 




./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights \
/home/jcq/chunxia.jpeg -thresh 0.1
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.1








## 1.2、、测试图像


./darknet detector  test  data/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yolov3_tiny_quant_channelwise-test.cfg \
/media/jcq/Soft/Pytorch/YOLO/YOLO_quantiztion/yolo_quantization/backup/yolov3_tiny_quant_channelwise-train_100000.weights \
 data/dog.jpg  -thresh 0.55





## 1.3、断点续训

 ./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yolov3_tiny_quant_channelwise-train.cfg \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yoloface-500k-v2/voc-20210105/cfg/yoloface-500k-v2.conv.15





./darknet detector train cfg/voc.data cfg/yoloface-500k-v2.cfg backup/yoloface-500k-v2.backup



# 5. yoloface-50k


## 5.1 

cd /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.cfg 





./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/backup/yoloface-50k_last.weights







./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/backup/yoloface-50k_last.weights \
/home/jcq/chunxia.jpeg -thresh 0.5
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.5



标准模型：

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.weights \
/home/jcq/chunxia.jpeg -thresh 0.5
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/data/predictions_2.png  -thresh 0.5





# 6.yolov3-tiny-face_HW56


## 6.1.



 cd /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0



./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_HW56.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15


./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_HW56.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15




./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_HW56.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_HW56.cfg 

修改完成后可以运行，就是不知道具体效果如何。



./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_HW56.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup_HW56/yolov3-tiny-face_HW56_221000.weights \
/home/jcq/chunxia.jpeg -thresh 0.1





/home/jcq/chunxia.jpeg: Predicted in 7.975000 milli-seconds.


/home/jcq/1.jpeg -thresh 0.7




# 7.yolov3-tiny-face_20210706



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_20210706.cfg 



## 7.2、 test

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_20210706.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yolov3-tiny-face_20210706/yolov3-tiny-face_20210706_last.weights \
/home/jcq/chunxia.jpeg -thresh 0.5



./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face_20210706.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yolov3-tiny-face_20210706/yolov3-tiny-face_20210706_last.weights \
/home/jcq/chunxia.jpeg -thresh 0.5


(√)  thresh 0.1

[yolo] params: iou loss: mse (2), iou_norm: 0.75, cls_norm: 1.00, scale_x_y: 1.00
Total BFLOPS 1.705 
avg_outputs = 166662 
 Allocate additional workspace_size = 13.11 MB 
Loading weights from /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yolov3-tiny-face_20210706/yolov3-tiny-face_20210706_last.weights...
 seen 64, trained: 704 K-images (11 Kilo-batches_64) 
Done! Loaded 24 layers from weights-file 
/home/jcq/chunxia.jpeg: Predicted in 1.734000 milli-seconds.
Face: 13%
Face: 10%
Face: 10%
Face: 10%
Face: 20%
Face: 11%
Face: 10%
Face: 15%
Face: 79%



Total BFLOPS 1.705 
avg_outputs = 166662 
 Allocate additional workspace_size = 13.11 MB 
Loading weights from /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yolov3-tiny-face_20210706/yolov3-tiny-face_20210706_last.weights...
 seen 64, trained: 704 K-images (11 Kilo-batches_64) 
Done! Loaded 24 layers from weights-file 
/home/jcq/chunxia.jpeg: Predicted in 1.780000 milli-seconds.
Face: 79%

 > ：这个明显效果更加突出，检测率更高，非常满意实际上





./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights \
/home/jcq/chunxia.jpeg -thresh 0.5

[yolo] params: iou loss: mse (2), iou_norm: 0.75, cls_norm: 1.00, scale_x_y: 1.00
Total BFLOPS 5.448 
avg_outputs = 324846 
 Allocate additional workspace_size = 52.43 MB 
Loading weights from /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/backup/yolov3-tiny-face_310000.weights...
 seen 64, trained: 1240 K-images (19 Kilo-batches_64) 
Done! Loaded 24 layers from weights-file 
/home/jcq/chunxia.jpeg: Predicted in 2.810000 milli-seconds.
Face: 58%





