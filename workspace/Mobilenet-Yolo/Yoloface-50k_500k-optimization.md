  
# 一、yoloface-500k-v2 


## 1.1. env setup

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











##1.3、断点续训

 ./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yolov3_tiny_quant_channelwise-train.cfg \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yoloface-500k-v2/voc-20210105/cfg/yoloface-500k-v2.conv.15





./darknet detector train cfg/voc.data cfg/yoloface-500k-v2.cfg backup/yoloface-500k-v2.backup




## 1.2、quantized 


./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2_quantized.cfg 


./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2_quantized.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/backup/yoloface-500k-v2_last.weights

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights \
/home/jcq/chunxia.jpeg -thresh 0.1



## 1.3、channelwise 


./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2-channelwise.cfg 


./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2-quantized.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/backup/yoloface-500k-v2_last.weights

./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights \
/home/jcq/chunxia.jpeg -thresh 0.1






  
# 二、yoloface-50k


## 2.1. env setup

cd /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0




./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k_SPP.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.weights 


 不迁移学习

./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k_SPP.cfg 



./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k_SPP.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yoloface-50k_SPP/yoloface-50k_SPP_last.weights





断点续训

 ./darknet detector train /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/voc.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yolov3_tiny_quant_channelwise-train.cfg \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yoloface-500k-v2/voc-20210105/cfg/yoloface-500k-v2.conv.15





./darknet detector train cfg/voc.data cfg/yoloface-50k_SPP.cfg backup/yoloface-500k-v2.backup











## 2.2、测试图像


测试数据：
./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k_SPP.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/yoloface-50k_SPP/yoloface-50k_SPP_last.weights \
/home/jcq/chunxia.jpeg -thresh 0.5






./darknet detector  test \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.weights \
/home/jcq/chunxia.jpeg -thresh 0.5



 Warning: width=56 and height=56 in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! 

Loading weights from /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-50k/yoloface-50k.weights...
 seen 64, trained: 504 K-images (7 Kilo-batches_64) 
Done! Loaded 34 layers from weights-file 
/home/jcq/chunxia.jpeg: Predicted in 0.522000 milli-seconds.
Face: 80%
 结果



