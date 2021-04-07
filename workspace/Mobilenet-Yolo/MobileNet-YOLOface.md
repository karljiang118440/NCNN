  
#1. yoloface-500k-v2 


##1.1. 

cd /media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0




./darknet partial /media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg /media/jcq/Backup/DL_models/YOLO/yoloface-500k-v2.weights yoloface-500k-v2.conv.15 15


./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/v2/yoloface-500k-v2.weights 





./darknet detector train cfg/voc.data cfg/yoloface-500k-v2.cfg backup/yoloface-500k-v2.backup




##1.2、、测试图像


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




# 三 、yolov3-tiny  进行训练


cd /media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0




./darknet partial \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Backup/DL_models/YOLO/yolov3-tiny.weights \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15




./darknet detector train \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/face.data \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny-face.cfg \
/media/jcq/Soft/NCNN/workspace/MobileNet-Yolo/yoloface-500k/yolov3-tiny/cfg/yolov3-tiny.conv.15 15



./darknet detector  test data/voc.data \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yolov3-tiny/voc-20210105/cfg/yolov3-tiny-test.cfg \
/media/jcq/Soft/Pytorch/YOLO/Yolo-Fastest/backup/yolov3-tiny/voc-20210105/yolov3-tiny_100000.weights\
 data/dog.jpg  -thresh 0.55