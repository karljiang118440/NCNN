#!/bin/bash
make clean & make -j8
echo 123 | sudo -S scp yolo-fastest.elf root@192.168.0.168:/home/root/ncnn/yolo-fastest

