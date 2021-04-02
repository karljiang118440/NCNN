#voc_label.py


"""
 @Usage: generate custom voc-format-dataset labels, convert .xml to .txt for each image
 @author: sun qian
 @date: 2019/9/25
 @note: dataset file structure must be modified as:
 --VOCdevkit
   --VOC2012
     --Annotations
     --ImageSets
        --Main (include train.txt, test.txt, val.txt)
     --JPEGImages
     --labels
 @ merge val and test: Run command: type 2012_test.txt 2012_val.txt  > test.txt
"""
import xml.etree.ElementTree as ET
import os
from os import getcwd

# file list - train.txt, test.txt, val.txt
sets = [('2012', 'train'), ('2012', 'val')]

# class name
classes = ["face"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt' % (year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == '__main__':
    wd = getcwd()
    for year, image_set in sets:
        if not os.path.exists('VOCdevkit/VOC%s/labels/' % (year)):
            os.makedirs('VOCdevkit/VOC%s/labels/' % (year))
        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            line = '%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' % (wd, year, image_id)
            list_file.write(line.replace("\\", '/'))
            convert_annotation(year, image_id)
        list_file.close()
