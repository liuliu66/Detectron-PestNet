#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import io
import sys
import json
from PIL import Image

def add_annotations(annotID, labelID, imageID, area=0, bbox=[], seg=[]):
    return {
        u"area": area,
        u"id": annotID,
        u"category_id": labelID,
        u"ignore": 0,
        u"segmentation": seg,
        u"image_id": imageID,
        u"bbox":  bbox,
        u"iscrowd": 0
    }


def add_image(imageID, fileName, Width, Height):
    return {
        u"id": imageID,
        u"file_name": fileName,
        u"width": Width,
        u"height": Height
    }
batch = 'val'
label_txt = batch+'.txt'
image_path = batch
json_output = batch+'.json'
json_data = {
    u"categories": [],
    u"annotations": [],
    u"images": [],
    u"type": "classification"}
annotID = 0
imageID = 0
categories_list = []
for line in open(label_txt, 'r').readlines():
    label = line.split(' ')[1][0:-1]
    image_file = line.split(' ')[0].split('/')[-1]
    json_data['annotations'].append(
        add_annotations(annotID, int(label), imageID, area=1)
    )
    img = Image.open(os.path.join(image_path, image_file))
    json_data['images'].append(
        add_image(imageID, image_file, img.size[0], img.size[1])
    )
    if label in categories_list:
        pass
    else:
        categories_list.append(label)
        json_data['categories'].append({u'supercategory': None, u'id': int(label), u'name': label})
    annotID += 1
    imageID += 1
    sys.stdout.write("\r\x1b[K" +image_file)
    sys.stdout.flush()
with io.open(json_output, 'w+', encoding='utf-8') as f:
    f.write(json.dumps(json_data, ensure_ascii=False))
print('\n')
print('num of labels is', len(categories_list))





