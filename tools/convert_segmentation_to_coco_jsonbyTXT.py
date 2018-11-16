# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:33:15 2018

@author: 刘浏
"""

import os, json
from parse_jsonfile import parse_json
from PIL import Image

TXT_PATH = "test.txt"
JSON_PATH = "Json"
IMAGE_PATH = "JPEGImages"
JSON_OUTPUT = "record_from_json.json"
json_obj = {}
images = []
annotations = []
categories = []
categories_list = []
annotation_id = 1

print ("-----------------Start------------------")
jsonfile_names = []
sum = 0
f = open(TXT_PATH)
lines = f.readlines()
for line in lines:
    line = line.strip("\n") + ".json"
    print (line)
    jsonfile_names.append(line)
    sum = sum + 1
print ("json个数：",sum)
f.close()
for jsonfile in jsonfile_names:
    print("processing: ", jsonfile)
    label, seg, area, bbox = parse_json(os.path.join(JSON_PATH, jsonfile))
    #print(area)
    if len(label) == 0:
        print (jsonfile, "no object")
        continue
    else:
        image = {}
        image_name = os.path.splitext(jsonfile)[0];  # 文件名
        #print(image_name)
        image["file_name"] = image_name + ".jpg"
        img = Image.open(os.path.join(IMAGE_PATH, image["file_name"]))
        image["width"] = img.size[0]
        image["height"] = img.size[1]
        image["id"] = int(image_name)
        images.append(image)
        #print(images)
        #print(bbox)
    for obj_index in range(len(label)):
        annotation = {}
        annotation["segmentation"] = [seg[obj_index]]
        annotation["area"] = area[obj_index]
        annotation["iscrowd"] = 0
        annotation["image_id"] = image["id"]
        annotation["bbox"] = [bbox["xmin"][obj_index], bbox["ymin"][obj_index], bbox["xmax"][obj_index] - bbox["xmin"][obj_index], bbox["ymax"][obj_index] - bbox["ymin"][obj_index]]
        annotation["category_id"] = label[obj_index]
        annotation["id"] = annotation_id
        annotation_id += 1
        annotation["ignore"] = 0
        annotations.append(annotation)
        
        if int(label[obj_index]) in categories_list:
            pass
        else:
            categories_list.append(label[obj_index])
            categorie = {}
            categorie["supercategory"] = "none"
            categorie["id"] = int(label[obj_index])
            categorie["name"] = str(label[obj_index])
            categories.append(categorie)
    #print(categories)
    #print(annotations)
json_obj["images"] = images
json_obj["type"] = "instances"
json_obj["annotations"] = annotations
json_obj["categories"] = categories

f = open(JSON_OUTPUT, "w")
#json.dump(json_obj, f)
json_str = json.dumps(json_obj)
f.write(json_str)
print ("------------------End-------------------")
#print(images)  