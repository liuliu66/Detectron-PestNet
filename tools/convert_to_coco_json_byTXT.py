#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Author: hbchen
# @Time: 2018-01-29
# @Description: 

import os, sys, json

from xml.etree.ElementTree import ElementTree, Element
imageset = 'trainval'

TXT_PATH = 'ImageSets/Main/'+imageset+'.txt'
XML_PATH = 'Annotations'
JSON_PATH = imageset+'.json'
json_obj = {}
images = []
annotations = []
categories = []
categories_list = []
image_id = 0
annotation_id = 0
catID = 0

classes = ['12111','12122']

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def if_match(node, kv_map):
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True

def get_node_by_keyvalue(nodelist, kv_map):
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes

def find_nodes(tree, path):
    return tree.findall(path)

print "-----------------Start------------------"
xml_names = []
sum = 0
f = open(TXT_PATH)
lines = f.readlines()
for line in lines:
    line = line.strip("\n") + ".xml"
    #print line
    xml_names.append(line)
    sum = sum + 1
#print "xml",sum
f.close()

for xml in xml_names:
    tree = read_xml(XML_PATH + "/" + xml)
    object_nodes = get_node_by_keyvalue(find_nodes(tree, "object"), {})
    if len(object_nodes) == 0:
        image = {}
        file_name = os.path.splitext(xml)[0]; 
        image["file_name"] = file_name + ".jpg"
        width_nodes = get_node_by_keyvalue(find_nodes(tree, "size/width"), {})
        image["width"] = int(width_nodes[0].text)
        height_nodes = get_node_by_keyvalue(find_nodes(tree, "size/height"), {})
        image["height"] = int(height_nodes[0].text)
        image["id"] = image_id
        images.append(image)
        print xml, "no object"

    else:
        image = {}
        file_name = os.path.splitext(xml)[0]; 
        image["file_name"] = file_name + ".jpg"
        width_nodes = get_node_by_keyvalue(find_nodes(tree, "size/width"), {})
        image["width"] = int(width_nodes[0].text)
        height_nodes = get_node_by_keyvalue(find_nodes(tree, "size/height"), {})
        image["height"] = int(height_nodes[0].text)
        image["id"] = image_id
        images.append(image) 

        name_nodes = get_node_by_keyvalue(find_nodes(tree, "object/name"), {})
        xmin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmin"), {})
        ymin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymin"), {})
        xmax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmax"), {})
        ymax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymax"), {})
       # print ymax_nodes
        for index, node in enumerate(object_nodes):
            if name_nodes[index].text in classes:
                annotation = {}
                segmentation = []
                bbox = []
                seg_coordinate = [] 
                seg_coordinate.append(int(xmin_nodes[index].text))
                seg_coordinate.append(int(ymin_nodes[index].text))
                seg_coordinate.append(int(xmin_nodes[index].text))
                seg_coordinate.append(int(ymax_nodes[index].text))
                seg_coordinate.append(int(xmax_nodes[index].text))
                seg_coordinate.append(int(ymax_nodes[index].text))
                seg_coordinate.append(int(xmax_nodes[index].text))
                seg_coordinate.append(int(ymin_nodes[index].text))
                segmentation.append(seg_coordinate)
                width = int(xmax_nodes[index].text) - int(xmin_nodes[index].text)
                height = int(ymax_nodes[index].text) - int(ymin_nodes[index].text)
                area = width * height
                bbox.append(int(xmin_nodes[index].text))
                bbox.append(int(ymin_nodes[index].text))
                bbox.append(width)
                bbox.append(height)

                annotation["segmentation"] = segmentation
                annotation["area"] = area
                annotation["iscrowd"] = 0
                annotation["image_id"] = image_id
                annotation["bbox"] = bbox
                annotation["category_id"] = int(name_nodes[index].text)
                annotation["id"] = annotation_id
                annotation_id += 1
                annotation["ignore"] = 0
                annotations.append(annotation)

            #if int(name_nodes[index].text) in categories_list:
                #pass
            #else:
                #categories_list.append(int(name_nodes[index].text))
                #categorie = {}
                #categorie["supercategory"] = None
                #categorie["id"] = catID
                #categorie["id"] = int(name_nodes[index].text)
                #categorie["name"] = name_nodes[index].text
                #categories.append(categorie)
                #catID += 1
    image_id += 1
    print "processing " + xml


for i in classes:
    categorie = {}
    categorie["supercategory"] = None
    categorie["id"] = int(i)
    categorie["name"] = i
    categories.append(categorie)


json_obj["images"] = images
json_obj["type"] = "instances"
json_obj["annotations"] = annotations
json_obj["categories"] = categories

f = open(JSON_PATH, "w")
#json.dump(json_obj, f)
json_str = json.dumps(json_obj)
f.write(json_str)

print "------------------End-------------------"
