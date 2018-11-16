# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:31:31 2018

@author: 刘浏
"""

import json
import math

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

def GetAreaOfPolyGon(points):

    area = 0
    if(len(points)<3):
        
         raise Exception("error")

    p1 = points[0]
    for i in range(1,len(points)-1):
        p2 = points[i]
        p3 = points[i+1]
        #print(p2.x,p2.y)
        #print(p3.x,p3.y)

        #计算向量
        #print(p2)
        vecp1p2 = Point(p2.x-p1.x,p2.y-p1.y)
        vecp2p3 = Point(p3.x-p2.x,p3.y-p2.y)


        
        #判断顺时针还是逆时针，顺时针面积为正，逆时针面积为负
        vecMult = vecp1p2.x*vecp2p3.y - vecp1p2.y*vecp2p3.x   #判断正负方向比较有意思
        sign = 0
        if(vecMult>0):
            sign = 1
        elif(vecMult<0):
            sign = -1

        triArea = GetAreaOfTriangle(p1,p2,p3)*sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1,p2,p3):
    '''计算三角形面积   海伦公式'''
    area = 0
    p1p2 = GetLineLength(p1,p2)
    p2p3 = GetLineLength(p2,p3)
    p3p1 = GetLineLength(p3,p1)
    s = (p1p2 + p2p3 + p3p1)/2
    area = s*(s-p1p2)*(s-p2p3)*(s-p3p1)   #海伦公式
    area = math.sqrt(area)
    return area

def GetLineLength(p1,p2):
    '''计算边长'''
    length = math.pow((p1.x-p2.x),2) + math.pow((p1.y-p2.y),2)  #pow  次方
    length = math.sqrt(length)   
    return length
def parse_json(in_path):
    seg = []
    #seg_x = []
    #seg_y = []
    label = []
    area = []
    bbox = {"xmin":[], "xmax":[], "ymin": [], "ymax": []}
    fileJson = json.load(open(in_path,'r'))
    for obj_index in range(len(fileJson['shapes'])):
        seg_x = []
        seg_y = []
        label.append(int(fileJson['shapes'][obj_index]['label']))
        points = fileJson['shapes'][obj_index]['points']
        seg.append([])
        points_area = []
        #seg_x.append([])
        #seg_y.append([])
        for point in points:
            seg[obj_index].append(round(point[0],2))
            seg[obj_index].append(round(point[1],2))
            points_area.append(Point(round(point[0],2), round(point[1],2)))
            seg_x.append(round(point[0],2))
            #print(seg_x)
            seg_y.append(round(point[1],2))
            #seg_x.append([])
            #seg_y.append([])
            #print(seg_y)
        #print(points_area[0])
        #print(seg_x)
        #print(seg_y)
        bbox["xmin"].append(min(seg_x)) 
        bbox["xmax"].append(max(seg_x))
        bbox["ymin"].append(min(seg_y))
        bbox["ymax"].append(max(seg_y))
        area.append(round(GetAreaOfPolyGon(points_area),2))
    #print(bbox["xmin"])
    return label, seg, area, bbox