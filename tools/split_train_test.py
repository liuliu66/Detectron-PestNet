# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:31:31 2018

@author: LiuLiu
"""

"""split anntations files to train set and test set by 9:1 ratio"""

import os
import numpy as np
Annotations_path = 'Annotations/'
Annotations_ext = 'xml'
file_num = 0
file_names = []
for path,d,filelist in os.walk(Annotations_path):
    for filename in filelist:
        if filename.endswith(Annotations_ext):
            #print(filename[0:-4])
            file_names.append(filename[0:-4])
            file_num += 1
file_names.sort()
test_index = []
train_index=list(range(file_num))
for i in range(int(file_num*0.1)):
    randomIndex=int(np.random.uniform(0,len(train_index)))
    test_index.append(train_index[randomIndex])
    del train_index[randomIndex]
test_index = sorted(test_index)
print(len(test_index))
print(len(train_index))
with open("test.txt","w") as f:
    for i in test_index:
        f.write(file_names[i]+'\n')
f.close()
with open("trainval.txt","w") as f:
    for i in train_index:
        f.write(file_names[i]+'\n')
f.close()