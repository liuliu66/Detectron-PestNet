# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', '11111', '11121', '11122', '11123', '11131', '1'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_voc_dataset():
    ds = AttrDict()
    classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_classification_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__'
    ]
    for i in range(61):
        classes.append(str(i))
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_cifar100_dataset():
    """A dummy cifar100 dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [u'__background__', u'rocket', u'camel', u'crocodile',
               u'motorcycle', u'keyboard', u'chair', u'seal', u'sunflower',
               u'cup', u'rose', u'orange', u'porcupine', u'plate',
               u'lawn_mower', u'bear', u'caterpillar', u'snake',
               u'sweet_pepper', u'dinosaur', u'poppy', u'willow_tree',
               u'aquarium_fish', u'turtle', u'bicycle', u'house', u'spider',
               u'lion', u'lobster', u'sea', u'cattle', u'girl', u'orchid',
               u'clock', u'fox', u'skyscraper', u'trout', u'pear', u'kangaroo',
               u'cockroach', u'shrew', u'boy', u'wolf', u'hamster', u'raccoon',
               u'castle', u'road', u'apple', u'table', u'cloud', u'streetcar',
               u'crab', u'dolphin', u'squirrel', u'oak_tree', u'bus',
               u'chimpanzee', u'tiger', u'train', u'rabbit', u'baby',
               u'otter', u'television', u'tank', u'palm_tree', u'plain',
               u'pine_tree', u'worm', u'bed', u'bee', u'wardrobe', u'lizard',
               u'can', u'maple_tree', u'tractor', u'pickup_truck', u'bridge',
               u'shark', u'beetle', u'telephone', u'woman', u'beaver', u'mouse',
               u'ray', u'mountain', u'mushroom', u'bowl', u'couch', u'lamp',
               u'forest', u'elephant', u'butterfly', u'snail', u'leopard',
               u'possum', u'whale', u'man', u'flatfish', u'tulip', u'bottle',
               u'skunk']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds