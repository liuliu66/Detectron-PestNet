#!/usr/bin/env python2

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

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import numpy as np
import cPickle as pickle
from PIL import Image
from detectron.datasets.dataset_catalog import DATASETS
from detectron.datasets.dataset_catalog import DEVKIT_DIR
from detectron.datasets.dataset_catalog import ROOT_DIR
from detectron.datasets.json_dataset import JsonDataset

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
import detectron.datasets.voc_dataset_evaluator as voc_dataset_evaluator

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
#cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See lib/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    all_results = run_inference(
            cfg.TEST.WEIGHTS,
            ind_range=args.range,
            multi_gpu_testing=args.multi_gpu_testing,
            check_expected_results=False,
            evaluation = False
        )
    all_boxes = all_results['all_boxes']

    test_dataset = JsonDataset(cfg.TEST.DATASETS[0])

    image_set = test_dataset.name.split('_')[-1]
    root_path = DATASETS[test_dataset.name][ROOT_DIR]
    image_set_path = os.path.join(root_path, 'ImageSets', 'Main', image_set + '.txt')
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]

    test_roidb = test_dataset.get_roidb()
    for i,entry in enumerate(test_roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]

    # crop images based on detected boxes and store into imgs_crop
    imgs_crop = []
    for cls_ind, cls in enumerate(test_dataset.classes):
        if cls == '__background__':
            continue
        for im_ind, index in enumerate(image_index):
            dets = all_boxes[cls_ind][im_ind]
            if type(dets) == list:
                assert len(dets) == 0, \
                    'dets should be numpy.ndarray or empty list'
                continue
            
            for k in range(dets.shape[0]):
                im = Image.open(test_roidb[im_ind]['image'])
                size = im.size
                box = []
                box.append(int((max(0, dets[k, 0] - 10))))
                box.append(int((max(0, dets[k, 1] - 10))))
                box.append(int((min(im.size[0], dets[k, 2] + 10))))
                box.append(int((min(im.size[1], dets[k, 3] + 10))))
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]

                scale_ratio = size[0] / box_width
                if box_height * scale_ratio > size[1]:
                    scale_ratio = im.size[1] / box_height

                im_crop = {}
                new_size = (int(box_width * scale_ratio), int(box_height * scale_ratio))
                im_crop['img_data'] = np.array(im.crop(box).resize(new_size, Image.ANTIALIAS))
                im_crop['img_org_name'] = index
                im_crop['box'] = box
                im_crop['size'] = new_size
                imgs_crop.append(im_crop)
    print(len(imgs_crop))
    
    # test the croped images into the next models
    cfg = 'experiments/cfgs/e2e_mask_rcnn_resnet-50-FPN.yaml'
    weights = 'experiments/output/mask_rcnn_fpn_chimeibing_class1/resnet-50/train/coco_chimeibing_single_crop_class1_trainval/generalized_rcnn/model_final.pkl'
    model = infer_engine.initialize_model_from_cfg(weights)
    boxes = []
    for img in imgs_crop:
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, img['img_data'], None
            )
        boxes.append(cls_boxes)
        #logger.info('Inference time: {:.3f}s'.format(time.time() - t))

    new_boxes = []
    new_boxes.append([])
    new_boxes.append([])
    new_boxes.append([])
    bbox = []
    for img_ind,box in enumerate(boxes):
        current_img_name = imgs_crop[img_ind]['img_org_name']
        if img_ind != 0 and current_img_name != imgs_crop[img_ind-1]['img_org_name']:
            new_boxes[1].append(np.array(bbox))
            bbox = []
        for i in range(len(box[1])):
            if box[1] != []:
                org_box = imgs_crop[img_ind]['box']
                #new_boxes[1][j] = []
                box[1][i, 0] = box[1][i, 0] + org_box[0]
                box[1][i, 1] = box[1][i, 1] + org_box[1]
                box[1][i, 2] = box[1][i, 2] + org_box[0]
                box[1][i, 3] = box[1][i, 3] + org_box[1]
                bbox.append(box[1][i, :])
                #bbox = box[1][i, 0:4]
                #score = box[1][i, -1]
            #print(score)
    new_boxes[1].append(np.array(bbox))

    weights2 = 'experiments/output/mask_rcnn_fpn_chimeibing_class2/resnet-50/train/coco_chimeibing_single_crop_class2_trainval/generalized_rcnn/model_final.pkl'
    model2 = infer_engine.initialize_model_from_cfg(weights2)
    boxes = []
    for img in imgs_crop:
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model2, img['img_data'], None
            )
        boxes.append(cls_boxes)
        #logger.info('Inference time: {:.3f}s'.format(time.time() - t))

    bbox = []
    for img_ind,box in enumerate(boxes):
        current_img_name = imgs_crop[img_ind]['img_org_name']
        if img_ind != 0 and current_img_name != imgs_crop[img_ind-1]['img_org_name']:
            new_boxes[2].append(np.array(bbox))
            bbox = []
        for i in range(len(box[1])):
            if box[1] != []:
                org_box = imgs_crop[img_ind]['box']
                org_width = org_box[2] - org_box[0]
                org_height = org_box[3] - org_box[1]
                #new_boxes[2][j] = []
                size = imgs_crop[img_ind]['size']
                
                box[1][i, 0] = box[1][i, 0] * org_width / size[0] + org_box[0]
                box[1][i, 1] = box[1][i, 1] * org_height / size[1] + org_box[1]
                box[1][i, 2] = box[1][i, 2] * org_width / size[0] + org_box[0]
                box[1][i, 3] = box[1][i, 3] * org_height / size[1] + org_box[1]
                bbox.append(box[1][i, :])
    new_boxes[2].append(np.array(bbox))

    with open('new_boxes.pkl', 'w') as f:
        pickle.dump(new_boxes, f, protocol=pickle.HIGHEST_PROTOCOL)
    final_test_dataset = JsonDataset('coco_chimeibing_single_test')
    voc_eval = voc_dataset_evaluator.evaluate_boxes(
            final_test_dataset, new_boxes, output_dir='tools/'
        )
    





