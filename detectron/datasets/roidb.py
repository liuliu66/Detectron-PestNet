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

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from past.builtins import basestring
import logging
import pprint
import numpy as np

from detectron.core.config import cfg
from detectron.datasets.json_dataset import JsonDataset
import detectron.utils.boxes as box_utils
import detectron.utils.keypoints as keypoint_utils
import detectron.utils.segms as segm_utils

logger = logging.getLogger(__name__)


def combined_roidb_for_training(dataset_names, proposal_files):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(dataset_name, proposal_file):
        ds = JsonDataset(dataset_name)
        roidb = ds.get_roidb(
            gt=True,
            proposal_file=proposal_file,
            crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH
        )
        if cfg.TRAIN.USE_FLIPPED:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb, ds)
        if cfg.TRAIN.USE_CROPPED:
            logger.info('Appending cropped training examples...')
            extend_with_cropped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidb

    if isinstance(dataset_names, basestring):
        dataset_names = (dataset_names, )
    if isinstance(proposal_files, basestring):
        proposal_files = (proposal_files, )
    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    roidbs = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    if not cfg.MODEL.CLASSIFICATION:
        roidb = filter_for_training(roidb)

        logger.info('Computing bounding-box regression targets...')
        add_bbox_regression_targets(roidb)
        logger.info('done')

    _compute_and_log_stats(roidb)

    return roidb


def extend_with_flipped_entries(roidb, dataset):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        flipped_entry = {}
        if not cfg.MODEL.CLASSIFICATION:
            boxes = entry['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 2] = width - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            flipped_entry['boxes'] = boxes
            flipped_entry['segms'] = segm_utils.flip_segms(
                entry['segms'], entry['height'], entry['width']
            )
            if dataset.keypoints is not None:
                flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
                    dataset.keypoints, dataset.keypoint_flip_map,
                    entry['gt_keypoints'], entry['width']
                )
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)

def extend_with_cropped_entries(roidb, dataset):
    """crop each entry in the given roidb by 2x2 or 4x4 grid and return a new roidb that is the
    concatenation of the original roidb and the cropped entries.

    "cropping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are cropped by defined grid.
    """
    crop_size = cfg.TRAIN.CROPPED_SIZE
    cropped_roidb = []
    id = roidb[-1]['id'] + 1
    for i in range(crop_size ** 2):
        for entry in roidb:
            width = entry['width']
            height = entry['height']
            cropped_entry = {}
            if not cfg.MODEL.CLASSIFICATION:
                new_image = [0] * 4
                new_image[0] = width / crop_size * int(round(i/2))
                new_image[1] = height / crop_size * int(round(i/2))
                new_image[2] = width / crop_size * int(round(i/2)+1)
                new_image[3] = height / crop_size * int(round(i/2)+1)
                boxes = entry['boxes'].copy()
                new_boxes = []
                indexes = []
                for index,box in enumerate(boxes):
                    if _box_in_cropped_image(box, new_image):
                        box[0] -= new_image[0]
                        box[1] -= new_image[1]
                        box[2] -= new_image[0]
                        box[3] -= new_image[1]
                        assert (box > 0).all()
                        new_boxes.append(box)
                        indexes.append(index)

                cropped_entry['boxes'] = np.array(new_boxes)
                cropped_entry['width'] = width / crop_size
                cropped_entry['height'] = height / crop_size
                cropped_entry['gt_classes'] = entry['gt_classes'][indexes]
                cropped_entry['max_classes'] = entry['max_classes'][indexes]
                cropped_entry['seg_areas'] = entry['seg_areas'][indexes]
                cropped_entry['is_crowd'] = entry['is_crowd'][indexes]
                cropped_entry['max_overlaps'] = entry['max_overlaps'][indexes]
                cropped_entry['box_to_gt_ind_map'] = np.array(range(len(indexes))).astype(np.int32)
                cropped_entry['gt_overlaps'] = entry['gt_overlaps'][indexes]
                cropped_entry['id'] = id
                id += 1
                #cropped_entry['segms'] = segm_utils.crop_segms(
                    #entry['segms'], entry['height'], entry['width']
                #)
                if dataset.keypoints is not None:
                    cropped_entry['gt_keypoints'] = keypoint_utils.crop_keypoints(
                        dataset.keypoints, dataset.keypoint_crop_map,
                        entry['gt_keypoints'], entry['width']
                    )
            dont_copy = ('boxes', 'segms', 'gt_keypoints', 'cropped', 'width', 
                         'height', 'gt_classes', 'is_crowd', 'max_overlaps', 'box_to_gt_ind_map', 
                         'gt_overlaps', 'id', 'seg_areas', 'max_classes')
            for k, v in entry.items():
                if k not in dont_copy:
                    cropped_entry[k] = v
            cropped_entry['cropped'] = [0] * (crop_size ** 2)
            cropped_entry['cropped'][i] = 1
            if len(cropped_entry['boxes']) > 0:
                cropped_roidb.append(cropped_entry)
    roidb.extend(cropped_roidb)


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if cfg.MODEL.KEYPOINTS_ON:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))

def _box_in_cropped_image(box, image):
    point_min = box[0] > image[0] and box[1] > image[1]
    point_max = box[2] < image[2] and box[2] < image[3]
    return point_min and point_max
