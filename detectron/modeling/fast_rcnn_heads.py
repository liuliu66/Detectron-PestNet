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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.ops.weighted_loss import *


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        if not cfg.MODEL.WEIGHTED_LOSS:
            model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
        else:
            model.Softmax('cls_score', 'cls_prob1', engine='CUDNN')
            model.net.Sigmoid('cls_score', 'cls_prob2', engine='CUDNN')
            model.net.Mean(['cls_prob1', 'cls_prob2'], 'cls_prob')
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        model.num_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])

def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    loss_scalar = 1.0
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_LOSS:
        loss_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale() * loss_scalar
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale() * loss_scalar
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients

def add_fast_rcnn_weighted_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    cls_prob1, loss_cls1 = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob1', 'loss_cls1'],
        scale=model.GetLossScale()
    )
    cls_prob2 = model.net.Sigmoid('cls_score', 'cls_prob2')
    model.LabelCrossEntropy(['cls_prob2', 'labels_int32'], 'sigmoid_cross_entropy')
    loss_cls2 = model.AveragedLoss('sigmoid_cross_entropy', 'loss_cls2')
    loss_cls = model.net.Python(weighted_loss_forward,weighted_loss_backward)(['loss_cls1', 'weight_cls1', 'loss_cls2', 'weight_cls2'], 'loss_cls')
    cls_prob = model.Mean(['cls_prob1', 'cls_prob2'], 'cls_prob')

    loss_bbox1 = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox1',
        scale=model.GetLossScale()
    )
    loss_bbox2 = model.net.MSELoss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox2',
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.Python(weighted_loss_forward,weighted_loss_backward)(['loss_bbox1', 'weight_bbox1', 'loss_bbox2', 'weight_bbox2'], 'loss_bbox')
    #loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls1, loss_cls2, loss_bbox1, loss_bbox2])
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    #model.Accuracy(['cls_prob2', 'labels_int32'], 'accuracy_cls2')
    #model.AddLosses(['loss_cls1', 'loss_bbox1'])
    #model.AddLosses(['loss_cls2', 'loss_bbox2'])
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics(['accuracy_cls', 'weight_cls1', 'weight_cls2', 'weight_bbox1', 'weight_bbox2'])
    #model.AddMetrics('accuracy_cls2')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    # normalize the gradient by the number of cascade heads
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        for idx in range(-2, 0):
            model.stage_params['1'].append(model.weights[idx])
            model.stage_params['1'].append(model.biases[idx])
    return 'fc7', hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    # normalize the gradient by the number of cascade heads
    if cfg.MODEL.CASCADE_ON and cfg.CASCADE_RCNN.SCALE_GRAD:
        grad_scalar = cfg.CASCADE_RCNN.STAGE_WEIGHTS[0]
        model.net.Scale(
            roi_feat, roi_feat, scale=1.0, scale_grad=grad_scalar
        )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    if cfg.MODEL.CASCADE_ON:
        # add stage parameters to list
        if '1' not in model.stage_params:
            model.stage_params['1'] = []
        num_params = 2 * num_convs + 1
        for idx in range(-num_params, 0):
            model.stage_params['1'].append(model.weights[idx])
        # head convs don't have bias
        num_params = num_convs + 1
        for idx in range(-num_params, 0):
            model.stage_params['1'].append(model.biases[idx])
    return 'fc6', fc_dim
