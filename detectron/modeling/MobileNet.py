from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg

def add_MobileNet_conv5_body(model):
    # stem
    model.Conv('data', 'conv1', 3, 32, 3, pad=1, stride=2, no_bias=1)
    model.AffineChannel('conv1', 'conv1', dim=32, inplace=True)
    model.Relu('conv1', 'conv1')

    # block 2
    for lvl in range(1, 2+1):
        blob_in = 'conv2_'+str(lvl-1)+'_sep'
        if lvl == 1:
            blob_in = 'conv1'
        dim_in = 32 * lvl
        dim_out = 64 * lvl
        group = 32 * lvl
        slvl = str(lvl)
        model.Conv(blob_in, 'conv2_'+slvl+'_dw', dim_in, dim_in, 3, pad=1, stride=1, group=group, no_bias=1)
        model.AffineChannel('conv2_'+slvl+'_dw', 'conv2_'+slvl+'_dw', dim=dim_in, inplace=True)
        model.Relu('conv2_'+slvl+'_dw', 'conv2_'+slvl+'_dw')
        model.Conv('conv2_'+slvl+'_dw', 'conv2_'+slvl+'_sep', dim_in, dim_out, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('conv2_'+slvl+'_sep', 'conv2_'+slvl+'_sep', dim=dim_out, inplace=True)
        model.Relu('conv2_'+slvl+'_sep', 'conv2_'+slvl+'_sep')

    # block 3
    model.Conv('conv2_2_sep', 'conv3_1_dw', 128, 128, 3, pad=1, stride=1, group=128, no_bias=1)
    model.AffineChannel('conv3_1_dw', 'conv3_1_dw', dim=128, inplace=True)
    model.Relu('conv3_1_dw', 'conv3_1_dw')
    model.Conv('conv3_1_dw', 'conv3_1_sep', 128, 128, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv3_1_sep', 'conv3_1_sep', dim=128, inplace=True)
    model.Relu('conv3_1_sep', 'conv3_1_sep')

    model.Conv('conv3_1_sep', 'conv3_2_dw', 128, 128, 3, pad=1, stride=2, group=128, no_bias=1)
    model.AffineChannel('conv3_2_dw', 'conv3_2_dw', dim=128, inplace=True)
    model.Relu('conv3_2_dw', 'conv3_2_dw')
    model.Conv('conv3_2_dw', 'conv3_2_sep', 128, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv3_2_sep', 'conv3_2_sep', dim=256, inplace=True)
    model.Relu('conv3_2_sep', 'conv3_2_sep')

    # block 4
    model.Conv('conv3_2_sep', 'conv4_1_dw', 256, 256, 3, pad=1, stride=1, group=256, no_bias=1)
    model.AffineChannel('conv4_1_dw', 'conv4_1_dw', dim=256, inplace=True)
    model.Relu('conv4_1_dw', 'conv4_1_dw')
    model.Conv('conv4_1_dw', 'conv4_1_sep', 256, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv4_1_sep', 'conv4_1_sep', dim=256, inplace=True)
    model.Relu('conv4_1_sep', 'conv4_1_sep')

    model.Conv('conv4_1_sep', 'conv4_2_dw', 256, 256, 3, pad=1, stride=2, group=256, no_bias=1)
    model.AffineChannel('conv4_2_dw', 'conv4_2_dw', dim=256, inplace=True)
    model.Relu('conv4_2_dw', 'conv4_2_dw')
    model.Conv('conv4_2_dw', 'conv4_2_sep', 256, 512, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv4_2_sep', 'conv4_2_sep', dim=512, inplace=True)
    model.Relu('conv4_2_sep', 'conv4_2_sep')

    # block 5
    for lvl in range(1, 6+1):
        blob_in = 'conv5_'+str(lvl-1)+'_sep'
        if lvl == 1:
            blob_in = 'conv4_2_sep'
        dim_in = 512
        dim_out = 512 if lvl < 6 else 1024
        group = dim_in
        slvl = str(lvl)
        model.Conv(blob_in, 'conv5_'+slvl+'_dw', dim_in, dim_in, 3, pad=1, stride=1, group=group, no_bias=1)
        model.AffineChannel('conv5_'+slvl+'_dw', 'conv5_'+slvl+'_dw', dim=dim_in, inplace=True)
        model.Relu('conv5_'+slvl+'_dw', 'conv5_'+slvl+'_dw')
        model.Conv('conv5_'+slvl+'_dw', 'conv5_'+slvl+'_sep', dim_in, dim_out, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('conv5_'+slvl+'_sep', 'conv5_'+slvl+'_sep', dim=dim_out, inplace=True)
        model.Relu('conv5_'+slvl+'_sep', 'conv5_'+slvl+'_sep')

    # block 6
    model.Conv('conv5_6_sep', 'conv6_dw', 1024, 1024, 3, pad=1, stride=1, group=1024, no_bias=1)
    model.AffineChannel('conv6_dw', 'conv6_dw', dim=1024)
    model.Relu('conv6_dw', 'conv6_dw')
    model.Conv('conv6_dw', 'conv6_sep', 1024, 1024, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv6_sep', 'conv6_sep', dim=1024)
    blob_out = model.Relu('conv6_sep', 'conv6_sep')
 
    return blob_out, 1024, 1. / 8.












