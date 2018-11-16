from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (DenseNet121, DenseNet161, ...)
# ---------------------------------------------------------------------------- #

def add_DenseNet121_conv5_body(model):
    return add_DenseNetX_conv5_body(model, DenseNet121_level_info_conv5), 1024, 1. / 32. 

def add_DenseNet161_conv5_body(model):
    return add_DenseNetX_conv5_body(model, DenseNet161_level_info_conv5), 2208, 1. / 28. 

def add_DenseNet169_conv5_body(model):
    return add_DenseNetX_conv5_body(model, DenseNet169_level_info_conv5), 1664, 1. / 28.

def add_DenseNet201_conv5_body(model):
    return add_DenseNetX_conv5_body(model, DenseNet201_level_info_conv5), 1920, 1. / 28.

# ------------------------------------------------------------------------------
# Generic DenseNet components
# ------------------------------------------------------------------------------

def add_DenseNetX_conv5_body(model, DenseNet_level_info_func):
    DenseNet_level_info = DenseNet_level_info_func()
    [block2, block3, block4, block5] = DenseNet_level_info.block_counts[:]
    [dim_stem, dim_block_1, dim_block_2] = DenseNet_level_info.dims[:]
    

    model.Conv('data', 'conv1', 3, dim_stem, 7, pad=3, stride=2, no_bias=1)
    model.AffineChannel('conv1', 'conv1', dim=dim_stem, inplace=True)
    model.Relu('conv1', 'conv1')
    model.MaxPool('conv1', 'pool1', kernel=3, pad=1, stride=2)

    
    # block 2
    for lvl in range(block2):
        blob_in = 'pool1' if (lvl == 0) else 'concat_2_'+str(lvl)
        dim_in = dim_stem + dim_block_2*lvl
        slvl = str(lvl+1)        
        model.AffineChannel(blob_in, 'conv2_'+slvl+'_x1_bn', dim=dim_in)
        model.Relu('conv2_'+slvl+'_x1_bn', 'conv2_'+slvl+'_x1_bn')
        model.Conv('conv2_'+slvl+'_x1_bn', 'conv2_'+slvl+'_x1', dim_in, dim_block_1, 1, no_bias=1)
        model.AffineChannel('conv2_'+slvl+'_x1', 'conv2_'+slvl+'_x2_bn', dim=dim_block_1)
        model.Relu('conv2_'+slvl+'_x2_bn','conv2_'+slvl+'/x2/bn')
        model.Conv('conv2_'+slvl+'_x2_bn', 'conv2_'+slvl+'_x2', dim_block_1, dim_block_2, 3, pad=1, no_bias=1)
        model.Concat([blob_in, 'conv2_'+slvl+'_x2'], 'concat_2_'+slvl)

    dim_block2_output = dim_stem + dim_block_2 * block2
    model.AffineChannel('concat_2_'+str(block2), 'conv2_blk_bn', dim=dim_block2_output)
    model.Relu('conv2_blk_bn', 'conv2_blk_bn')
    model.Conv('conv2_blk_bn', 'conv2_blk', dim_block2_output, int(dim_block2_output/2), 1, no_bias=1)
    model.AveragePool('conv2_blk', 'pool2', kernel=2, stride=2)

    # block 3
    for lvl in range(block3):
        blob_in = 'pool2' if (lvl == 0) else 'concat_3_'+str(lvl)
        dim_in = dim_block2_output / 2 + dim_block_2*lvl
        slvl = str(lvl+1)    
        model.AffineChannel(blob_in, 'conv3_'+slvl+'_x1_bn', dim=dim_in)
        model.Relu('conv3_'+slvl+'_x1_bn', 'conv3_'+slvl+'_x1_bn')
        model.Conv('conv3_'+slvl+'_x1_bn', 'conv3_'+slvl+'_x1', dim_in, dim_block_1, 1, no_bias=1)
        model.AffineChannel('conv3_'+slvl+'_x1', 'conv3_'+slvl+'_x2_bn', dim=dim_block_1)
        model.Relu('conv3_'+slvl+'_x2_bn','conv3_'+slvl+'_x2_bn')
        model.Conv('conv3_'+slvl+'_x2_bn', 'conv3_'+slvl+'_x2', dim_block_1, dim_block_2, 3, pad=1, no_bias=1)
        model.Concat([blob_in, 'conv3_'+slvl+'_x2'], 'concat_3_'+slvl)

    dim_block3_output = dim_block2_output / 2 + dim_block_2 * block3
    model.AffineChannel('concat_3_'+str(block3), 'conv3_blk_bn', dim=dim_block3_output)
    model.Relu('conv3_blk_bn', 'conv3_blk_bn')
    model.Conv('conv3_blk_bn', 'conv3_blk', dim_block3_output, int(dim_block3_output / 2), 1, no_bias=1)
    model.AveragePool('conv3_blk', 'pool3', kernel=2, stride=2)

    # block 4
    for lvl in range(block4):
        blob_in = 'pool3' if (lvl == 0) else 'concat_4_'+str(lvl)
        dim_in = dim_block3_output / 2 + dim_block_2*lvl
        slvl = str(lvl+1)
        model.AffineChannel(blob_in, 'conv4_'+slvl+'_x1_bn', dim=dim_in)
        model.Relu('conv4_'+slvl+'_x1_bn', 'conv4_'+slvl+'_x1_bn')
        model.Conv('conv4_'+slvl+'_x1_bn', 'conv4_'+slvl+'_x1', dim_in, dim_block_1, 1, no_bias=1)
        model.AffineChannel('conv4_'+slvl+'_x1', 'conv4_'+slvl+'_x2_bn', dim=dim_block_1)
        model.Relu('conv4_'+slvl+'_x2_bn','conv4_'+slvl+'_x2_bn')
        model.Conv('conv4_'+slvl+'_x2_bn', 'conv4_'+slvl+'_x2', dim_block_1, dim_block_2, 3, pad=1, no_bias=1)
        model.Concat([blob_in, 'conv4_'+slvl+'_x2'], 'concat_4_'+slvl)

    dim_block4_output = dim_block3_output / 2 + dim_block_2 * block4
    model.AffineChannel('concat_4_'+str(block4), 'conv4_blk_bn', dim=dim_block4_output)
    model.Relu('conv4_blk_bn', 'conv4_blk_bn')
    model.Conv('conv4_blk_bn', 'conv4_blk', dim_block4_output, int(dim_block4_output / 2), 1, no_bias=1)
    model.AveragePool('conv4_blk', 'pool4', kernel=2, stride=2)

    # block 5
    for lvl in range(block5):
        blob_in = 'pool4' if (lvl == 0) else 'concat_5_'+str(lvl)
        dim_in = dim_block4_output / 2 + dim_block_2*lvl
        slvl = str(lvl+1)
        model.AffineChannel(blob_in, 'conv5_'+slvl+'_x1_bn', dim=dim_in)
        model.Relu('conv5_'+slvl+'_x1_bn', 'conv5_'+slvl+'_x1_bn')
        model.Conv('conv5_'+slvl+'_x1_bn', 'conv5_'+slvl+'_x1', dim_in, dim_block_1, 1, no_bias=1, inplace=True)
        model.AffineChannel('conv5_'+slvl+'_x1', 'conv5_'+slvl+'_x2_bn', dim=128)
        model.Relu('conv5_'+slvl+'_x2_bn','conv5_'+slvl+'_x2_bn')
        model.Conv('conv5_'+slvl+'_x2_bn', 'conv5_'+slvl+'_x2', dim_block_1, dim_block_2, 3, pad=1, no_bias=1)
        model.Concat([blob_in, 'conv5_'+slvl+'_x2'], 'concat_5_'+slvl)

    dim_block5_output = dim_block4_output / 2 + dim_block_2 * block5
    model.AffineChannel('concat_5_'+str(block5), 'conv5_blk_bn', dim=dim_block5_output)
    blob_out = model.Relu('conv5_blk_bn', 'conv5_blk_bn')
    return blob_out

# ---------------------------------------------------------------------------- #
# DenseNet level info for DenseNet121, DenseNet161, ...
# ---------------------------------------------------------------------------- #


DenseNetLevelInfo = collections.namedtuple(
    'DenseNetLevelInfo',
    ['block_counts', 'dims']
)

def DenseNet121_level_info_conv5():
    return DenseNetLevelInfo(
        block_counts=(6, 12, 24, 16),
        dims = (64, 128, 32)
    )

def DenseNet161_level_info_conv5():
    return DenseNetLevelInfo(
        block_counts=(6, 12, 36, 24),
        dims = (96, 192, 48)
    )

def DenseNet169_level_info_conv5():
    return DenseNetLevelInfo(
        block_counts=(6, 12, 32, 32),
        dims = (64, 128, 32)
    )

def DenseNet201_level_info_conv5():
    return DenseNetLevelInfo(
        block_counts=(6, 12, 48, 32),
        dims = (64, 128, 32)
    )







