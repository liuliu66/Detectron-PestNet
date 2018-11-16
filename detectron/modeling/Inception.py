from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg

# ------------------------------------------------------------------------------
# Inception-v1
# ------------------------------------------------------------------------------

def add_Inception_v1_conv5_body(model):
    # stem
    model.Conv('data', 'conv1_7x7_s2', 3, 64, 7, pad=3, stride=2)
    model.Relu('conv1_7x7_s2', 'conv1_7x7_s2')
    model.MaxPool('conv1_7x7_s2', 'pool1_3x3_s2', kernel=3, pad=0, stride=2)
    model.LRN('pool1_3x3_s2', 'pool1_norm1', size=5, alpha=0.0001, beta=0.75)
    model.Conv('pool1_norm1', 'conv2_3x3_reduce',64, 64, 1)
    model.Relu('conv2_3x3_reduce', 'conv2_3x3_reduce')
    model.Conv('conv2_3x3_reduce', 'conv2_3x3', 64, 192, kernel=3, pad=1)
    model.Relu('conv2_3x3', 'conv2_3x3')
    model.LRN('conv2_3x3', 'conv2_norm2', size=5, alpha=0.0001, beta=0.75)
    if cfg.MODEL.ATTENTIONAL_TRANSITION:
        model.ChannelAttentionalTransition('conv2_norm2', 'conv2', dim=192)
        model.MaxPool('conv2', 'pool2_3x3_s2', kernel=3, pad=0, stride=2)
    else:
        model.MaxPool('conv2_norm2', 'pool2_3x3_s2', kernel=3, pad=0, stride=2)
    
    # Inception 3a module, output dim 256, spatial_scale 1. / 8.
    model.Conv('pool2_3x3_s2', 'inception_3a_1x1', 192, 64, 1)
    model.Relu('inception_3a_1x1', 'inception_3a_1x1')

    model.Conv('pool2_3x3_s2', 'inception_3a_3x3_reduce', 192, 96, 1)
    model.Relu('inception_3a_3x3_reduce', 'inception_3a_3x3_reduce')
    model.Conv('inception_3a_3x3_reduce', 'inception_3a_3x3', 96, 128, 3, pad=1)
    model.Relu('inception_3a_3x3', 'inception_3a_3x3')

    model.Conv('pool2_3x3_s2', 'inception_3a_5x5_reduce', 192, 16, 1)
    model.Relu('inception_3a_5x5_reduce', 'inception_3a_5x5_reduce')
    model.Conv('inception_3a_5x5_reduce', 'inception_3a_5x5', 16, 32, 5, pad=2)
    model.Relu('inception_3a_5x5', 'inception_3a_5x5')

    model.MaxPool('pool2_3x3_s2', 'inception_3a_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_3a_pool', 'inception_3a_pool_proj', 192, 32, 1)
    model.Relu('inception_3a_pool_proj', 'inception_3a_pool_proj')

    model.Concat(['inception_3a_1x1', 'inception_3a_3x3', 'inception_3a_5x5', 'inception_3a_pool_proj'], 'inception_3a_output')


    # Inception 3b module, output dim 480, spatial_scale 1. / 8.
    model.Conv('inception_3a_output', 'inception_3b_1x1', 64+128+32+32, 128, 1)
    model.Relu('inception_3b_1x1', 'inception_3b_1x1')

    model.Conv('inception_3a_output', 'inception_3b_3x3_reduce', 64+128+32+32, 128, 1)
    model.Relu('inception_3b_3x3_reduce', 'inception_3b_3x3_reduce')
    model.Conv('inception_3b_3x3_reduce', 'inception_3b_3x3', 128, 192, 3, pad=1)
    model.Relu('inception_3b_3x3', 'inception_3b_3x3')

    model.Conv('inception_3a_output', 'inception_3b_5x5_reduce', 64+128+32+32, 32, 1)
    model.Relu('inception_3b_5x5_reduce', 'inception_3b_5x5_reduce')
    model.Conv('inception_3b_5x5_reduce', 'inception_3b_5x5', 32, 96, 5, pad=2)
    model.Relu('inception_3b_5x5', 'inception_3b_5x5')

    model.MaxPool('inception_3a_output', 'inception_3b_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_3b_pool', 'inception_3b_pool_proj', 64+128+32+32, 64, 1)
    model.Relu('inception_3b_pool_proj', 'inception_3b_pool_proj')

    model.Concat(['inception_3b_1x1', 'inception_3b_3x3', 'inception_3b_5x5', 'inception_3b_pool_proj'], 'inception_3b_output')
    if cfg.MODEL.ATTENTIONAL_TRANSITION:
        model.ChannelAttentionalTransition('inception_3b_output', 'inception_3b', dim=480)
        model.MaxPool('inception_3b', 'pool3_3x3_s2', kernel=3, pad=0, stride=2)


    # Inception 4a module, output dim 512, spatial_scale 1. / 16.
    else:
        model.MaxPool('inception_3b_output', 'pool3_3x3_s2', kernel=3, pad=0, stride=2)
    model.Conv('pool3_3x3_s2', 'inception_4a_1x1', 128+192+96+64, 192, 1)
    model.Relu('inception_4a_1x1', 'inception_4a_1x1')

    model.Conv('pool3_3x3_s2', 'inception_4a_3x3_reduce', 128+192+96+64, 96, 1)
    model.Relu('inception_4a_3x3_reduce', 'inception_4a_3x3_reduce')
    model.Conv('inception_4a_3x3_reduce', 'inception_4a_3x3', 96, 208, 3, pad=1)
    model.Relu('inception_4a_3x3', 'inception_4a_3x3')

    model.Conv('pool3_3x3_s2', 'inception_4a_5x5_reduce', 128+192+96+64, 16, 1)
    model.Relu('inception_4a_5x5_reduce', 'inception_4a_5x5_reduce')
    model.Conv('inception_4a_5x5_reduce', 'inception_4a_5x5', 16, 48, 5, pad=2)
    model.Relu('inception_4a_5x5', 'inception_4a_5x5')

    model.MaxPool('pool3_3x3_s2', 'inception_4a_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_4a_pool', 'inception_4a_pool_proj', 128+192+96+64, 64, 1)
    model.Relu('inception_4a_pool_proj', 'inception_4a_pool_proj')

    model.Concat(['inception_4a_1x1', 'inception_4a_3x3', 'inception_4a_5x5', 'inception_4a_pool_proj'], 'inception_4a_output')


    # Inception 4b module, output dim 512, spatial_scale 1. / 16.
    model.Conv('inception_4a_output', 'inception_4b_1x1', 192+208+48+64, 160, 1)
    model.Relu('inception_4b_1x1', 'inception_4b_1x1')

    model.Conv('inception_4a_output', 'inception_4b_3x3_reduce', 192+208+48+64, 112, 1)
    model.Relu('inception_4b_3x3_reduce', 'inception_4b_3x3_reduce')
    model.Conv('inception_4b_3x3_reduce', 'inception_4b_3x3', 112, 224, 3, pad=1)
    model.Relu('inception_4b_3x3', 'inception_4b_3x3')

    model.Conv('inception_4a_output', 'inception_4b_5x5_reduce', 192+208+48+64, 24, 1)
    model.Relu('inception_4b_5x5_reduce', 'inception_4b_5x5_reduce')
    model.Conv('inception_4b_5x5_reduce', 'inception_4b_5x5', 24, 64, 5, pad=2)
    model.Relu('inception_4b_5x5', 'inception_4b_5x5')

    model.MaxPool('inception_4a_output', 'inception_4b_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_4b_pool', 'inception_4b_pool_proj', 192+208+48+64, 64, 1)
    model.Relu('inception_4b_pool_proj', 'inception_4b_pool_proj')

    model.Concat(['inception_4b_1x1', 'inception_4b_3x3', 'inception_4b_5x5', 'inception_4b_pool_proj'], 'inception_4b_output')


    # Inception 4c module, output dim 512, spatial_scale 1. / 16.
    model.Conv('inception_4b_output', 'inception_4c_1x1', 160+224+64+64, 128, 1)
    model.Relu('inception_4c_1x1', 'inception_4c_1x1')

    model.Conv('inception_4b_output', 'inception_4c_3x3_reduce', 160+224+64+64, 128, 1)
    model.Relu('inception_4c_3x3_reduce', 'inception_4c_3x3_reduce')
    model.Conv('inception_4c_3x3_reduce', 'inception_4c_3x3', 128, 256, 3, pad=1)
    model.Relu('inception_4c_3x3', 'inception_4c_3x3')

    model.Conv('inception_4b_output', 'inception_4c_5x5_reduce', 160+224+64+64, 24, 1)
    model.Relu('inception_4c_5x5_reduce', 'inception_4c_5x5_reduce')
    model.Conv('inception_4c_5x5_reduce', 'inception_4c_5x5', 24, 64, 5, pad=2)
    model.Relu('inception_4c_5x5', 'inception_4c_5x5')

    model.MaxPool('inception_4b_output', 'inception_4c_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_4c_pool', 'inception_4c_pool_proj', 160+224+64+64, 64, 1)
    model.Relu('inception_4c_pool_proj', 'inception_4c_pool_proj')

    model.Concat(['inception_4c_1x1', 'inception_4c_3x3', 'inception_4c_5x5', 'inception_4c_pool_proj'], 'inception_4c_output')


    # Inception 4d module, output dim 528, spatial_scale 1. / 16.
    model.Conv('inception_4c_output', 'inception_4d_1x1', 128+256+64+64, 112, 1)
    model.Relu('inception_4d_1x1', 'inception_4d_1x1')

    model.Conv('inception_4c_output', 'inception_4d_3x3_reduce', 128+256+64+64, 144, 1)
    model.Relu('inception_4d_3x3_reduce', 'inception_4d_3x3_reduce')
    model.Conv('inception_4d_3x3_reduce', 'inception_4d_3x3', 144, 288, 3, pad=1)
    model.Relu('inception_4d_3x3', 'inception_4d_3x3')

    model.Conv('inception_4c_output', 'inception_4d_5x5_reduce', 128+256+64+64, 32, 1)
    model.Relu('inception_4d_5x5_reduce', 'inception_4d_5x5_reduce')
    model.Conv('inception_4d_5x5_reduce', 'inception_4d_5x5', 32, 64, 5, pad=2)
    model.Relu('inception_4d_5x5', 'inception_4d_5x5')

    model.MaxPool('inception_4c_output', 'inception_4d_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_4d_pool', 'inception_4d_pool_proj', 128+256+64+64, 64, 1)
    model.Relu('inception_4d_pool_proj', 'inception_4d_pool_proj')

    model.Concat(['inception_4d_1x1', 'inception_4d_3x3', 'inception_4d_5x5', 'inception_4d_pool_proj'], 'inception_4d_output')


    # Inception 4e module, output dim 832, spatial_scale 1. / 16.
    model.Conv('inception_4d_output', 'inception_4e_1x1', 112+288+64+64, 256, 1)
    model.Relu('inception_4e_1x1', 'inception_4e_1x1')

    model.Conv('inception_4d_output', 'inception_4e_3x3_reduce', 112+288+64+64, 160, 1)
    model.Relu('inception_4e_3x3_reduce', 'inception_4e_3x3_reduce')
    model.Conv('inception_4e_3x3_reduce', 'inception_4e_3x3', 160, 320, 3, pad=1)
    model.Relu('inception_4e_3x3', 'inception_4e_3x3')

    model.Conv('inception_4d_output', 'inception_4e_5x5_reduce', 112+288+64+64, 32, 1)
    model.Relu('inception_4e_5x5_reduce', 'inception_4e_5x5_reduce')
    model.Conv('inception_4e_5x5_reduce', 'inception_4e_5x5', 32, 128, 5, pad=2)
    model.Relu('inception_4e_5x5', 'inception_4e_5x5')

    model.MaxPool('inception_4d_output', 'inception_4e_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_4e_pool', 'inception_4e_pool_proj', 112+288+64+64, 128, 1)
    model.Relu('inception_4e_pool_proj', 'inception_4e_pool_proj')

    model.Concat(['inception_4e_1x1', 'inception_4e_3x3', 'inception_4e_5x5', 'inception_4e_pool_proj'], 'inception_4e_output')
    if cfg.MODEL.ATTENTIONAL_TRANSITION:
        model.ChannelAttentionalTransition('inception_4e_output', 'inception_4e', dim=832)
        model.MaxPool('inception_4e', 'pool4_3x3_s2', kernel=3, pad=0, stride=2)


    # Inception 5a module
    else:
        model.MaxPool('inception_4e_output', 'pool4_3x3_s2', kernel=3, pad=0, stride=2)
    model.Conv('pool4_3x3_s2', 'inception_5a_1x1', 256+320+128+128, 256, 1)
    model.Relu('inception_5a_1x1', 'inception_5a_1x1')

    model.Conv('pool4_3x3_s2', 'inception_5a_3x3_reduce', 256+320+128+128, 160, 1)
    model.Relu('inception_5a_3x3_reduce', 'inception_5a_3x3_reduce')
    model.Conv('inception_5a_3x3_reduce', 'inception_5a_3x3', 160, 320, 3, pad=1)
    model.Relu('inception_5a_3x3', 'inception_5a_3x3')

    model.Conv('pool4_3x3_s2', 'inception_5a_5x5_reduce', 256+320+128+128, 32, 1)
    model.Relu('inception_5a_5x5_reduce', 'inception_5a_5x5_reduce')
    model.Conv('inception_5a_5x5_reduce', 'inception_5a_5x5', 32, 128, 5, pad=2)
    model.Relu('inception_5a_5x5', 'inception_5a_5x5')

    model.MaxPool('pool4_3x3_s2', 'inception_5a_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_5a_pool', 'inception_5a_pool_proj', 256+320+128+128, 128, 1)
    model.Relu('inception_5a_pool_proj', 'inception_5a_pool_proj')

    model.Concat(['inception_5a_1x1', 'inception_5a_3x3', 'inception_5a_5x5', 'inception_5a_pool_proj'], 'inception_5a_output')


    # Inception 5b module
    model.Conv('inception_5a_output', 'inception_5b_1x1', 256+320+128+128, 384, 1)
    model.Relu('inception_5b_1x1', 'inception_5b_1x1')

    model.Conv('inception_5a_output', 'inception_5b_3x3_reduce', 256+320+128+128, 192, 1)
    model.Relu('inception_5b_3x3_reduce', 'inception_5b_3x3_reduce')
    model.Conv('inception_5b_3x3_reduce', 'inception_5b_3x3', 192, 384, 3, pad=1)
    model.Relu('inception_5b_3x3', 'inception_5b_3x3')

    model.Conv('inception_5a_output', 'inception_5b_5x5_reduce', 256+320+128+128, 48, 1)
    model.Relu('inception_5b_5x5_reduce', 'inception_5b_5x5_reduce')
    model.Conv('inception_5b_5x5_reduce', 'inception_5b_5x5', 48, 128, 5, pad=2)
    model.Relu('inception_5b_5x5', 'inception_5b_5x5')

    model.MaxPool('inception_5a_output', 'inception_5b_pool', kernel=3, pad=1, stride=1)
    model.Conv('inception_5b_pool', 'inception_5b_pool_proj', 256+320+128+128, 128, 1)
    model.Relu('inception_5b_pool_proj', 'inception_5b_pool_proj')

    blob_out = model.Concat(['inception_5b_1x1', 'inception_5b_3x3', 'inception_5b_5x5', 'inception_5b_pool_proj'], 'inception_5b_output')
    if cfg.MODEL.ATTENTIONAL_TRANSITION:
        blob_out = model.ChannelAttentionalTransition('inception_5b_output', 'inception_5b', dim=1024)
    return blob_out, 1024, 1. / 32.

# ------------------------------------------------------------------------------
# Inception-v4
# ------------------------------------------------------------------------------

def add_Inception_v4_conv5_body(model):
    # stem 1
    model.Conv('data', 'conv1_3x3_s2', 3, 32, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('conv1_3x3_s2', 'conv1_3x3_s2', dim=32)
    model.Relu('conv1_3x3_s2', 'conv1_3x3_s2')
    model.Conv('conv1_3x3_s2', 'conv2_3x3_s1', 32, 32, 3, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv2_3x3_s1', 'conv2_3x3_s1', dim=32)
    model.Relu('conv2_3x3_s1', 'conv2_3x3_s1')
    model.Conv('conv2_3x3_s1', 'conv3_3x3_s1', 32, 64, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('conv3_3x3_s1', 'conv3_3x3_s1', dim=64)
    model.Conv('conv3_3x3_s1', 'inception_stem1_3x3_s2', 64, 96, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('inception_stem1_3x3_s2', 'inception_stem1_3x3_s2', dim=96)
    model.Relu('inception_stem1_3x3_s2', 'inception_stem1_3x3_s2')
    model.MaxPool('conv3_3x3_s1', 'inception_stem1_pool', kernel=3, stride=2)
    model.Concat(['inception_stem1_pool', 'inception_stem1_3x3_s2'], 'inception_stem1')

    # stem 2
    model.Conv('inception_stem1', 'inception_stem2_3x3_reduce', 96+64, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_3x3_reduce', 'inception_stem2_3x3_reduce', dim=64)
    model.Relu('inception_stem2_3x3_reduce', 'inception_stem2_3x3_reduce')
    model.Conv('inception_stem2_3x3_reduce', 'inception_stem2_3x3', 64, 96, 3, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_3x3', 'inception_stem2_3x3', dim=96)
    model.Relu('inception_stem2_3x3', 'inception_stem2_3x3')

    model.Conv('inception_stem1', 'inception_stem2_1x7_reduce', 96+64, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_1x7_reduce', 'inception_stem2_1x7_reduce', dim=64)
    model.Relu('inception_stem2_1x7_reduce', 'inception_stem2_1x7_reduce')
    model.Conv('inception_stem2_1x7_reduce', 'inception_stem2_1x7', 64, 64, [1, 7], pad_h=0, pad_w=3, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_1x7', 'inception_stem2_1x7', dim=64)
    model.Relu('inception_stem2_1x7', 'inception_stem2_1x7')
    model.Conv('inception_stem2_1x7', 'inception_stem2_7x1', 64, 64, [7, 1], pad_h=3, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_7x1', 'inception_stem2_7x1', dim=64)
    model.Relu('inception_stem2_7x1', 'inception_stem2_7x1')
    model.Conv('inception_stem2_7x1', 'inception_stem2_3x3_2', 64, 96, 3, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_stem2_3x3_2', 'inception_stem2_3x3_2', dim=96)
    model.Relu('inception_stem2_3x3_2', 'inception_stem2_3x3_2')
    model.Concat(['inception_stem2_3x3_2', 'inception_stem2_3x3'], 'inception_stem2')

    # stem 3
    model.Conv('inception_stem2', 'inception_stem3_3x3_s2', 96+96, 192, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('inception_stem3_3x3_s2', 'inception_stem3_3x3_s2', dim=192)
    model.Relu('inception_stem3_3x3_s2', 'inception_stem3_3x3_s2')
    model.MaxPool('inception_stem2', 'inception_stem3_pool', kernel=3, stride=2)
    model.Concat(['inception_stem3_3x3_s2', 'inception_stem3_pool'], 'inception_stem3')

    # Inception a modules
    for i in range(4):
        if i == 0:
            add_Inception_v4_a_modules(model, 'inception_stem3', 'inception_a' + str(i+1) + '_concat', str(i+1))
        else:
            add_Inception_v4_a_modules(model, 'inception_a' + str(i) + '_concat', 'inception_a' + str(i+1) + '_concat', str(i+1))

    model.Conv('inception_a4_concat', 'reduction_a_3x3', 384, 384, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_a_3x3', 'reduction_a_3x3', dim=384)
    model.Relu('reduction_a_3x3', 'reduction_a_3x3')

    model.Conv('inception_a4_concat', 'reduction_a_3x3_2_reduce', 384, 192, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2_reduce', dim=192)
    model.Relu('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2_reduce')
    model.Conv('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2', 192, 224, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('reduction_a_3x3_2', 'reduction_a_3x3_2', dim=224)
    model.Relu('reduction_a_3x3_2', 'reduction_a_3x3_2')
    model.Conv('reduction_a_3x3_2', 'reduction_a_3x3_3', 224, 256, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_a_3x3_3', 'reduction_a_3x3_3', dim=256)
    model.Relu('reduction_a_3x3_3', 'reduction_a_3x3_3')

    model.MaxPool('inception_a4_concat', 'reduction_a_pool', kernel=3, stride=2)
    model.Concat(['reduction_a_3x3', 'reduction_a_3x3_3', 'reduction_a_pool'], 'reduction_a_concat')

    # Inception b modules
    for i in range(7):
        if i == 0:
            add_Inception_v4_b_modules(model, 'reduction_a_concat', 'inception_b' + str(i+1) + '_concat', str(i+1))
        else:
            add_Inception_v4_b_modules(model, 'inception_b' + str(i) + '_concat', 'inception_b' + str(i+1) + '_concat', str(i+1))

    model.Conv('inception_b7_concat', 'reduction_b_3x3_reduce', 1024, 192, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_3x3_reduce', 'reduction_b_3x3_reduce', dim=192)
    model.Relu('reduction_b_3x3_reduce', 'reduction_b_3x3_reduce')
    model.Conv('reduction_b_3x3_reduce', 'reduction_b_3x3', 192, 192, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_b_3x3', 'reduction_b_3x3', dim=192)
    model.Relu('reduction_b_3x3', 'reduction_b_3x3')

    model.Conv('inception_b7_concat', 'reduction_b_1x7_reduce', 1024, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_1x7_reduce', 'reduction_b_1x7_reduce', dim=256)
    model.Relu('reduction_b_1x7_reduce', 'reduction_b_1x7_reduce')
    model.Conv('reduction_b_1x7_reduce', 'reduction_b_1x7', 256, 256, [1, 7], pad_h=0, pad_w=3, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_1x7', 'reduction_b_1x7', dim=256)
    model.Relu('reduction_b_1x7', 'reduction_b_1x7')
    model.Conv('reduction_b_1x7', 'reduction_b_7x1', 256, 320, [7, 1], pad_h=3, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_7x1', 'reduction_b_7x1', dim=320)
    model.Relu('reduction_b_7x1', 'reduction_b_7x1')
    model.Conv('reduction_b_7x1', 'reduction_b_3x3_2', 320, 320, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_b_3x3_2', 'reduction_b_3x3_2', dim=320)
    model.Relu('reduction_b_3x3_2', 'reduction_b_3x3_2')

    model.MaxPool('inception_b7_concat', 'reduction_b_pool', kernel=3, stride=2)
    model.Concat(['reduction_b_3x3', 'reduction_b_3x3_2', 'reduction_b_pool'], 'reduction_b_concat')

    # Inception c modules
    for i in range(3):
        if i == 0:
            add_Inception_v4_c_modules(model, 'reduction_b_concat', 'inception_c' + str(i+1) + '_concat', str(i+1))
        else:
            add_Inception_v4_c_modules(model, 'inception_c' + str(i) + '_concat', 'inception_c' + str(i+1) + '_concat', str(i+1))
    return model, 1536, 1. / 37.5

# ------------------------------------------------------------------------------
# various sub-modules (a, b, c...)
# ------------------------------------------------------------------------------

def add_Inception_v4_a_modules(model, blob_in, blob_out, num):
    model.Conv(blob_in, 'inception_a'+num+'_1x1_2', 384, 96, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_1x1_2', 'inception_a'+num+'_1x1_2', dim=96)
    model.Relu('inception_a'+num+'_1x1_2', 'inception_a'+num+'_1x1_2')

    model.Conv(blob_in, 'inception_a'+num+'_3x3_reduce', 384, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_3x3_reduce', 'inception_a'+num+'_3x3_reduce', dim=64)
    model.Relu('inception_a'+num+'_3x3_reduce', 'inception_a'+num+'_3x3_reduce')
    model.Conv('inception_a'+num+'_3x3_reduce', 'inception_a'+num+'_3x3', 64, 96, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_3x3', 'inception_a'+num+'_3x3', dim=96)
    model.Relu('inception_a'+num+'_3x3', 'inception_a'+num+'_3x3')

    model.Conv(blob_in, 'inception_a'+num+'_3x3_2_reduce', 384, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_3x3_2_reduce', 'inception_a'+num+'_3x3_2_reduce', dim=64)
    model.Relu('inception_a'+num+'_3x3_2_reduce', 'inception_a'+num+'_3x3_2_reduce')
    model.Conv('inception_a'+num+'_3x3_2_reduce', 'inception_a'+num+'_3x3_2', 64, 96, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_3x3_2', 'inception_a'+num+'_3x3_2', dim=96)
    model.Relu('inception_a'+num+'_3x3_2', 'inception_a'+num+'_3x3_2')
    model.Conv('inception_a'+num+'_3x3_2', 'inception_a'+num+'_3x3_3', 96, 96, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_3x3_3', 'inception_a'+num+'_3x3_3', dim=96)
    model.Relu('inception_a'+num+'_3x3_3', 'inception_a'+num+'_3x3_3')

    model.AveragePool(blob_in, 'inception_a'+num+'_pool_ave', kernel=3, pad=1, stride=1)
    model.Conv('inception_a'+num+'_pool_ave', 'inception_a'+num+'_1x1', 384, 96, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_a'+num+'_1x1', 'inception_a'+num+'_1x1', dim=96)
    model.Relu('inception_a'+num+'_1x1', 'inception_a'+num+'_1x1')

    model.Concat(['inception_a'+num+'_1x1_2', 'inception_a'+num+'_3x3', 'inception_a'+num+'_3x3_3', 'inception_a'+num+'_1x1'], blob_out)

def add_Inception_v4_b_modules(model, blob_in, blob_out, num):
    model.Conv(blob_in, 'inception_b'+num+'_1x1_2', 1024, 384, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x1_2', 'inception_b'+num+'_1x1_2', dim=384)
    model.Relu('inception_b'+num+'_1x1_2', 'inception_b'+num+'_1x1_2')

    model.Conv(blob_in, 'inception_b'+num+'_1x7_reduce', 1024, 192, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x7_reduce', 'inception_b'+num+'_1x7_reduce', dim=192)
    model.Relu('inception_b'+num+'_1x7_reduce', 'inception_b'+num+'_1x7_reduce')
    model.Conv('inception_b'+num+'_1x7_reduce', 'inception_b'+num+'_1x7', 192, 224, [1, 7], pad_h=0, pad_w=3, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x7', 'inception_b'+num+'_1x7', dim=224)
    model.Relu('inception_b'+num+'_1x7', 'inception_b'+num+'_1x7')
    model.Conv('inception_b'+num+'_1x7', 'inception_b'+num+'_7x1', 224, 256, [7, 1], pad_h=0, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_7x1', 'inception_b'+num+'_7x1', dim=256)
    model.Relu('inception_b'+num+'_7x1', 'inception_b'+num+'_7x1')

    model.Conv(blob_in, 'inception_b'+num+'_7x1_2_reduce', 1024, 192, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_7x1_2_reduce', 'inception_b'+num+'_7x1_2_reduce', dim=192)
    model.Relu('inception_b'+num+'_7x1_2_reduce', 'inception_b'+num+'_7x1_2_reduce')
    model.Conv('inception_b'+num+'_7x1_2_reduce', 'inception_b'+num+'_7x1_2', 192, 192, [7, 1], pad_h=3, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_7x1_2', 'inception_b'+num+'_7x1_2', dim=192)
    model.Relu('inception_b'+num+'_7x1_2', 'inception_b'+num+'_7x1_2')
    model.Conv('inception_b'+num+'_7x1_2', 'inception_b'+num+'_1x7_2', 192, 224, [1, 7], pad_h=0, pad_w=3, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x7_2', 'inception_b'+num+'_1x7_2', dim=224)
    model.Relu('inception_b'+num+'_1x7_2', 'inception_b'+num+'_1x7_2')
    model.Conv('inception_b'+num+'_1x7_2', 'inception_b'+num+'_7x1_3', 224, 224, [7, 1], pad_h=3, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_7x1_3', 'inception_b'+num+'_7x1_3', dim=224)
    model.Relu('inception_b'+num+'_7x1_3', 'inception_b'+num+'_7x1_3')
    model.Conv('inception_b'+num+'_7x1_3', 'inception_b'+num+'_1x7_3', 224, 256, [1, 7], pad_h=0, pad_w=3, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x7_3', 'inception_b'+num+'_1x7_3', dim=256)
    model.Relu('inception_b'+num+'_1x7_3', 'inception_b'+num+'_1x7_3')

    model.AveragePool(blob_in, 'inception_b'+num+'_pool_ave', kernel=3, pad=1, stride=1)
    model.Conv('inception_b'+num+'_pool_ave', 'inception_b'+num+'_1x1', 1024, 128, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_b'+num+'_1x1', 'inception_b'+num+'_1x1', dim=128)
    model.Relu('inception_b'+num+'_1x1', 'inception_b'+num+'_1x1')

    model.Concat(['inception_b'+num+'_1x1_2', 'inception_b'+num+'_7x1', 'inception_b'+num+'_1x7_3', 'inception_b'+num+'_1x1'], blob_out)

def add_Inception_v4_c_modules(model, blob_in, blob_out, num):
    model.Conv(blob_in, 'inception_c'+num+'_1x1_2', 1536, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x1_2', 'inception_c'+num+'_1x1_2', dim=256)
    model.Relu('inception_c'+num+'_1x1_2', 'inception_c'+num+'_1x1_2')

    model.Conv(blob_in, 'inception_c'+num+'_1x1_3', 1536, 384, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x1_3', 'inception_c'+num+'_1x1_3', dim=384)
    model.Relu('inception_c'+num+'_1x1_3', 'inception_c'+num+'_1x1_3')
    model.Conv('inception_c'+num+'_1x1_3', 'inception_c'+num+'_1x3', 384, 256, [1, 3], pad_h=0, pad_w=1, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x3', 'inception_c'+num+'_1x3', dim=256)
    model.Relu('inception_c'+num+'_1x3', 'inception_c'+num+'_1x3')
    model.Conv('inception_c'+num+'_1x1_3', 'inception_c'+num+'_3x1', 384, 256, [3, 1], pad_h=1, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_3x1', 'inception_c'+num+'_3x1', dim=256)
    model.Relu('inception_c'+num+'_3x1', 'inception_c'+num+'_3x1')

    model.Conv(blob_in, 'inception_c'+num+'_1x1_4', 1536, 384, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x1_4', 'inception_c'+num+'_1x1_4', dim=384)
    model.Relu('inception_c'+num+'_1x1_4', 'inception_c'+num+'_1x1_4')
    model.Conv('inception_c'+num+'_1x1_4', 'inception_c'+num+'_3x1_2', 384, 448, [3, 1], pad_h=1, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_3x1_2', 'inception_c'+num+'_3x1_2', dim=448)
    model.Relu('inception_c'+num+'_3x1_2', 'inception_c'+num+'_3x1_2')
    model.Conv('inception_c'+num+'_3x1_2', 'inception_c'+num+'_1x3_2', 448, 512, [1, 3], pad_h=0, pad_w=1, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x3_2', 'inception_c'+num+'_1x3_2', dim=512)
    model.Relu('inception_c'+num+'_1x3_2', 'inception_c'+num+'_1x3_2')
    model.Conv('inception_c'+num+'_1x3_2', 'inception_c'+num+'_1x3_3', 512, 256, [1, 3], pad_h=0, pad_w=1, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x3_3', 'inception_c'+num+'_1x3_3', dim=256)
    model.Relu('inception_c'+num+'_1x3_3', 'inception_c'+num+'_1x3_3')
    model.Conv('inception_c'+num+'_1x3_2', 'inception_c'+num+'_3x1_3', 512, 256, [3, 1], pad_h=1, pad_w=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_3x1_3', 'inception_c'+num+'_3x1_3', dim=256)
    model.Relu('inception_c'+num+'_3x1_3', 'inception_c'+num+'_3x1_3')

    model.AveragePool(blob_in, 'inception_c'+num+'_pool_ave', kernel=3, pad=1, stride=1)
    model.Conv('inception_c'+num+'_pool_ave', 'inception_c'+num+'_1x1', 1536, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('inception_c'+num+'_1x1', 'inception_c'+num+'_1x1', dim=256)
    model.Relu('inception_c'+num+'_1x1', 'inception_c'+num+'_1x1')

    model.Concat(['inception_c'+num+'_1x1_2', 'inception_c'+num+'_1x3', 'inception_c'+num+'_3x1', 'inception_c'+num+'_1x3_3', 'inception_c'+num+'_3x1_3', 'inception_c'+num+'_1x1'], blob_out)

# ------------------------------------------------------------------------------
# Inception-ResNet-v2
# ------------------------------------------------------------------------------

def add_Inception_ResNet_v2_conv5_body(model):
    #stem conv 1
    model.Conv('data', 'conv1_3x3_s2', 3, 32, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('conv1_3x3_s2', 'conv1_3x3_s2', dim=32)
    model.Relu('conv1_3x3_s2', 'conv1_3x3_s2')

    #stem conv 2
    model.Conv('conv1_3x3_s2', 'conv2_3x3_s1', 32, 32, 3, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv2_3x3_s1', 'conv2_3x3_s1', dim=32)
    model.Relu('conv2_3x3_s1', 'conv2_3x3_s1')

    #stem conv 3
    model.Conv('conv2_3x3_s1', 'conv3_3x3_s1', 32, 64, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('conv3_3x3_s1', 'conv3_3x3_s1', dim=64)
    model.Relu('conv3_3x3_s1', 'conv3_3x3_s1')
    model.MaxPool('conv3_3x3_s1', 'pool1_3x3_s2', kernel=3, stride=2)

    #stem conv 4
    model.Conv('pool1_3x3_s2', 'conv4_3x3_reduce', 64, 80, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv4_3x3_reduce', 'conv4_3x3_reduce', dim=80)
    model.Relu('conv4_3x3_reduce', 'conv4_3x3_reduce')
    model.Conv('conv4_3x3_reduce', 'conv4_3x3', 80, 192, 3, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv4_3x3', 'conv4_3x3', dim=192)
    model.Relu('conv4_3x3', 'conv4_3x3')
    model.MaxPool('conv4_3x3', 'pool2_3x3_s2', kernel=3, stride=2)

    #stem conv 5
    model.Conv('pool2_3x3_s2', 'conv5_1x1', 192, 96, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv5_1x1', 'conv5_1x1', dim=96)
    model.Relu('conv5_1x1', 'conv5_1x1')

    model.Conv('pool2_3x3_s2', 'conv5_5x5_reduce', 192, 48, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv5_5x5_reduce', 'conv5_5x5_reduce', dim=48)
    model.Relu('conv5_5x5_reduce', 'conv5_5x5_reduce')
    model.Conv('conv5_5x5_reduce', 'conv5_5x5', 48, 64, 5, pad=2, stride=1, no_bias=1)
    model.AffineChannel('conv5_5x5', 'conv5_5x5', dim=64)
    model.Relu('conv5_5x5', 'conv5_5x5')

    model.Conv('pool2_3x3_s2', 'conv5_3x3_reduce', 192, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv5_3x3_reduce', 'conv5_3x3_reduce', dim=64)
    model.Relu('conv5_3x3_reduce', 'conv5_3x3_reduce')
    model.Conv('conv5_3x3_reduce', 'conv5_3x3', 64, 96, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('conv5_3x3', 'conv5_3x3', dim=96)
    model.Relu('conv5_3x3', 'conv5_3x3')
    model.Conv('conv5_3x3', 'conv5_3x3_2', 96, 96, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('conv5_3x3_2', 'conv5_3x3_2', dim=96)
    model.Relu('conv5_3x3_2', 'conv5_3x3_2')

    model.AveragePool('pool2_3x3_s2', 'ave_pool', kernel=3, pad=1, stride=1)
    model.Conv('ave_pool', 'conv5_1x1_ave', 192, 64, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv5_1x1_ave', 'conv5_1x1_ave', dim=64)
    model.Relu('conv5_1x1_ave', 'conv5_1x1_ave')

    model.Concat(['conv5_1x1', 'conv5_5x5', 'conv5_3x3_2', 'conv5_1x1_ave'], 'stem_concat')

    #inception resnet module a
    for lvl in range(1, 10+1):
        blob_in = 'inception_resnet_v2_a'+str(lvl-1)+'_residual_eltwise'
        if lvl == 1:
            blob_in = 'stem_concat'
        slvl = str(lvl)
        model.Conv(blob_in, 'inception_resnet_v2_a'+slvl+'_1x1', 320, 32, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_1x1', 'inception_resnet_v2_a'+slvl+'_1x1', dim=32)
        model.Relu('inception_resnet_v2_a'+slvl+'_1x1', 'inception_resnet_v2_a'+slvl+'_1x1')

        model.Conv(blob_in, 'inception_resnet_v2_a'+slvl+'_3x3_reduce', 320, 32, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_3x3_reduce', 'inception_resnet_v2_a'+slvl+'_3x3_reduce', dim=32)
        model.Relu('inception_resnet_v2_a'+slvl+'_3x3_reduce', 'inception_resnet_v2_a'+slvl+'_3x3_reduce')
        model.Conv('inception_resnet_v2_a'+slvl+'_3x3_reduce', 'inception_resnet_v2_a'+slvl+'_3x3', 32, 32, 3, pad = 1, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_3x3', 'inception_resnet_v2_a'+slvl+'_3x3', dim=32)
        model.Relu('inception_resnet_v2_a'+slvl+'_3x3', 'inception_resnet_v2_a'+slvl+'_3x3')

        model.Conv(blob_in, 'inception_resnet_v2_a'+slvl+'_3x3_2_reduce', 320, 32, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_3x3_2_reduce', 'inception_resnet_v2_a'+slvl+'_3x3_2_reduce', dim=32)
        model.Relu('inception_resnet_v2_a'+slvl+'_3x3_2_reduce', 'inception_resnet_v2_a'+slvl+'_3x3_2_reduce')
        model.Conv('inception_resnet_v2_a'+slvl+'_3x3_2_reduce', 'inception_resnet_v2_a'+slvl+'_3x3_2', 32, 48, 3, pad=1, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_3x3_2', 'inception_resnet_v2_a'+slvl+'_3x3_2', dim=48)
        model.Relu('inception_resnet_v2_a'+slvl+'_3x3_2', 'inception_resnet_v2_a'+slvl+'_3x3_2')
        model.Conv('inception_resnet_v2_a'+slvl+'_3x3_2', 'inception_resnet_v2_a'+slvl+'_3x3_3', 48, 64, 3, pad=1, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_a'+slvl+'_3x3_3', 'inception_resnet_v2_a'+slvl+'_3x3_3', dim=64)
        model.Relu('inception_resnet_v2_a'+slvl+'_3x3_3', 'inception_resnet_v2_a'+slvl+'_3x3_3')

        model.Concat(['inception_resnet_v2_a'+slvl+'_1x1', 'inception_resnet_v2_a'+slvl+'_3x3', 'inception_resnet_v2_a'+slvl+'_3x3_3'], 'inception_resnet_v2_a'+slvl+'_concat')

        model.Conv('inception_resnet_v2_a'+slvl+'_concat', 'inception_resnet_v2_a'+slvl+'_up', 128, 320, 1, pad=0, stride=1)
        model.net.Sum([blob_in, 'inception_resnet_v2_a'+slvl+'_up'], 'inception_resnet_v2_a'+slvl+'_residual_eltwise')
        model.Relu('inception_resnet_v2_a'+slvl+'_residual_eltwise', 'inception_resnet_v2_a'+slvl+'_residual_eltwise')

    model.Conv('inception_resnet_v2_a10_residual_eltwise', 'reduction_a_3x3', 320, 384, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_a_3x3', 'reduction_a_3x3', dim=384)
    model.Relu('reduction_a_3x3', 'reduction_a_3x3')

    model.Conv('inception_resnet_v2_a10_residual_eltwise', 'reduction_a_3x3_2_reduce', 320, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2_reduce', dim=256)
    model.Relu('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2_reduce')
    model.Conv('reduction_a_3x3_2_reduce', 'reduction_a_3x3_2', 256, 256, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('reduction_a_3x3_2', 'reduction_a_3x3_2', dim=256)
    model.Relu('reduction_a_3x3_2', 'reduction_a_3x3_2')
    model.Conv('reduction_a_3x3_2', 'reduction_a_3x3_3', 256, 384, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_a_3x3_3', 'reduction_a_3x3_3', dim=384)
    model.Relu('reduction_a_3x3_3', 'reduction_a_3x3_3')

    model.MaxPool('inception_resnet_v2_a10_residual_eltwise', 'reduction_a_pool', kernel=3, stride=2)

    model.Concat(['reduction_a_3x3', 'reduction_a_3x3_3', 'reduction_a_pool'], 'reduction_a_concat')

    #inception resnet module b
    for lvl in range(1, 20+1):
        blob_in = 'inception_resnet_v2_b'+str(lvl-1)+'_residual_eltwise'
        if lvl == 1:
            blob_in = 'reduction_a_concat'
        slvl = str(lvl)
        model.Conv(blob_in, 'inception_resnet_v2_b'+slvl+'_1x1', 1088, 192, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_b'+slvl+'_1x1', 'inception_resnet_v2_b'+slvl+'_1x1', dim=192)
        model.Relu('inception_resnet_v2_b'+slvl+'_1x1', 'inception_resnet_v2_b'+slvl+'_1x1')

        model.Conv(blob_in, 'inception_resnet_v2_b'+slvl+'_1x7_reduce', 1088, 128, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_b'+slvl+'_1x7_reduce', 'inception_resnet_v2_b'+slvl+'_1x7_reduce', dim=128)
        model.Relu('inception_resnet_v2_b'+slvl+'_1x7_reduce', 'inception_resnet_v2_b'+slvl+'_1x7_reduce')
        model.Conv('inception_resnet_v2_b'+slvl+'_1x7_reduce', 'inception_resnet_v2_b'+slvl+'_1x7', 128, 160, [1, 7], pad=3, stride=1,no_bias=1)
        model.AffineChannel('inception_resnet_v2_b'+slvl+'_1x7', 'inception_resnet_v2_b'+slvl+'_1x7', dim=160)
        model.Relu('inception_resnet_v2_b'+slvl+'_1x7', 'inception_resnet_v2_b'+slvl+'_1x7')
        model.Conv('inception_resnet_v2_b'+slvl+'_1x7', 'inception_resnet_v2_b'+slvl+'_7x1', 160, 192, [7, 1], pad=0, stride=1, no_bias=1)
        model.AffineChannel( 'inception_resnet_v2_b'+slvl+'_7x1',  'inception_resnet_v2_b'+slvl+'_7x1', dim=192)
        model.Relu( 'inception_resnet_v2_b'+slvl+'_7x1',  'inception_resnet_v2_b'+slvl+'_7x1')
        #model.Conv('inception_resnet_v2_b'+slvl+'_1x7_reduce', 'inception_resnet_v2_b'+slvl+'_7x7', 128, 192, 3, pad=1, stride=1, no_bias=1)
        #model.AffineChannel('inception_resnet_v2_b'+slvl+'_7x7', 'inception_resnet_v2_b'+slvl+'_7x7', dim=192)
        #model.Relu('inception_resnet_v2_b'+slvl+'_7x7', 'inception_resnet_v2_b'+slvl+'_7x7')

        model.Concat(['inception_resnet_v2_b'+slvl+'_1x1', 'inception_resnet_v2_b'+slvl+'_7x1'], 'inception_resnet_v2_b'+slvl+'_concat')

        model.Conv('inception_resnet_v2_b'+slvl+'_concat', 'inception_resnet_v2_b'+slvl+'_up', 384, 1088, 1, pad=0, stride=1)
        model.net.Sum([blob_in, 'inception_resnet_v2_b'+slvl+'_up'], 'inception_resnet_v2_b'+slvl+'_residual_eltwise')
        model.Relu('inception_resnet_v2_b'+slvl+'_residual_eltwise', 'inception_resnet_v2_b'+slvl+'_residual_eltwise')

    model.Conv('inception_resnet_v2_b20_residual_eltwise', 'reduction_b_3x3_reduce', 1088, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_3x3_reduce', 'reduction_b_3x3_reduce', dim=256)
    model.Relu('reduction_b_3x3_reduce', 'reduction_b_3x3_reduce')
    model.Conv('reduction_b_3x3_reduce', 'reduction_b_3x3', 256, 384, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_b_3x3', 'reduction_b_3x3', dim=384)
    model.Relu('reduction_b_3x3', 'reduction_b_3x3')

    model.Conv('inception_resnet_v2_b20_residual_eltwise', 'reduction_b_3x3_2_reduce', 1088, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_3x3_2_reduce', 'reduction_b_3x3_2_reduce', dim=256)
    model.Relu('reduction_b_3x3_2_reduce', 'reduction_b_3x3_2_reduce')
    model.Conv('reduction_b_3x3_2_reduce', 'reduction_b_3x3_2', 256, 288, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_b_3x3_2', 'reduction_b_3x3_2', dim=288)
    model.Relu('reduction_b_3x3_2', 'reduction_b_3x3_2')

    model.Conv('inception_resnet_v2_b20_residual_eltwise', 'reduction_b_3x3_3_reduce', 1088, 256, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_3x3_3_reduce', 'reduction_b_3x3_3_reduce', dim=256)
    model.Relu('reduction_b_3x3_3_reduce', 'reduction_b_3x3_3_reduce')
    model.Conv('reduction_b_3x3_3_reduce', 'reduction_b_3x3_3', 256, 288, 3, pad=1, stride=1, no_bias=1)
    model.AffineChannel('reduction_b_3x3_3', 'reduction_b_3x3_3', dim=288)
    model.Relu('reduction_b_3x3_3', 'reduction_b_3x3_3')
    model.Conv('reduction_b_3x3_3', 'reduction_b_3x3_4', 288, 320, 3, pad=0, stride=2, no_bias=1)
    model.AffineChannel('reduction_b_3x3_4', 'reduction_b_3x3_4', dim=320)
    model.Relu('reduction_b_3x3_4', 'reduction_b_3x3_4')

    model.MaxPool('inception_resnet_v2_b20_residual_eltwise', 'reduction_b_pool', kernel=3, stride=2)

    model.Concat(['reduction_b_3x3', 'reduction_b_3x3_2', 'reduction_b_3x3_4', 'reduction_b_pool'], 'reduction_b_concat')

    #inception resnet module c
    for lvl in range(1, 10+1):
        blob_in = 'inception_resnet_v2_c'+str(lvl-1)+'_residual_eltwise'
        if lvl == 1:
            blob_in = 'reduction_b_concat'
        slvl = str(lvl)
        model.Conv(blob_in, 'inception_resnet_v2_c'+slvl+'_1x1', 2080, 192, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_c'+slvl+'_1x1', 'inception_resnet_v2_c'+slvl+'_1x1', dim=192)
        model.Relu('inception_resnet_v2_c'+slvl+'_1x1', 'inception_resnet_v2_c'+slvl+'_1x1')
        
        model.Conv(blob_in, 'inception_resnet_v2_c'+slvl+'_1x3_reduce', 2080, 192, 1, pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_c'+slvl+'_1x3_reduce', 'inception_resnet_v2_c'+slvl+'_1x3_reduce', dim=192)
        model.Relu('inception_resnet_v2_c'+slvl+'_1x3_reduce', 'inception_resnet_v2_c'+slvl+'_1x3_reduce')
        model.Conv('inception_resnet_v2_c'+slvl+'_1x3_reduce', 'inception_resnet_v2_c'+slvl+'_1x3', 192, 224, [1, 3], pad=1, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_c'+slvl+'_1x3', 'inception_resnet_v2_c'+slvl+'_1x3', dim=224)
        model.Relu('inception_resnet_v2_c'+slvl+'_1x3', 'inception_resnet_v2_c'+slvl+'_1x3')
        model.Conv('inception_resnet_v2_c'+slvl+'_1x3', 'inception_resnet_v2_c'+slvl+'_3x1', 224, 256, [3, 1], pad=0, stride=1, no_bias=1)
        model.AffineChannel('inception_resnet_v2_c'+slvl+'_3x1', 'inception_resnet_v2_c'+slvl+'_3x1', dim=256)
        model.Relu('inception_resnet_v2_c'+slvl+'_3x1', 'inception_resnet_v2_c'+slvl+'_3x1')
        #model.Conv('inception_resnet_v2_c'+slvl+'_1x3_reduce', 'inception_resnet_v2_c'+slvl+'_3x3',192, 256, 3, pad=1, stride=1, no_bias=1)
        #model.AffineChannel('inception_resnet_v2_c'+slvl+'_3x3', 'inception_resnet_v2_c'+slvl+'_3x3', dim=256)
        #model.Relu('inception_resnet_v2_c'+slvl+'_3x3', 'inception_resnet_v2_c'+slvl+'_3x3')

        model.Concat(['inception_resnet_v2_c'+slvl+'_1x1', 'inception_resnet_v2_c'+slvl+'_3x1'], 'inception_resnet_v2_c'+slvl+'_concat')
        model.Conv('inception_resnet_v2_c'+slvl+'_concat', 'inception_resnet_v2_c'+slvl+'_up', 448, 2080, 1, pad=0, stride=1)
        
        model.net.Sum([blob_in, 'inception_resnet_v2_c'+slvl+'_up'], 'inception_resnet_v2_c'+slvl+'_residual_eltwise')
        model.Relu('inception_resnet_v2_c'+slvl+'_residual_eltwise', 'inception_resnet_v2_c'+slvl+'_residual_eltwise')

    #conv 6
    model.Conv('inception_resnet_v2_c10_residual_eltwise', 'conv6_1x1', 2080, 1536, 1, pad=0, stride=1, no_bias=1)
    model.AffineChannel('conv6_1x1', 'conv6_1x1', dim=1536)
    blob_out = model.Relu('conv6_1x1', 'conv6_1x1')

    return blob_out, 1536, 1. / 32.
















