from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg

def add_Squeezenet_conv5_body(model):
    #kernel size=3, num_output=64, pad=0, stride=2
    model.Conv('data', 'conv1', 3, 64, kernel=3, stride=2)
    model.Relu('conv1', 'conv1')
    model.MaxPool('conv1', 'pool1', stride=2, kernel=3)
    #kernel=1, num_output=16
    model.Conv('pool1', 'fire2-squeeze1x1', 64, 16, kernel=1, pad=0)
    model.Relu('fire2-squeeze1x1', 'fire2-sqeeze1x1')

    #divide
    model.Conv('fire2-sqeeze1x1', 'fire2-expand1x1', 16, 64, kernel=1, stride=1)
    model.Relu('fire2-expand1x1', 'fire2-expand1x1')
    model.Conv('fire2-sqeeze1x1', 'fire2-expand3x3', 16, 64, kernel=3, pad=1 )
    model.Relu('fire2-expand3x3', 'fire2-expand3x3')
    model.Concat(['fire2-expand1x1', 'fire2-expand3x3'], 'fire2-concat')


    model.Conv('fire2-concat', 'fire3-squeeze1x1', 128, 16, kernel=1)
    model.Relu('fire3-squeeze1x1', 'fire3-squeeze1x1')

    #divide
    model.Conv('fire3-squeeze1x1', 'fire3-expand1x1', 16, 64, kernel=1)
    model.Relu('fire3-expand1x1', 'fire3-expand1x1')

    model.Conv('fire3-squeeze1x1', 'fire3-expand3x3', 16, 64, kernel=3, pad=1)
    model.Relu('fire3-expand3x3', 'fire3-expand3x3')
    model.Concat(['fire3-expand1x1' , 'fire3-expand3x3'], 'fire3-concat')
    
    model.MaxPool('fire3-concat', 'pool3', kernel=3, stride=2)
    model.Conv('pool3', 'fire4-squeeze1x1', 128, 32, kernel=1)
    model.Relu('fire4-squeeze1x1', 'fire4-squeeze1x1')

    #divide
    model.Conv('fire4-squeeze1x1', 'fire4-expand1x1', 32, 128, kernel=1)
    model.Relu('fire4-expand1x1', 'fire4-expand1x1')

    model.Conv('fire4-squeeze1x1', 'fire4-expand3x3', 32, 128, pad=1, kernel=3)
    model.Relu('fire4-expand3x3', 'fire4-expand3x3')
    model.Concat(['fire4-expand1x1', 'fire4-expand3x3'], 'fire4-concat')

    #fire5
    model.Conv('fire4-concat', 'fire5-squeeze1x1', 256, 32, kernel=1)
    model.Relu('fire5-squeeze1x1', 'fire5-squeeze1x1')

    #divide
    model.Conv('fire5-squeeze1x1', 'fire5-expand1x1', 32, 128, kernel=1)
    model.Relu('fire5-expand1x1', 'fire5-expand1x1')

    model.Conv('fire5-squeeze1x1', 'fire5-expand3x3', 32, 128, pad=1, kernel=3)
    model.Relu('fire5-expand3x3', 'fire5-expand3x3')
    model.Concat(['fire5-expand1x1', 'fire5-expand3x3'], 'fire5-concat')
    model.MaxPool('fire5-concat', 'pool5', kernel=3, stride=2)

    #fire6
    model.Conv('pool5', 'fire6-squeeze1x1', 256, 48, kernel=1)
    model.Relu('fire6-squeeze1x1', 'fire6-squeeze1x1')

    #divide
    model.Conv('fire6-squeeze1x1', 'fire6-expand1x1', 48, 192, kernel=1)
    model.Relu('fire6-expand1x1', 'fire6-expand1x1')

    model.Conv('fire6-squeeze1x1', 'fire6-expand3x3', 48, 192, pad=1, kernel=3)
    model.Relu('fire6-expand3x3', 'fire6-expand3x3')
    model.Concat(['fire6-expand1x1', 'fire6-expand3x3'], 'fire6-concat')

    #fire7
    model.Conv('fire6-concat', 'fire7-squeeze1x1', 384, 48, kernel=1)
    model.Relu('fire7-squeeze1x1', 'fire7-squeeze1x1')

    #divide
    model.Conv('fire7-squeeze1x1', 'fire7-expand1x1', 48, 192, kernel=1)
    model.Relu('fire7-expand1x1', 'fire7-expand1x1')

    model.Conv('fire7-squeeze1x1', 'fire7-expand3x3', 48, 192, pad=1, kernel=3)
    model.Relu('fire7-expand3x3', 'fire7-expand3x3')
    model.Concat(['fire7-expand1x1', 'fire7-expand3x3'], 'fire7-concat')

    #fire8
    model.Conv('fire7-concat', 'fire8-squeeze1x1', 384, 64, kernel=1)
    model.Relu('fire8-squeeze1x1', 'fire8-squeeze1x1')

    #divide
    model.Conv('fire8-squeeze1x1', 'fire8-expand1x1', 64, 256, kernel=1)
    model.Relu('fire8-expand1x1', 'fire8-expand1x1')

    model.Conv('fire8-squeeze1x1', 'fire8-expand3x3', 64, 256, pad=1, kernel=3)
    model.Relu('fire8-expand3x3', 'fire8-expand3x3')
    model.Concat(['fire8-expand1x1', 'fire8-expand3x3'], 'fire8-concat')

    #fire9
    model.Conv('fire8-concat', 'fire9-squeeze1x1', 512, 64, kernel=1)
    model.Relu('fire9-squeeze1x1', 'fire9-squeeze1x1')

    #divide
    model.Conv('fire9-squeeze1x1', 'fire9-expand1x1', 64, 256, kernel=1)
    model.Relu('fire9-expand1x1', 'fire9-expand1x1')

    model.Conv('fire9-squeeze1x1', 'fire9-expand3x3', 64, 256, pad=1, kernel=3)
    model.Relu('fire9-expand3x3', 'fire9-expand3x3')
    blob_out = model.Concat(['fire9-expand1x1', 'fire9-expand3x3'], 'fire9-concat')

    return blob_out, 512, 1. / 16.