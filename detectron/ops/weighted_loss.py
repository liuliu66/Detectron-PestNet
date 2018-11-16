import numpy as np

from detectron.core.config import cfg
import detectron.utils.boxes as box_utils  

def weighted_loss_forward(inputs, outputs):
    loss1 = inputs[0].data
    weight1 = inputs[1].data
    loss2 = inputs[2].data
    weight2 = inputs[3].data
    outputs[0].reshape(inputs[0].shape)
    outputs[0].data[...] = loss1 * weight1 + loss2 * weight2

def weighted_loss_backward(inputs,outputs):
    grad_output = inputs[-1]

    grad_input = outputs

    grad_input[0].reshape(grad_output.shape)
    grad_input[1].reshape(grad_output.shape)
    grad_input[2].reshape(grad_output.shape)
    grad_input[3].reshape(grad_output.shape)

    grad_input[0].data[...] = grad_output.data * inputs[1].data
    grad_input[1].data[...] = grad_output.data * inputs[2].data
    grad_input[2].data[...] = grad_output.data * inputs[3].data
    grad_input[3].data[...] = grad_output.data * inputs[0].data

