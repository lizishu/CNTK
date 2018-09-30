import numpy as np
import cntk as C
from cntk.initializer import he_normal, normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense, Dropout
from cntk.ops import element_times, relu

#
# assembly components
#
def conv_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), bn_init_scale=1):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False, init_scale=bn_init_scale, disable_regularization=True)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1, 1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init, 1)
    return relu(r)

#
# ResNet components
#
def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3, 3), num_filters)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    p = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2, 2)):
    c1 = conv_bn_relu(input, (3, 3), num_filters, strides)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    s = conv_bn(input, (1, 1), num_filters, strides) # Shortcut
    p = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_basic(l, num_filters)
    return l

def resnet_bottleneck(input, out_num_filters, inter_out_num_filters):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    p = c3 + input
    return relu(p)

def resnet_bottleneck_inc(input, out_num_filters, inter_out_num_filters, stride1x1, stride3x3):
    c1 = conv_bn_relu(input, (1, 1), inter_out_num_filters, strides=stride1x1)
    c2 = conv_bn_relu(c1, (3, 3), inter_out_num_filters, strides=stride3x3)
    c3 = conv_bn(c2, (1, 1), out_num_filters, bn_init_scale=0)
    stride = np.multiply(stride1x1, stride3x3)
    s = conv_bn(input, (1, 1), out_num_filters, strides=stride) # Shortcut
    p = c3 + s
    return relu(p)

def resnet_bottleneck_stack(input, num_stack_layers, out_num_filters, inter_out_num_filters): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_bottleneck(l, out_num_filters, inter_out_num_filters)
    return l

def resnet_drop(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters, bn_init_scale = 1)

    b1 = Dropout(0.5)(input)

    p = b1 + c2

    return relu(p)

def resnet_drop_stack(input, num_stack_layers, num_filters):
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_drop(l, num_filters)
    return l

#
# Defines the residual network model for classifying images
#
def create_cifar10_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]

    conv = conv_bn_relu(input, (3, 3), c_map[0])
    r1 = resnet_drop_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_drop_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_drop_stack(r3_1, num_stack_layers-1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8), name='final_avg_pooling')(r3_2)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z