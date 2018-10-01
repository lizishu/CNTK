import numpy as np
from cntk.initializer import he_normal, normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
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

def mobilenet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), 1)
    c2 = conv_bn_relu(c1, (1,1), num_filters)
    return c2

def mobilenet_basic_inc(input, num_filters, strides = (2,2)):
    c1 = conv_bn_relu(input, (3,3), 1, strides)
    c2 = conv_bn_relu(c1, (1,1), num_filters)
    return c2

def mobilenet_basic_stack(input, num_stack_layers, num_filters):
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = mobilenet_basic(l, num_filters)
    return l

def create_cifar10_model(input, num_classes):
    c_map = [64, 128, 256]

    conv = conv_bn_relu(input, (3, 3), c_map[0])
    r1 = mobilenet_basic_stack(conv, 5, c_map[0])

    r2_1 = mobilenet_basic_inc(r1, c_map[1])
    r2_2 = mobilenet_basic_stack(r2_1, 5, c_map[1])

    r3_1 = mobilenet_basic_inc(r2_2, c_map[2])
    r3_2 = mobilenet_basic_stack(r3_1, 5, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8))(r3_2)
    z = Dense(num_classes, init=he_normal())(pool)
    return z