import numpy as np
import cntk as C
from cntk.initializer import he_normal, normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, splice

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

def Res_A(input, a1, b1, c1, c2, c3, d1):
    A1 = conv_bn_relu(input, (1,1), a1)

    B1 = conv_bn(input, (1,1), b1, bn_init_scale = 1)
    B2 = conv_bn(B1, (3,3), b1, bn_init_scale = 1)

    C1 = conv_bn(input, (1,1), c1, bn_init_scale = 1)
    C2 = conv_bn(C1, (3,3), c2, bn_init_scale = 1)
    C3 = conv_bn(C2, (3,3), c3, bn_init_scale = 1)

    out = splice(A1, B2, C3, axis = 0)
    out2 = conv_bn(out, (1,1), d1, bn_init_scale = 1)

    p = out2 + input
    return relu(p)

def Res_B(input, a1, b1, b2, b3, c1):
    A1 = conv_bn(input, (1,1), a1, bn_init_scale = 1)

    B1 = conv_bn(input, (1,1), b1, bn_init_scale = 1)
    B2 = conv_bn(B1, (1,3), b2, bn_init_scale = 1)
    B3 = conv_bn(B2, (3,1), b3, bn_init_scale = 1)

    C = splice(A1, B3, axis = 0)

    D = conv_bn(C, (1,1), c1, bn_init_scale = 1)

    p = D + input
    return relu(p)

def Res_C(input, a1, b1, b2, b3, c1):
    A1 = conv_bn(input, (1,1), a1, bn_init_scale = 1)
    
    B1 = conv_bn(input, (1,1), b1, bn_init_scale = 1)
    B2 = conv_bn(B1, (1,3), b2, bn_init_scale = 1)
    B3 = conv_bn(B2, (3,1), b3, bn_init_scale = 1)

    C = splice(A1, B3, axis = 0)

    D = conv_bn(C, (1,1), c1, bn_init_scale = 1)
    
    p = D + input
    return relu(p)

def reduction_A(input, b1, c1, c2, c3):
    A1 = MaxPooling((3,3), strides = (2,2), pad = True)(input)

    B1 = conv_bn_relu(input, (3,3), b1, (2,2))

    C1 = conv_bn_relu(input, (1,1), c1, (1,1))
    C2 = conv_bn_relu(C1, (3,3), c2, (1,1))
    C3 = conv_bn_relu(C2, (3,3), c3, (2,2))

    out = splice(A1, B1, C3, axis = 0)

    return out


def reduction_B(input, b1, b2, c1, c2, d1, d2, d3):
    A1 = MaxPooling(filter_shape = (3,3), strides = (2,2), pad = True)(input)

    B1 = conv_bn_relu(input, (1,1), b1, (1,1))
    B2 = conv_bn_relu(B1, (3,3), b2, (2,2))

    C1 = conv_bn_relu(input, (1,1), c1, (1,1))
    C2 = conv_bn_relu(C1, (3,3), c2, (2,2))

    D1 = conv_bn_relu(input, (1,1), d1, (1,1))
    D2 = conv_bn_relu(D1, (3,3), d2, (1,1))
    D3 = conv_bn_relu(D2, (3,3), d3, (2,2))

    out = splice(A1, B2, C2, D3, axis = 0)

    return out

def pre_block(input):
    # 32 x 32 x 3
    conv1a = conv_bn_relu(input, (3,3), 16, (1,1))
    # 32 x 32 x 32
    conv1b = conv_bn_relu(conv1a, (3,3), 16, (1,1))
    # 32 x 32 x 32
    conv1c = conv_bn_relu(conv1b, (3,3), 16, (1,1))

    c1 = MaxPooling((3,3), strides = (1,1), pad = True)(conv1c)
    c2 = conv_bn_relu(conv1c, (3,3), 16, (1,1))
    
    d = splice(c1,c2, axis = 0)
    
    # 32 x 32 x 32
    e1 = conv_bn_relu(d, (1,1), 32, (1,1))
    e2 = conv_bn_relu(e1, (3,3), 32, (1,1))

    f1 = conv_bn_relu(d, (1,1), 32, (1,1))
    f2 = conv_bn_relu(f1, (3,1), 32, (1,1))
    f3 = conv_bn_relu(f2, (1,3), 32,(1,1))
    f4 = conv_bn_relu(f3, (3,3), 32, (1,1))

    g = splice(e2, f4, axis = 0)
    # 32 x 32 x 64
    h1 = conv_bn_relu(g, (3,3), 64, (1,1))
    i1 = MaxPooling((3,3), strides = (1,1), pad = True)(g)

    out = splice(h1, i1, axis = 0)
    # 32 x 32 x 128

    return out


def Res_A_stack(input, num_stack_layers, a1, b1, c1, c2, c3, d1): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_A(l, a1, b1, c1, c2, c3, d1)
    return l

def Res_B_stack(input, num_stack_layers, a1, b1, b2, b3, c1): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_B(l, a1, b1, b2, b3, c1)
    return l

def Res_C_stack(input, num_stack_layers, a1, b1, b2, b3, c1): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_C(l, a1, b1, b2, b3, c1)
    return l


def Inception_ResNet_v2_model(input, num_classes):
    l = pre_block(input)

    # 32 x 32 x 128
    A1 = Res_A_stack(l, 5, 32, 32, 32, 48, 64, 128)

    # 32 x 32 x 128
    RA = reduction_A(A1, 64, 32, 64, 64)

    # 16 x 16 x 256
    B1 = Res_B_stack(RA, 10, 128, 64, 96, 128, 256)

    # 16 x 16 x 256
    RB = reduction_B(B1, 32, 64, 32, 64, 32, 64, 128)

    # 8 x 8 x 512
    C1 = Res_C_stack(RB, 5, 128, 64, 128, 256, 512)

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8))(C1)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z
