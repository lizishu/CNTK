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

def Res_A(input, n, m):
    a1 = conv_bn(input, (1,1), n, bn_init_scale = 1)

    b1 = conv_bn(input, (1,1), n, bn_init_scale = 1)
    b2 = conv_bn(b1, (3,3), n, bn_init_scale = 1)

    c1 = conv_bn(input, (1,1), n, bn_init_scale = 1)
    c2 = conv_bn(c1, (3,3), n, bn_init_scale = 1)
    c3 = conv_bn(c2, (3,3), n, bn_init_scale = 1)

    out = splice(a1, b2, c3, axis = 0)
    out2 = conv_bn(out, (1,1), m, bn_init_scale = 1)


    p = out2 + input
    return relu(p)

def Res_B(input, n, m):
    a1 = conv_bn(input, (1,1), n, bn_init_scale = 1)

    b1 = conv_bn(input, (1,1), n, bn_init_scale = 1)

  

    b2 = conv_bn(b1, (1,3), n, bn_init_scale = 1)

   

    b3 = conv_bn(b2, (3,1), n, bn_init_scale = 1)

    

    c = splice(a1, b3, axis = 0)

    d = conv_bn(c, (1,1), m, bn_init_scale = 1)

  

    p = d + input
    return relu(p)

def Res_C(input, n, m):
    A1 = conv_bn(input, (1,1), n, bn_init_scale = 1)
    
    B1 = conv_bn(input, (1,1), n, bn_init_scale = 1)
    B2 = conv_bn(B1, (1,3),n, bn_init_scale = 1)
    B3 = conv_bn(B2, (3,1), n, bn_init_scale = 1)

    C = splice(A1, B3, axis = 0)

    D = conv_bn(C, (1,1), m, bn_init_scale = 1)
    
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

def reduction_B(input, b1, b2, c1, c2, d1):
    A1 = MaxPooling(filter_shape = (3,3), strides = (2,2), pad = True)(input)

    B1 = conv_bn_relu(input, (1,1), b1, (1,1))
    B2 = conv_bn_relu(B1, (3,3), b2, (2,2))

    C1 = conv_bn_relu(input, (1,1), c1, (1,1))
    C2 = conv_bn_relu(C1, (3,3), c2, (2,2))

    D1 = conv_bn_relu(input, (1,1), d1, (1,1))
    D2 = conv_bn_relu(D1, (3,3), d1, (1,1))
    D3 = conv_bn_relu(D2, (3,3), d1, (2,2))

    out = splice(A1, B2, C2, D3, axis = 0)

    return out

def pre_block(input):
    c1 = conv_bn_relu(input, (3,3), 32, (1,1))
    c2 = conv_bn_relu(c1, (3,3), 32, (1,1))
    c3 = conv_bn_relu(c2, (3,3), 64, (1,1))
    c4 = MaxPooling(filter_shape = (3,3), strides = (1,1), pad = True, name = 'pool')(c3)

    c5 = conv_bn_relu(c4, (1,1), 80, (1,1))
    c6 = conv_bn_relu(c5, (3,3), 128, (1,1))
    c7 = conv_bn_relu(c6, (3,3), 128, (1,1))

    return c7

def Res_A_stack(input, num_stack_layers, n, m): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_A(l, n, m)
    return l

def Res_B_stack(input, num_stack_layers, n, m): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_B(l, n, m)
    return l

def Res_C_stack(input, num_stack_layers, n, m): 
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = Res_C(l, n, m)
    return l


def InceptionV4_ResNet_model(input, num_classes):
    l = pre_block(input)

    # 32 x 32 x 128
    A1 = Res_A_stack(l,5, 64, 128)

    RA = reduction_A(A1, 32, 32, 64, 64)

    # 16 x 16 x 224
    B1 = Res_B_stack(RA, 10, 128, 224)

    # 16 x 16 x 448
    RB = reduction_B(B1, 32, 64, 32, 64, 128)

    # 8 x 8 x 480
    C1 = Res_C_stack(RB, 5, 192, 480)

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8), name='final_avg_pooling')(C1)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z
