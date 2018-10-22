from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, splice

def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1,1), pad=True, bnTimeConst=4096, init=he_normal()):
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

def pre_block(input, bnTimeConst):
    # 32 x 32 x 3
    conv1a = conv_bn_relu_layer(input, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1b = conv_bn_relu_layer(conv1a, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1c = conv_bn_relu_layer(conv1b, 64, (3,3), (1,1), True, bnTimeConst)

    c1 = MaxPooling((3,3), strides = (1,1), pad = True)(conv1c)
    c2 = conv_bn_relu_layer(conv1c, 96, (3,3), (1,1), True, bnTimeConst)

    d = splice(c1,c2, axis = 0)
    
    e1 = conv_bn_relu_layer(d, 64, (1,1), (1,1), True, bnTimeConst)
    e2 = conv_bn_relu_layer(e1, 96, (3,3), (1,1), True, bnTimeConst)

    f1 = conv_bn_relu_layer(d, 64, (1,1), (1,1), True, bnTimeConst)
    f2 = conv_bn_relu_layer(f1, 64, (3,1), (1,1), True, bnTimeConst)
    f3 = conv_bn_relu_layer(f2, 64, (1,3), (1,1), True, bnTimeConst)
    f4 = conv_bn_relu_layer(f3, 96, (3,3), (1,1), True, bnTimeConst)

    g = splice(e2, f4, axis = 0)

    h1 = conv_bn_relu_layer(g, 128, (3,3), (1,1), True, bnTimeConst)
    i1 = MaxPooling((3,3), strides = (1,1), pad = True)(g)

    out = splice(h1, i1, axis = 0)

    return out

def Inception_A(input, a1,b1,c1,c2,d1,d2, bnTimeConst):
    A1 = AveragePooling((3,3), strides = (1,1), pad = True)(input)
    A2 = conv_bn_relu_layer(A1, a1, (3,3), (1,1), True, bnTimeConst)

    B1 = conv_bn_relu_layer(input, b1, (1,1), (1,1), True, bnTimeConst)

    C1 = conv_bn_relu_layer(input, c1, (1,1), (1,1), True, bnTimeConst)
    C2 = conv_bn_relu_layer(C1, c2, (3,3), (1,1), True, bnTimeConst)

    D1 = conv_bn_relu_layer(input, d1, (1,1), (1,1), True, bnTimeConst)
    D2 = conv_bn_relu_layer(D1, d2, (3,3), (1,1), True, bnTimeConst)
    D3 = conv_bn_relu_layer(D2, d2, (3,3), (1,1), True, bnTimeConst)

    out = splice(A2, B1, C2, D3, axis = 0)
    return out

def Inception_B(input, a1, b1, c1,c2,c3,d1,d2,d3, bnTimeConst):
    A1 = AveragePooling((3,3), strides = (1,1), pad = True)(input)
    A2 = conv_bn_relu_layer(A1, a1, (1,1), (1,1), True, bnTimeConst)

    B1 = conv_bn_relu_layer(input, b1, (1,1), (1,1), True, bnTimeConst)

    C1 = conv_bn_relu_layer(input, c1, (1,1), (1,1), True, bnTimeConst)
    C2 = conv_bn_relu_layer(C1, c2, (1,3), (1,1), True, bnTimeConst)
    C3 = conv_bn_relu_layer(C2, c3, (1,3), (1,1), True, bnTimeConst)

    D1 = conv_bn_relu_layer(input, d1, (1,1), (1,1), True, bnTimeConst)
    D2 = conv_bn_relu_layer(D1, d1, (1,3), (1,1), True, bnTimeConst)
    D3 = conv_bn_relu_layer(D2, d2, (3,1), (1,1), True, bnTimeConst)
    D4 = conv_bn_relu_layer(D3, d3, (1,3), (1,1), True, bnTimeConst)
    D5 = conv_bn_relu_layer(D4, d3, (3,1), (1,1), True, bnTimeConst)

    out = splice(A2, B1, C3, D5, axis = 0)

    return out

def Inception_C(input, a1, b1, c1, c2, d1,d2,d3,d4, bnTimeConst):
    A1 = AveragePooling((3,3), strides = (1,1), pad = True)(input)
    A2 = conv_bn_relu_layer(A1, a1, (1,1), (1,1), True, bnTimeConst)

    B1 = conv_bn_relu_layer(input, b1, (1,1), (1,1), True, bnTimeConst)

    C1 = conv_bn_relu_layer(input, c1, (1,1), (1,1), True, bnTimeConst)
    C21 = conv_bn_relu_layer(C1, c2, (1,3), (1,1), True, bnTimeConst)
    C22 = conv_bn_relu_layer(C1, c2, (3,1), (1,1), True, bnTimeConst)

    D1 = conv_bn_relu_layer(input, d1, (1,1), (1,1), True, bnTimeConst)
    D2 = conv_bn_relu_layer(D1, d2, (1,3), (1,1), True, bnTimeConst)
    D3 = conv_bn_relu_layer(D2, d3, (3,1), (1,1), True, bnTimeConst)
    D41 = conv_bn_relu_layer(D3, d4, (3,1), (1,1), True, bnTimeConst)
    D42 = conv_bn_relu_layer(D3, d4, (1,3), (1,1), True, bnTimeConst)

    out = splice(A2, B1, C21, C22, D41, D42, axis = 0)

    return out

def reduction_A(input, b1, c1, c2, c3, bnTimeConst):
    A1 = MaxPooling((3,3), strides = (2,2), pad = True)(input)

    B1 = conv_bn_relu_layer(input, b1, (3,3), (2,2), True, bnTimeConst)

    C1 = conv_bn_relu_layer(input, c1, (1,1), (1,1), True, bnTimeConst)
    C2 = conv_bn_relu_layer(C1, c2, (3,3), (1,1), True, bnTimeConst)
    C3 = conv_bn_relu_layer(C2, c3, (3,3), (2,2), True, bnTimeConst)

    out = splice(A1, B1, C3, axis = 0)

    return out

def reduction_B(input, b1, c1, c2, bnTimeConst):
    A1 = MaxPooling((3,3), strides = (2,2), pad = True)(input)

    B1 = conv_bn_relu_layer(input, b1, (1,1), (1,1), True, bnTimeConst)
    B2 = conv_bn_relu_layer(B1, b1, (3,3), (2,2), True, bnTimeConst)

    C1 = conv_bn_relu_layer(input, c1, (1,1), (1,1), True, bnTimeConst)
    C2 = conv_bn_relu_layer(C1, c1, (1,3), (1,1), True, bnTimeConst)
    C3 = conv_bn_relu_layer(C2, c2, (3,1), (1,1), True, bnTimeConst)
    C4 = conv_bn_relu_layer(C3, c2, (3,3), (2,2), True, bnTimeConst)

    out = splice(A1, B2, C4, axis = 0)

    return out


def inception_v4_model(input, labelDim, bnTimeConst):
    l = pre_block(input, bnTimeConst)

    # 32 x 32
    A1 = Inception_A(l, 32, 32, 32, 64, 32, 64, bnTimeConst)
    A2 = Inception_A(A1, 32, 32, 32, 64, 32, 64, bnTimeConst)
    A3 = Inception_A(A2, 32, 32, 32, 64, 32, 64, bnTimeConst)
    A4 = Inception_A(A3, 32, 32, 32, 64, 32, 64, bnTimeConst)

    # 32 x 32 x 192
    RA = reduction_A(A4, 32, 32, 64, 64, bnTimeConst)

    # 16 x 16 x 288
    B1 = Inception_B(RA, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B2 = Inception_B(B1, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B3 = Inception_B(B2, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B4 = Inception_B(B3, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B5 = Inception_B(B4, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B6 = Inception_B(B5, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)
    B7 = Inception_B(B6, 128, 32, 32, 64, 96, 32, 64, 96, bnTimeConst)

    # 16 x 16 x 352
    RB = reduction_B(B7,64, 64,96, bnTimeConst)

    # 8 x 8 x 512
    C1 = Inception_C(RB, 128, 128, 96, 64, 64, 128, 160, 64, bnTimeConst)
    C2 = Inception_C(C1, 128, 128, 96, 64, 64, 128, 160, 64, bnTimeConst)
    C3 = Inception_C(C2, 128, 128, 96, 64, 64, 128, 160, 64, bnTimeConst)

    # 8 x 8 x 512
    pool1 = AveragePooling(filter_shape=(8,8))(C3)

    # 1 x 1 x 512
    z = Dense(labelDim, init=he_normal())(pool1)

    return z
