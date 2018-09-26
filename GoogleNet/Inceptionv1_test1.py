# UPD 2018.9.22 net
from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, splice

def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1,1), pad=True, bnTimeConst=4096, init=he_normal()):
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

def inception_block_with_avgpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # 1x1 -> 3x3 -> 3x3
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # avg pooling -> 1x1
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out
    
# output channls : num1x1 + num3x3 + num3x3dbl + numPool
def inception_block_with_maxpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):

    # 1x1 
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 1x1 -> 3x3 
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # 1x1 -> 3x3 -> 3x3
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # max pooling -> 1x1 
    branchPool_maxpool = MaxPooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_maxpool, numPool, (1,1), (1,1), True, bnTimeConst)
    
    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

def inception_block_pass_through(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    
    # 1x1 -> 3x3  branch
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (2,2), True, bnTimeConst)

    # 1x1 -> 3x3 -> 3x3 branch
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (2,2), True, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=True)(input)

    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)

    return out


def inceptionv1_cifar_model2(input, labelDim, bnTimeConst):

    # 32 x 32 x 3
    conv1 = conv_bn_relu_layer(input, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv2 = conv_bn_relu_layer(conv1, 32, (3,3), (1,1), True, bnTimeConst)

    # Inception Blocks
    # 32 x 32 x 64
    inception3a = inception_block_with_maxpool(conv2, 32, 32, 32, 32, 48, 16, bnTimeConst)
    # 32 x 32 x 128
    inception3b = inception_block_with_maxpool(inception3a, 32, 32, 32, 32, 48, 16, bnTimeConst)

    maxpool1 = MaxPooling((3,3), strides = (2,2), pad = True)(inception3b)

    # 16 x 16 x 128
    inception4a = inception_block_with_maxpool(maxpool1, 96, 48, 64, 48, 64, 64, bnTimeConst)
    # 16 x 16 x 288
    inception4b = inception_block_with_maxpool(inception4a, 96, 48, 64, 48, 64, 64, bnTimeConst)
    # 16 x 16 x 288
    inception4c = inception_block_with_maxpool(inception4b, 96, 48, 64, 48, 64, 64, bnTimeConst)
    # 16 x 16 x 288
    inception4d = inception_block_with_maxpool(inception4c, 96, 48, 64, 48, 64, 64, bnTimeConst)
    # 16 x 16 x 288
    inception4e = inception_block_with_maxpool(inception4d, 96, 48, 64, 48, 64, 64, bnTimeConst)

    maxpool2 = MaxPooling((3,3), strides = (2,2), pad = True)(inception4e)

    # 8 x 8 x 288
    inception5a = inception_block_with_maxpool(inception4e, 176, 96, 160, 96, 112, 64, bnTimeConst)
    # 8 x 8 x 512
    inception5b = inception_block_with_maxpool(inception5a, 176, 96, 160, 96, 112, 64, bnTimeConst)

    # Global Average
    # 8 x 8 x 512
    pool1 = AveragePooling(filter_shape=(8,8))(inception5b)
    # 1 x 1 x 512

    z = Dense(labelDim, init=he_normal())(pool1)

    return z


