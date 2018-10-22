import cntk as C
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

    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # Average Pooling
    branchPool_avgpool = AveragePooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1,1), (1,1), True, bnTimeConst)

    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out
    
def inception_block_with_maxpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):

    # 1x1 Convolution
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1,1), (1,1), True, bnTimeConst)

    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (1,1), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (1,1), True, bnTimeConst)

    # Max Pooling
    branchPool_maxpool = MaxPooling((3,3), strides=(1,1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_maxpool, numPool, (1,1), (1,1), True, bnTimeConst)
    
    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)

    return out

def inception_block_pass_through(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    
    # 3x3 Convolution
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1,1), (1,1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3,3), (2,2), True, bnTimeConst)

    # Double 3x3 Convolution
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1,1), (1,1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3,3), (1,1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3,3), (2,2), True, bnTimeConst)

    # Max Pooling
    branchPool = MaxPooling((3,3), strides=(2,2), pad=True)(input)

    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)

    return out


def bn_inception_cifar_model(input, labelDim, bnTimeConst):
    # 32 x 32 x 3
    conv1a = conv_bn_relu_layer(input, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1b = conv_bn_relu_layer(conv1a, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv1c = conv_bn_relu_layer(conv1b, 32, (3,3), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv2a = conv_bn_relu_layer(conv1c, 32, (1,1), (1,1), True, bnTimeConst)
    # 32 x 32 x 32
    conv2b = conv_bn_relu_layer(conv2a, 64, (3,3), (1,1), True, bnTimeConst)
    
    # Inception Blocks
    # 32 x 32 x 64
    inception3a = inception_block_with_avgpool(conv2b, 32, 32, 32, 32, 48, 16, bnTimeConst)
    # 32 x 32 x 128
    inception3b = inception_block_pass_through(inception3a, 0, 64, 80, 32, 48, 0, bnTimeConst) 
    # 16 x 16 x 256
    inception4a = inception_block_with_avgpool(inception3b, 96, 48, 64, 48, 64, 64, bnTimeConst) 
    # 16 x 16 x 288
    inception4b = inception_block_with_avgpool(inception4a, 48, 64, 96, 80, 96, 64, bnTimeConst) 

    inception4c = inception_block_with_avgpool(inception4b, 48, 64, 96, 80, 96, 64, bnTimeConst)

    # 16 x 16 x 288
    inception4d = inception_block_pass_through(inception4c, 0, 128, 192, 192, 256, 0, bnTimeConst)
    # 8 x 8 x 512
    inception5a = inception_block_with_maxpool(inception4d, 176, 96, 160, 96, 112, 64, bnTimeConst) 
    
    # Global Average
    # 8 x 8 x 512
    pool1 = AveragePooling(filter_shape=(8,8))(inception5a)
    # 1 x 1 x 512
    z = Dense(labelDim, init=he_normal())(pool1)

    return z