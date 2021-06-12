#!/usr/bin/env python
# encoding: utf-8

'''
@author: Zeqi@@
@file: Mobilenetv2_SSDLite.py
@time: 4/12/20 0:37
'''


import sys
# sys.path.append('../../Kitti-Object-Detection')
sys.path.append('../Models')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.getcwd())

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.initializers import Constant
from Models.anchor_layer import PriorBox
# from ads.Custom_layers.spike_counting_layer import spike_counter
from Models.prediction_load_layer import prediction_layer



def _depthwise_conv_block(inputs,
                          add_bias,
                          add_bn,
                          pointwise_conv_filters,
                          is_relu6,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_name='loc'):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=strides,
                        use_bias=add_bias,
                        kernel_initializer='he_normal',
                        bias_initializer=Constant(0.),
                        name='conv_dw_{}'.format(block_name))(inputs)

    if add_bn:
        x = BatchNormalization(name='conv_dw_{}_bn'.format(block_name))(x)

    if is_relu6:
        x = ReLU(max_value=6, name='conv_dw_{}_relu'.format(block_name))(x)
    else:
        x = ReLU(name='conv_dw_{}_relu'.format(block_name))(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               strides=(1, 1),
               use_bias=add_bias,
               kernel_initializer='he_normal',
               bias_initializer=Constant(0.),
               name='conv_pw_{}'.format(block_name))(x)
    return x

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    # Get input size
    img_dim = 2 if K.image_data_format() == 'channels_first' else 1
    input_size = K.int_shape(inputs)[img_dim:(img_dim + 2)]

    # Get kernel size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _make_divisible(v, divisor, layer_name, min_value=None):
    '''It ensures that all layers have a channel number that is divisible by 8'''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    # print('{3}, Original v: {0}, divisor: {1}, new_v: {2}'.format(v, divisor, new_v, layer_name))
    return new_v

def _inverted_res_block(inputs,
                        expansion,
                        stride,
                        alpha,
                        filters,
                        add_bias,
                        add_bn,
                        is_relu6,
                        block_id=1):
    """
     separable_conv2d
        3*3 depyhwise conv + bn + relu6  +  1*1 pointwise conv + bn + relu6
     expanded_conv
        1*1 expansion + depthwise + 1*1 projection
    """
    block_id_header_list = [5, 12]
    prefix = 'block_{}_'.format(block_id)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8, layer_name=prefix)
    x = inputs


    if block_id:
        # block_id is 0, keep dimension, expansion = 1
        # block_id is not 0, raise dimension, expansion = 6
        # Expand
        if isinstance(expansion, (float)):
            print('Extra layers {}'.format(int(expansion * in_channels)))
        x_out = Conv2D(int(expansion * in_channels),
                                  kernel_size=1,
                                  padding='same',
                                  use_bias=add_bias,
                                  activation=None,
                                  kernel_initializer='he_normal',
                                  bias_initializer=Constant(0.),
                                  name='conv_pw_%d_expand'%block_id)(x)
        if add_bn:
            x = BatchNormalization(axis=channel_axis,
                                  # epsilon=1e-3,
                                  # momentum=0.999,
                                  name='conv_pw_%d_expand_bn'%block_id)(x_out)
        else:
            x = x_out

        if is_relu6:
            x = ReLU(6., name='conv_pw_%d_expend_relu' % block_id)(x)
        else:
            x = ReLU(name='conv_pw_%d_expend_relu' % block_id)(x)

        if block_id in block_id_header_list:
            expand_out = x_out
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                                 name= 'zero_pad_%d'%block_id)(x)
    x = DepthwiseConv2D(kernel_size=3,
                       strides=stride,
                       activation=None,
                       use_bias=add_bias,
                       padding='same' if stride == 1 else 'valid',
                       kernel_initializer='he_normal',
                       bias_initializer=Constant(0.),
                       name='conv_dw_%d' % block_id)(x)
    if add_bn:
        x = BatchNormalization(axis=channel_axis,
                              # epsilon=1e-3,
                              # momentum=0.999,
                              name='conv_dw_%d_bn' % block_id)(x)
    if is_relu6:
        x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
    else:
        x = ReLU(name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x = Conv2D(pointwise_filters,
              kernel_size=1,
              padding='same',
              use_bias=add_bias,
              activation=None,
              kernel_initializer='he_normal',
              bias_initializer=Constant(0.),
              name='conv_pw_%d_project'%block_id)(x)
    if add_bn:
            x = BatchNormalization(axis=channel_axis,
                                   # epsilon=1e-3,
                                   # momentum=0.999,
                                   name='conv_pw_%d_project_bn'%block_id)(x)

    if in_channels == pointwise_filters and stride == 1:
        if block_id in block_id_header_list:
            return Add(name='add_%d' % block_id)([inputs, x]), expand_out
        return Add(name='add_%d'%block_id)([inputs, x])
    return x


def _inverted_res_block_extend(inputs,
                               filter_num,
                                stride,
                                add_bias,
                                add_bn,
                                block_id=1):
    """
     separable_conv2d
        3*3 depyhwise conv + bn + relu6  +  1*1 pointwise conv + bn + relu6
     expanded_conv
        1*1 expansion + depthwise + 1*1 projection
    """
    # block_id_header_list = [5, 12]
    # prefix = 'block_{}_'.format(block_id)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    #in_channels = K.int_shape(inputs)[channel_axis]
    # pointwise_conv_filters = int(filters * alpha)
    # pointwise_filters = _make_divisible(pointwise_conv_filters, 8, layer_name=prefix)
    x = inputs


    x = Conv2D(int(filter_num//2),
              kernel_size=1,
              padding='same',
              use_bias=add_bias,
              activation=None,
              kernel_initializer='he_normal',
              bias_initializer=Constant(0.),
              name='conv_pw_%d_expand'%block_id)(x)
    if add_bn:
        x = BatchNormalization(axis=channel_axis,
                              # epsilon=1e-3,
                              # momentum=0.999,
                              name='conv_pw_%d_expand_bn'%block_id)(x)

    x = ReLU(name='conv_pw_%d_expend_relu' % block_id)(x)


    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(x, 3),
                                 name= 'zero_pad_%d'%block_id)(x)

    x = DepthwiseConv2D(kernel_size=3,
                       strides=stride,
                       activation=None,
                       use_bias=add_bias,
                       padding='same' if stride == 1 else 'valid',
                       kernel_initializer='he_normal',
                       bias_initializer=Constant(0.),
                       name='conv_dw_%d' % block_id)(x)

    if add_bn:
        x = BatchNormalization(axis=channel_axis,
                              # epsilon=1e-3,
                              # momentum=0.999,
                              name='conv_dw_%d_bn' % block_id)(x)

    x = ReLU(name='conv_dw_%d_relu' % block_id)(x)

    # Project
    x_out = Conv2D(filter_num,
              kernel_size=1,
              padding='same',
              use_bias=add_bias,
              activation=None,
              kernel_initializer='he_normal',
              bias_initializer=Constant(0.),
              name='conv_pw_%d_project'%block_id)(x)
    if add_bn:
        x = BatchNormalization(axis=channel_axis,
                               # epsilon=1e-3,
                               # momentum=0.999,
                               name='conv_pw_%d_project_bn'%block_id)(x_out)
    else:
        x = x_out

    x = ReLU(name='conv_pw_%d_relu' % block_id)(x)

    return x, x_out



def conv2d_bn(conv_ind,
              x,
              filters,
              num_row,
              num_col,
              padding='same',
              stride=1,
              dilation_rate=1,
              relu=True,
              is_relu6=True,
              add_bias=False,
              add_bn=False):

    x = Conv2D(filters, (num_row, num_col),
                strides=(stride, stride),
                padding=padding,
                dilation_rate=(dilation_rate, dilation_rate),
                use_bias=add_bias,
                bias_initializer=Constant(0.),
                kernel_initializer='he_normal',
                name='conv2d_%d'%conv_ind)(x)
    if add_bn:
        x = BatchNormalization(name='batch_normalization_%d'%conv_ind)(x)
    if relu:
        if is_relu6:
            x = ReLU(max_value=6)(x)
        else:
            x = ReLU()(x)
    return x


def mobilenetv2(input_tensor, add_bias, add_bn, is_relu6=True, alpha=1.0):
    net = {}
    # Block 1
    # 160, 480, 3 -> 160, 480, 3
    x = input_tensor

    first_block_filters = _make_divisible(32 * alpha, 8, layer_name='First_block_filters')
    # After 3*3 convolution, height/width is half of the previous
    x = ZeroPadding2D(padding=correct_pad(input_tensor, 3),
                      name='conv1_pad')(x)

    # 160, 480, 3 -> 80, 240, 32
    x = Conv2D(first_block_filters,
              kernel_size=3,
              strides=(2, 2),
              padding='valid',
              use_bias=add_bias,
              kernel_initializer='he_normal',
              bias_initializer=Constant(0.),
              name='conv1')(x)

    if add_bn:
        x = BatchNormalization(# epsilon=1e-3,
                               # momentum=0.999,
                               name='conv1_bn')(x)
    if is_relu6:
        x = ReLU(max_value=6, name='conv1_relu')(x)
    else:
        x = ReLU(name='conv1_relu')(x)

    # 80, 240, 32 -> 80, 240, 16
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=1, block_id=0)

    # 80, 240, 16 -> 80, 240, 96 -> 40, 120, 96 -> 40, 120, 24 (no add)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=1)
    # 40, 120, 24 -> 40, 120, 144 -> 40, 120, 24
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=2)

    #  40, 120, 24- > 40, 120, 32 -> 40, 120, 144 -> 20, 60, 144 -> 20, 60, 32(no add)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=3)
    # 20, 60, 32 -> 20, 60, 192 -> 20, 60, 32
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=4)
    # 20, 60, 32 -> 20, 60, 192 -> 20, 60, 32
    x, expand_out_1 = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=5)
    net['conv4_3'] = expand_out_1  # v1 20,60,256

    # 20, 60, 32 -> 20, 60, 64 -> 20, 60, 192 -> 10, 30, 192 -> 10, 30, 64 (no add)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=6)
    # 10, 30, 64 -> 10, 30, 384 -> 10, 30, 64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=7)
    # 10, 30, 64 -> 10, 30, 384 -> 10, 30, 64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=8)
    # 10, 30, 64 -> 10, 30, 384 -> 10, 30, 64
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=9)

    #  10, 30, 64 -> 10, 30, 384 -> 10, 30, 96 (no add)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=10)
    #  10, 30, 96 -> 10, 30, 576 -> 10, 30, 96
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=11)
    #  10, 30, 96 -> 10, 30, 576 -> 10, 30, 96
    x, expand_out_2 = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=12)
    net['fc7'] = expand_out_2  # v1 10,30,1024  first port: 576

    # x = Dropout(0.5, name='drop7')(x)

    # Block 6
    # 10, 30, 96 ->  10, 30, 576 -> 5, 15, 576 -> 5, 15, 160 (no add)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=13)
    # 5, 15, 160 -> 5, 15, 960 -> 5, 15, 160
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=14)
    # 5, 15, 160 -> 5, 15, 960 -> 5, 15, 160
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=15)
    # 5, 15, 160 -> 5, 15, 960 -> 5, 15, 320  (no add)
    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, add_bias=add_bias,
                            add_bn=add_bn, is_relu6=is_relu6, expansion=6, block_id=16)

    # 5, 15, 320 -> 5, 15, 1280
    last_block_filters = 1280
    x_out = Conv2D(last_block_filters,
              kernel_size=1,
              use_bias=add_bias,
              kernel_initializer='he_normal',
              bias_initializer=Constant(0.),
              name='conv_1')(x)
    if add_bn:
        x = BatchNormalization(# epsilon=1e-3,
                               # momentum=0.999,
                               name='conv_1_bn')(x_out)
    else:
        x = x_out

    if is_relu6:
        x = ReLU(max_value=6, name='conv_1_relu')(x)
    else:
        x = ReLU(name='conv_1_relu')(x)

    net['conv6_2'] = x_out

    ####################################### Extra #######################################
    # Block 7
    #  5, 15, 1280 -> 3, 8, 512
    x, net['conv7_2'] = _inverted_res_block_extend(x, filter_num=512,
                                                 stride=2,
                                                 add_bias=add_bias,
                                                 add_bn=add_bn,
                                                 block_id=17)

    # Block 8
    # 3, 8, 256 -> 2, 4, 256
    x, net['conv8_2'] = _inverted_res_block_extend(x, filter_num=256,
                                                 stride=2,
                                                 add_bias=add_bias,
                                                 add_bn=add_bn,
                                                 block_id=18)

    # Block 9
    # 2, 4, 256 -> 1, 2, 256
    x, net['conv9_1'] = _inverted_res_block_extend(x, filter_num=128,
                                                 stride=2,
                                                 add_bias=add_bias,
                                                 add_bn=add_bn,
                                                 block_id=19)
    # 1, 2, 256 -> 1, 1, 64
    x, net['conv9_2'] = _inverted_res_block_extend(x, filter_num=128,
                                                 stride=2,
                                                 add_bias=add_bias,
                                                 add_bn=add_bn,
                                                 block_id=20)
    return net


def mobilenetv2_ssd(input_shape,
                    is_relu6=True,
                    num_classes=9,
                    add_encoder_bias=True,
                    add_decoder_bias=True,
                    add_bn=False,
                    pretrained=True,
                    structured=True):

    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # Backbone
    net = mobilenetv2(input_tensor, add_bias=add_encoder_bias, add_bn=add_bn, is_relu6=is_relu6)

    with_bias = add_decoder_bias
    ################################################## ##################################################
    net['conv4_3_norm'] = net['conv4_3']
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整 20, 60
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4,
                                          kernel_size=(3, 3),
                                          kernel_initializer='he_normal',
                                          padding='same',
                                          use_bias=with_bias,
                                          name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])

    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])

    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes,
                                           kernel_size=(3, 3),
                                           kernel_initializer='he_normal',
                                           padding='same',
                                           use_bias=with_bias,
                                           name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])

    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    priorbox = PriorBox(img_size, 10.0, max_size=21.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    ################################################## ##################################################

    ################################################## ##################################################
    # 对fc7层进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4,
                                 kernel_size=(3, 3),
                                 kernel_initializer='he_normal',
                                 padding='same',
                                 use_bias=with_bias,
                                 name='fc7_mbox_loc')(
        net['fc7'])

    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])

    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes,
                                  kernel_size=(3, 3),
                                  kernel_initializer='he_normal',
                                  padding='same',
                                  use_bias=with_bias,
                                  name='fc7_mbox_conf')(net['fc7'])

    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    priorbox = PriorBox(img_size, 21.0, max_size=45.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    ################################################## ##################################################

    ################################################## ##################################################
    # 对conv6_2进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv6_2_mbox_loc'] = Conv2D(num_priors * 4,
                                     kernel_size=(3, 3),
                                     kernel_initializer='he_normal',
                                     padding='same',
                                     use_bias=with_bias,
                                     name='conv6_2_mbox_loc')(net['conv6_2'])

    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv6_2_mbox_conf'] = Conv2D(num_priors * num_classes,
                                      kernel_size=(3, 3),
                                      kernel_initializer='he_normal',
                                      padding='same',
                                      use_bias=with_bias,
                                      name='conv6_2_mbox_conf')(net['conv6_2'])

    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])

    priorbox = PriorBox(img_size, 45.0, max_size=99.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    ################################################## ##################################################

    ################################################## ##################################################
    # 对conv7_2进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv7_2_mbox_loc'] = Conv2D(num_priors * 4,
                                     kernel_size=(3, 3),
                                     kernel_initializer='he_normal',
                                     padding='same',
                                     use_bias=with_bias,
                                     name='conv7_2_mbox_loc')(net['conv7_2'])

    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])

    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    x = Conv2D(num_priors * num_classes,
               kernel_size=(3, 3),
               kernel_initializer=Constant(1.),
               padding='same',
               use_bias=with_bias,
               name='conv7_2_mbox_conf')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x

    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])

    priorbox = PriorBox(img_size, 99.0, max_size=153.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    ################################################## ##################################################

    ################################################## ##################################################
    # 对conv8_2进行处理
    num_priors = 6
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv8_2_mbox_loc'] = Conv2D(num_priors * 4,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     use_bias=with_bias,
                                     name='conv8_2_mbox_loc')(net['conv8_2'])

    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])

    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv8_2_mbox_conf'] = Conv2D(num_priors * num_classes,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      use_bias=with_bias,
                                      name='conv8_2_mbox_conf')(
        net['conv8_2'])

    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])

    priorbox = PriorBox(img_size, 153.0, max_size=207.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    ################################################## ##################################################

    ################################################## ##################################################
    # 对conv9_2进行处理
    num_priors = 4
    # 预测框的处理
    # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整
    net['conv9_2_mbox_loc'] = Conv2D(num_priors * 4,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     use_bias=with_bias,
                                     name='conv9_2_mbox_loc')(net['conv9_2'])

    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # num_priors表示每个网格点先验框的数量，num_classes是所分的类
    net['conv9_2_mbox_conf'] = Conv2D(num_priors * num_classes,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      use_bias=with_bias,
                                      name='conv9_2_mbox_conf')(
        net['conv9_2'])

    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])

    priorbox = PriorBox(img_size, 207.0, max_size=261.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')

    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])
    ################################################## ##################################################

    if structured:
        net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                                       net['fc7_mbox_loc_flat'],
                                       net['conv6_2_mbox_loc_flat'],
                                       net['conv7_2_mbox_loc_flat']],
                                      axis=1, name='mbox_loc')
        net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                                        net['fc7_mbox_conf_flat'],
                                        net['conv6_2_mbox_conf_flat'],
                                        net['conv7_2_mbox_conf_flat']],
                                       axis=1, name='mbox_conf')
        net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                            net['fc7_mbox_priorbox'],
                                            net['conv6_2_mbox_priorbox'],
                                            net['conv7_2_mbox_priorbox']],
                                           axis=1, name='mbox_priorbox')
    else:
        net['mbox_loc'] = concatenate([net['conv4_3_norm_mbox_loc_flat'],
                                       net['fc7_mbox_loc_flat'],
                                       net['conv6_2_mbox_loc_flat'],
                                       net['conv7_2_mbox_loc_flat'],
                                       net['conv8_2_mbox_loc_flat'],
                                       net['conv9_2_mbox_loc_flat']],
                                      axis=1, name='mbox_loc')
        net['mbox_conf'] = concatenate([net['conv4_3_norm_mbox_conf_flat'],
                                        net['fc7_mbox_conf_flat'],
                                        net['conv6_2_mbox_conf_flat'],
                                        net['conv7_2_mbox_conf_flat'],
                                        net['conv8_2_mbox_conf_flat'],
                                        net['conv9_2_mbox_conf_flat']],
                                       axis=1, name='mbox_conf')
        net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                            net['fc7_mbox_priorbox'],
                                            net['conv6_2_mbox_priorbox'],
                                            net['conv7_2_mbox_priorbox'],
                                            net['conv8_2_mbox_priorbox'],
                                            net['conv9_2_mbox_priorbox']],
                                           axis=1, name='mbox_priorbox')

    if hasattr(net['mbox_loc'], 'shape'):
        num_boxes = net['mbox_loc'].shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4

    net['mbox_loc'] = Reshape((num_boxes, 4), name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    # net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
    net['mbox_conf'] = Softmax(name='mbox_conf_final', dtype=tf.float32)(net['mbox_conf'])


    net['predictions'] = concatenate([net['mbox_loc'],
                                      net['mbox_conf'],
                                      net['mbox_priorbox']],
                                     axis=2, name='predictions')

    net['predictions'] = prediction_layer()(net['predictions'])

    model = Model(input_tensor, net['predictions'], name='MobileNetV1-SSD')
    # print('No anchor files')

    if pretrained:
        model = load_mobilenetv2_ssd(model)

    return model

def load_mobilenetv2_ssd(model):
    for layer in model.layers:
        print(layer.name)

    tf_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights='imagenet')
    mobilenetv2_1_224 = tf_model.get_weights()

    model_w = model.get_weights()

    for i, w in enumerate(model_w):
        model_w[i] = mobilenetv2_1_224[i]
        print('{}, SSD: {}, mobilenetv2: {}'.format(i, np.shape(w), np.shape(mobilenetv2_1_224[i])))
        if i >= len(mobilenetv2_1_224) - 1:
            break

    model.set_weights(model_w)
    return model

if __name__ == '__main__':
    # Mobilenet v2 extractor
    """
        Total params: 2,936,768
        Trainable params: 2,897,920
        Non-trainable params: 38,848
    """
    input_tensor = Input(shape=(160, 480, 3))
    net = mobilenetv2(input_tensor, add_bias=False, add_bn=False, is_relu6=True)
    mbv2_model = Model(input_tensor, net['conv9_2'])
    # mbv2_model.summary()

    # Mobilenet v2 SSD detector
    """
        Total params: 3,234,624
        Trainable params: 3,184,256
        Non-trainable params: 50,368
    """
    model = mobilenetv2_ssd(input_shape=(160, 480, 3),
                            is_relu6=True,
                            num_classes=9,
                            add_encoder_bias=False,
                            add_decoder_bias=True,
                            add_bn=True,
                            pretrained=True)
    model.summary()




    # https://github.com/tanakataiki/ssd_kerasV2/blob/master/model/ssd300MobileNetV2Lite.py