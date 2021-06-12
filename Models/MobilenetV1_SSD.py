#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 1/11/21 3:01 PM
@fileName  : MobilenetV1_SSD.py
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('../Models')

import logging
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from Models.anchor_layer import PriorBox

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('MobilenetV1-SSD')


class Net_Backbone():
    def __init__(self, with_bias, no_bn,**kwargs):

        self.with_bias = with_bias
        self.no_bn = no_bn
        self.is_relu6 = True,
        self.alpha = 1.0

    def conv2d_bn(self,
                  inputs,
                  filters,
                  kernel_size=(3, 3),
                  padding='same',
                  stride=(1, 1),
                  conv_ind=1):

        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation_rate=(1, 1),
            use_bias=self.with_bias,
            kernel_initializer='he_normal', 
            bias_initializer=Constant(0.),
            name='conv2d_%d' % conv_ind)(inputs)

        if not self.no_bn:
            # x = BatchNormalization(name='batch_normalization_%d' % conv_ind)(x)
            x = BatchNormalization(name='conv2d_%d_bn' % conv_ind)(x)


        if self.is_relu6:
            # print('conv2d_%d_relu6'%conv_ind)
            x = ReLU(max_value=6, name='conv2d_%d_relu6'%conv_ind)(x)
        else:
            x = ReLU()(x)
        
        return x

    def _depthwise_conv_block(self,
                              inputs,
                              pointwise_conv_filters,
                              strides=(1, 1),
                              block_name=1):

        x = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=strides,
                            use_bias=self.with_bias,
                            depthwise_initializer='he_normal', 
                            bias_initializer=Constant(0.),
                            name='conv_dw_%d'% block_name)(inputs)

        if not self.no_bn:
            x = BatchNormalization(name='conv_dw_%d_bn'% block_name)(x)

        if self.is_relu6:
            x = ReLU(max_value=6, name='conv_dw_%d_relu6'% block_name)(x)
        else:
            x = ReLU(name='conv_dw_%d_relu' % block_name)(x)
        
        x = Conv2D(pointwise_conv_filters, (1, 1),
                   padding='same',
                   strides=(1, 1),
                   use_bias=self.with_bias,
                   kernel_initializer='he_normal', 
                   bias_initializer=Constant(0.),
                   name='conv_pw_%d' % block_name)(x)

        if not self.no_bn:
            x = BatchNormalization(name='conv_pw_%d_bn' % block_name)(x)
        if self.is_relu6:
            x = ReLU(max_value=6, name='conv_pw_%d_relu6' % block_name)(x)
        else:
            x = ReLU(name='conv_pw_%d_relu' % block_name)(x)

        return x


    def mobilenetv1(self,
                    input_tensor):
        net = {}

        # Block 1
        # 160, 480, 3 -> 160, 480, 3
        x = input_tensor

        # 160, 480, 3 -> 80, 240, 32
        x = Conv2D(32,
                  kernel_size=(3, 3),
                  strides=(2, 2),
                  padding='same',
                  use_bias=self.with_bias,
                  kernel_initializer='he_normal', 
                  bias_initializer=Constant(0.),
                  name='conv1')(x)

        if not self.no_bn:
            x = BatchNormalization(name='conv1_bn')(x)

        if self.is_relu6:
            x = ReLU(max_value=6, name='conv1_relu6')(x)
        else:
            x = ReLU(name='conv1_relu')(x)
        
        x = self._depthwise_conv_block(x, 64, block_name=1)

        # 80,240,64 -> 40,120,128
        x = self._depthwise_conv_block(x, 128, strides=(2, 2), block_name=2)
        x = self._depthwise_conv_block(x, 128, block_name=3)

        # Block 3
        # 40,120,128 -> 20,60,256
        x = self._depthwise_conv_block(x, 256, strides=(2, 2), block_name=4)

        x = self._depthwise_conv_block(x, 256, block_name=5)
        net['conv4_3'] = x

        # Block 4
        # 20,60,256 -> 10,30,512
        x = self._depthwise_conv_block(x, 512, strides=(2, 2), block_name=6)
        x = self._depthwise_conv_block(x, 512, block_name=7)
        x = self._depthwise_conv_block(x, 512, block_name=8)
        x = self._depthwise_conv_block(x, 512, block_name=9)
        x = self._depthwise_conv_block(x, 512, block_name=10)
        x = self._depthwise_conv_block(x, 512, block_name=11)
        # net['conv4_3'] = x

        # Block 5
        # 10,30,512 -> 10,30,1024
        x = self._depthwise_conv_block(x, 1024, strides=(1, 1), block_name=12)
        x = self._depthwise_conv_block(x, 1024, block_name=13)
        net['fc7'] = x

        # # x = Dropout(0.5, name='drop7')(x)
        # Block 6
        # 10,30,512 -> 5,15,512
        net['conv6_1'] = self.conv2d_bn(inputs=net['fc7'], filters=256, kernel_size=(1, 1), conv_ind=1)
        net['conv6_2'] = self.conv2d_bn(inputs=net['conv6_1'], filters=512, kernel_size=(3, 3), stride=(2, 2), conv_ind=2)

        # Block 7
        # 5,15,512 -> 3,8,256
        net['conv7_1'] = self.conv2d_bn(inputs=net['conv6_2'], filters=128, kernel_size=(1, 1), conv_ind=3)
        net['conv7_2'] = self.conv2d_bn(inputs=net['conv7_1'], filters=256, kernel_size=(3, 3), stride=(2, 2), conv_ind=4)

        # Block 8
        # 3,8,256 -> 2,4,256
        net['conv8_1'] = self.conv2d_bn(inputs=net['conv7_2'], filters=128, kernel_size=(1, 1), conv_ind=5)
        net['conv8_2'] = self.conv2d_bn(inputs=net['conv8_1'], filters=256, kernel_size=(3, 3), stride=(2, 2), conv_ind=6)

        # Block 9
        # 2,4,256 -> 1,1,128
        net['conv9_0'] = self.conv2d_bn(inputs=net['conv8_2'], filters=64, kernel_size=(1, 1), conv_ind=7)
        net['conv9_1'] = self.conv2d_bn(inputs=net['conv9_0'], filters=128, kernel_size=(3, 3), stride=(2, 2), conv_ind=8)
        net['conv9_2'] = self.conv2d_bn(inputs=net['conv9_1'], filters=128, kernel_size=(3, 3), stride=(2, 2), conv_ind=9)  # 1,2,128 -> 1,1,128
        return net

      

class SSD_detector():
    def __init__(self, num_classes, encoder_bias, decoder_bias, no_bn, anchor_file = False, **kwargs):

        self.no_bn = no_bn
        self.is_relu6 = True,
        self.alpha = 1.0
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.num_classes = num_classes
        self.backbone = 'MobileNetV1'
        self.anchor_file = anchor_file
        self.training = kwargs.get('training', True)
     
    def __call__(self, input_tensor, input_shape, pretrained, structured=True):

        x = input_tensor

        img_size = (input_shape[1], input_shape[0])

        BackBone = Net_Backbone(self.encoder_bias,
                                self.no_bn)

        # if self.backbone == 'MobileNetV1':
        net = BackBone.mobilenetv1(x)
        # net = mobilenet(input_tensor)
        self.with_bias = self.decoder_bias
        ################################################## ##################################################
        net['conv4_3_norm'] = net['conv4_3']
        num_priors = 6
        # 预测框的处理
        # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整 20, 60
        net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, 
                                              kernel_size=(3, 3), 
                                              kernel_initializer='he_normal', 
                                              padding='same',
                                              use_bias= self.with_bias,
                                              name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])


        net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                               kernel_size=(3, 3),
                                               kernel_initializer='he_normal', 
                                               padding='same',
                                               use_bias= self.with_bias,
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
                                     use_bias= self.with_bias,
                                     name='fc7_mbox_loc')(
            net['fc7'])

        net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['fc7_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                      kernel_size=(3, 3),
                                      kernel_initializer='he_normal', 
                                      padding='same',
                                      use_bias= self.with_bias,
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
                                         use_bias= self.with_bias,
                                         name='conv6_2_mbox_loc')(net['conv6_2'])

        net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv6_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3),
                                          kernel_initializer='he_normal', 
                                          padding='same', 
                                          use_bias= self.with_bias,
                                          name='conv6_2_mbox_conf')(
            net['conv6_2'])
        
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
                                         use_bias= self.with_bias,
                                         name='conv7_2_mbox_loc')(net['conv7_2'])

        net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        x = Conv2D(num_priors * self.num_classes, 
                   kernel_size=(3, 3), 
                   kernel_initializer=Constant(1.),
                   padding='same', 
                   use_bias= self.with_bias,
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
                                         use_bias= self.with_bias,
                                         name='conv8_2_mbox_loc')(net['conv8_2'])

        net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv8_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3), 
                                          padding='same', 
                                          kernel_initializer='he_normal', 
                                          use_bias= self.with_bias,
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
                                         use_bias= self.with_bias,
                                         name='conv9_2_mbox_loc')(net['conv9_2'])

        net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv9_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3), 
                                          padding='same', 
                                          kernel_initializer='he_normal', 
                                          use_bias= self.with_bias,
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
        net['mbox_conf'] = Reshape((num_boxes, self.num_classes), name='mbox_conf_logits')(net['mbox_conf'])
        # net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])
        net['mbox_conf'] = Softmax(name='mbox_conf_final', dtype=tf.float32)(net['mbox_conf'])

        if self.anchor_file:
            
            if self.training:
                # mbox_priorbox = pickle.load(open(self.anchor_file, 'rb'))  # (1, 9646, 8)

                net['predictions'] = concatenate([net['mbox_loc'],
                                                  net['mbox_conf']],
                                                 axis=2, name='predictions')

                model = Model(input_tensor, net['predictions'], name='MobileNetV1-SSD')
                # print('Import anchor file: {} not used in the training'.format(self.anchor_file))

            else:
                mbox_priorbox = pickle.load(open(self.anchor_file, 'rb'))  # (1, 9646, 8)

                net['predictions'] = concatenate([net['mbox_loc'],
                                                  net['mbox_conf'],
                                                  mbox_priorbox],
                                                 axis=2, name='predictions')

                model = Model(input_tensor, net['predictions'], name='MobileNetV1-SSD')
                # print('Import anchor file: {}'.format(self.anchor_file))

        else:

            net['predictions'] = concatenate([net['mbox_loc'],
                                              net['mbox_conf'],
                                              net['mbox_priorbox']],
                                             axis=2, name='predictions')

            net['predictions'] = tf.cast(net['predictions'], dtype=tf.float32)

            model = Model(input_tensor, net['predictions'], name='MobileNetV1-SSD')
            # print('No anchor files')
            logger.info('No anchor files')

        if pretrained:
            model =load_keras_pretrained_weights(model)
            logger.info('Load mobilenetv1 to the backbone of SSD')

        return model

def load_keras_pretrained_weights(model):
    
    tf_model = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights='imagenet')
    mobilenetv1_1_224 = tf_model.get_weights()

    model_w = model.get_weights()

    for i, w in enumerate(model_w):
        model_w[i] = mobilenetv1_1_224[i]
        logger.info('{}, SSD: {}, mobilenetv1: {}'.format(i, np.shape(w), np.shape(mobilenetv1_1_224[i])))
        if i >= len(mobilenetv1_1_224) - 1:
            break

    model.set_weights(model_w)

    return model

def load_pretrained_weights(model):
    
    tf_model = tf.keras.models.load_model('../Pretrained_models/original/essay_mobilenet_ssd_weights.h5')

    model_w = model.get_weights()

    for i, w in enumerate(model_w):
        model_w[i] = mobilenetv1_1_224[i]
        # logger.info('{}, SSD: {}, mobilenetv1: {}'.format(i, np.shape(w), np.shape(mobilenetv1_1_224[i])))
        if i >= len(mobilenetv1_1_224) - 1:
            break

    model.set_weights(model_w)

    return model

if __name__ == '__main__':
    # Mobilenet v1 extractor
    input_tensor = Input(shape=(160, 480, 3))
    BackBone = Net_Backbone(with_bias = True,
                            no_bn = False,
                            state_bits = tf.float32)

    net = BackBone.mobilenetv1(input_tensor)
    mbv1_model = Model(input_tensor, net['conv9_2'])
    mbv1_model.summary()

    # Mobilenet v1 SSD detector
    """
        Total params: 3,234,624
        Trainable params: 3,184,256
        Non-trainable params: 50,368
    """

    input_tensor = Input(shape=(160, 480, 3))
    SSD_model = SSD_detector(num_classes=9,
                         with_bias=True,
                         no_bn=False,
                         state_bits=tf.float32,
                         anchor_file='../pregraite/utils/profiler_sample/Mobilenetv2_ssdLite160_480_anchors.pkl').run(input_tensor, input_shape=(160, 480, 3))

    SSD_model.summary()
    
    SSD_model = SSD_detector(num_classes=config['num_classes'],
                            with_bias=False,
                            no_bn=False,
                            state_bits=tf.float32,
                            training=True,
                            set_threshold=False,
                            threshold_train=False,
                            anchor_file=config['anchor_path']).run(input_tensor, config['input_shape'])

    inputs = np.ones(shape=(160, 480, 3))
    inputs = np.expand_dims(inputs, axis=0)

    outputs = SSD_model.predict(inputs)
    print('Predictions: ', np.shape(outputs))

