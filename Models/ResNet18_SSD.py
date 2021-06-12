#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@time      : 3/16/21 11:57 AM
@fileName  : ResNet18_SSD.py
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.append('../Models')

import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.initializers import Constant
from Models.anchor_layer import PriorBox
from Models.prediction_load_layer import prediction_layer


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('ResNet18-SSD')


class Net_Backbone():
    def __init__(self, use_bn=True, use_bias=False, **kwargs):

        self.use_bn = use_bn
        self.use_bias = use_bias
        self.layers_dims = [2,2,2,2]
        

    def build_basic_block(self, inputs, filter_num, blocks, stride, module_name):
        #The first block stride of each layer may be non-1
        x = self.Basic_Block(inputs, filter_num, stride, block_name='{}_{}'.format(module_name, 0)) 

        for i in range(1, blocks):      
            x = self.Basic_Block(x, filter_num, stride=1, block_name='{}_{}'.format(module_name, i))

        return x
    
    def Basic_Block(self, inputs, filter_num, stride=1, block_name=None):
        conv_name_1 = 'block_' + block_name +'_conv_1'
        conv_name_2 = 'block_' + block_name +'_conv_2'
        skip_connection = 'block_' + block_name  + '_skip_connection' 

        # Part 1
        x = Conv2D(filter_num, (3,3), strides=stride, padding='same', use_bias=self.use_bias, name=conv_name_1)(inputs)
        if self.use_bn:
            x = BatchNormalization(name=conv_name_1+'_bn')(x)
        x = ReLU(name=conv_name_1+'_relu')(x)

        # Part 2
        x = Conv2D(filter_num, (3,3), strides=1, padding='same', use_bias=self.use_bias, name=conv_name_2)(x)
        if self.use_bn:
            x = BatchNormalization(name=conv_name_2+'_bn')(x)
        
        # skip
        if stride !=1:
            residual = Conv2D(filter_num, (1,1), strides=stride, use_bias=self.use_bias, name=skip_connection)(inputs)
        else:
            residual = inputs
        
        # Add
        x = Add(name='block_' + block_name +'_residual_add')([x, residual])
        out = ReLU(name='block_' + block_name +'_residual_add_relu')(x)

        return out
    
    def ConvBn_Block(self, inputs, filters, kernel_size, stride, block_name):
        conv_name = 'block_convbn' + block_name +'_conv'
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=self.use_bias, name=conv_name)(inputs)
        if self.use_bn:
            x = BatchNormalization(name=conv_name+'_bn')(x)
        out = ReLU(name=conv_name+'_relu')(x)
        return out
        

    def ResNet18(self, input_tensor):
        net = {}

        # Block 1
        # 160, 480, 3 -> 160, 480, 3
        x = input_tensor

         # Initial
        # 160, 480, 3 -> 80, 240, 3
        x = self.ConvBn_Block(x, filters=64, kernel_size=(7, 7), stride=(2, 2), block_name='0')
        # 80, 240, 64 -> 40, 120, 64
        x = MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

        # Basic blocks
        # 40, 120, 64 -> 40, 120, 64
        x = self.build_basic_block(x, filter_num=64,  blocks=self.layers_dims[0], stride=1, module_name='module_0')
        # 40, 120, 64 -> 20, 60, 128
        x = self.build_basic_block(x, filter_num=128, blocks=self.layers_dims[1], stride=2, module_name='module_1')
        net['conv4_3'] = x

        # 20, 60, 128 -> 10, 30, 256
        x = self.build_basic_block(x, filter_num=256, blocks=self.layers_dims[2], stride=2, module_name='module_2')
        net['fc7'] = x

        # 10, 30, 256 -> 5, 15, 512
        net['conv6_2'] = self.build_basic_block(net['fc7'], filter_num=512, blocks=self.layers_dims[3], stride=2, module_name='module_3')
    
        # Block 7
        # 5,15,512 -> 3,8,256
        net['conv7_1'] = self.ConvBn_Block(inputs=net['conv6_2'], filters=128, kernel_size=(1, 1), stride=(1, 1), block_name='3')
        net['conv7_2'] = self.ConvBn_Block(inputs=net['conv7_1'], filters=256, kernel_size=(3, 3), stride=(2, 2), block_name='4')

        # Block 8
        # 3,8,256 -> 2,4,256
        net['conv8_1'] = self.ConvBn_Block(inputs=net['conv7_2'], filters=128, kernel_size=(1, 1), stride=(1, 1), block_name='5')
        net['conv8_2'] = self.ConvBn_Block(inputs=net['conv8_1'], filters=256, kernel_size=(3, 3), stride=(2, 2), block_name='6')

        # Block 9
        # 2,4,256 -> 1,1,128
        net['conv9_0'] = self.ConvBn_Block(inputs=net['conv8_2'], filters=64, kernel_size=(1, 1),  stride=(1, 1), block_name='7')
        net['conv9_1'] = self.ConvBn_Block(inputs=net['conv9_0'], filters=128, kernel_size=(3, 3), stride=(2, 2), block_name='8')
        net['conv9_2'] = self.ConvBn_Block(inputs=net['conv9_1'], filters=128, kernel_size=(3, 3), stride=(2, 2), block_name='9')  # 1,2,128 -> 1,1,128
        return net

      

class SSD_detector():
    def __init__(self, num_classes, encoder_bias, decoder_bias, use_bn, anchor_file = False, training=True, **kwargs):
        
        self.use_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.use_bn = use_bn
        self.num_classes = num_classes
        self.anchor_file = anchor_file
        self.training = training

    def __call__(self, input_tensor, input_shape, structured=False):

        x = input_tensor

        img_size = (input_shape[1], input_shape[0])

        BackBone = Net_Backbone(use_bn=self.use_bn, use_bias=self.use_bias)

        net = BackBone.ResNet18(x)

        decoder_bias = self.decoder_bias
        ################################################## ##################################################
        net['conv4_3_norm'] = net['conv4_3']
        num_priors = 6
        # 预测框的处理
        # num_priors表示每个网格点先验框的数量，4是x,y,h,w的调整 20, 60
        net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, 
                                              kernel_size=(3, 3), 
                                              kernel_initializer='he_normal', 
                                              padding='same',
                                              use_bias= decoder_bias,
                                              name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
        net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                               kernel_size=(3, 3),
                                               kernel_initializer='he_normal', 
                                               padding='same',
                                               use_bias= decoder_bias,
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
                                     use_bias= decoder_bias,
                                     name='fc7_mbox_loc')(net['fc7'])
        net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['fc7_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                      kernel_size=(3, 3),
                                      kernel_initializer='he_normal', 
                                      padding='same',
                                      use_bias= decoder_bias,
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
                                         use_bias= decoder_bias,
                                         name='conv6_2_mbox_loc')(net['conv6_2'])
        net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv6_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3),
                                          kernel_initializer='he_normal', 
                                          padding='same', 
                                          use_bias= decoder_bias,
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
                                         use_bias= decoder_bias,
                                         name='conv7_2_mbox_loc')(net['conv7_2'])
        net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        x = Conv2D(num_priors * self.num_classes, 
                   kernel_size=(3, 3), 
                   kernel_initializer=Constant(1.),
                   padding='same', 
                   use_bias= decoder_bias,
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
                                         use_bias= decoder_bias,
                                         name='conv8_2_mbox_loc')(net['conv8_2'])
        net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])

        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv8_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3), 
                                          padding='same', 
                                          kernel_initializer='he_normal', 
                                          use_bias= decoder_bias,
                                          name='conv8_2_mbox_conf')(net['conv8_2'])
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
                                         use_bias= decoder_bias,
                                         name='conv9_2_mbox_loc')(net['conv9_2'])
        net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
        # num_priors表示每个网格点先验框的数量，num_classes是所分的类
        net['conv9_2_mbox_conf'] = Conv2D(num_priors * self.num_classes, 
                                          kernel_size=(3, 3), 
                                          padding='same', 
                                          kernel_initializer='he_normal', 
                                          use_bias= decoder_bias,
                                          name='conv9_2_mbox_conf')(net['conv9_2'])
        net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])

        priorbox = PriorBox(img_size, 207.0, max_size=261.0, aspect_ratios=[2],
                            variances=[0.1, 0.1, 0.2, 0.2],
                            name='conv9_2_mbox_priorbox')

        net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])


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

                net['predictions'] = prediction_layer()(net['predictions'])

                model = Model(input_tensor, net['predictions'], name='ResNetV118-SSD')
                # print('Import anchor file: {} not used in the training'.format(self.anchor_file))

            else:
                # mbox_priorbox = pickle.load(open(self.anchor_file, 'rb'), encoding='utf-8')  # (1, 9646, 8)

                import pandas as pd
                mbox_priorbox = pd.read_pickle(self.anchor_file)
                print(mbox_priorbox)

                net['predictions'] = concatenate([net['mbox_loc'],
                                                  net['mbox_conf'],
                                                  mbox_priorbox],
                                                 axis=2, name='predictions')

                net['predictions'] = prediction_layer()(net['predictions'])

                model = Model(input_tensor, net['predictions'], name='ResNetV118-SSD')
                # print('Import anchor file: {}'.format(self.anchor_file))

        else:

            net['predictions'] = concatenate([net['mbox_loc'],
                                              net['mbox_conf'],
                                              net['mbox_priorbox']],
                                             axis=2, name='predictions')

            net['predictions'] = prediction_layer()(net['predictions'])

            net['predictions'] = tf.cast(net['predictions'], dtype=tf.float32)

            model = Model(input_tensor, net['predictions'], name='ResNetV118-SSD')
            # print('No anchor files')
            logger.info('No anchor files')

        # if pretrained:
        #     model =load_keras_pretrained_weights(model)
        #     logger.info('Load mobilenetv1 to the backbone of SSD')

        return model

if __name__ == '__main__':
    # ResNet18 v1 extractor
    input_tensor = Input(shape=(160, 480, 3))
    BackBone = Net_Backbone(layers_dims=[2, 2, 2, 2], 
                            use_bn=True, 
                            use_bias=True)

    net = BackBone.ResNet18(input_tensor)
    mbv1_model = Model(input_tensor, net['fc7'])
    mbv1_model.summary()

    # ResNet18 v1 detector
    input_tensor = Input(shape=(160, 480, 3))
    SSD_model = SSD_detector(num_classes=9,
                            use_bias=True,
                            use_bn=True,
                            anchor_file='../pregraite/utils/profiler_sample/Mobilenetv2_ssdLite160_480_anchors.pkl')(input_tensor, input_shape=(160, 480, 3))

    SSD_model.summary()
