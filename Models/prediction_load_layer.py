
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@auther    : zzhu
@contact   : zzhu@graimatterlabs.ai
@time      : %(date)s
@fileName  : %(name)s
@license   : Copyright (c) GrAI Matter Labs SAS 2020. All rights reserved.
'''


import tensorflow as tf
from tensorflow.keras.layers import Layer

class prediction_layer(Layer):
    def __init__(self, name='prediction_holder', **kwargs):
        super(prediction_layer, self).__init__()
        self._name = name
      
    def build(self, input_shape):
        self.outputs = tf.Variable(name='predictions',
                                    shape=tf.TensorShape(None), 
                                    initial_value=0.0,  
                                    dtype=tf.float32,
                                    trainable=False)
        super(prediction_layer, self).build(input_shape)
    
    def get_value(self):
        return self.outputs

    def call(self, inputs, training=None, *args, **kwargs):
        out = inputs

        self.outputs.assign(out)
      
        return out

    def compute_output_shape(self, input_shape):
        return input_shape
