#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 1/11/21 4:01 PM
@fileName  : Loss.py
'''

import pickle
import tensorflow as tf
import numpy as np
from prettytable import PrettyTable
from tensorflow.keras.layers import concatenate

def get_Priorbox_predictions(model_out, anchor_file): # none, 9575, 13
    mbox_priorbox = pickle.load(open(anchor_file, 'rb'))  # (1, 9646, 8)
    predictions = concatenate([model_out, mbox_priorbox], axis=2)
    print('SDk predictions: {0}'.format(np.shape(predictions)))
    return predictions

class MultiboxLoss(object):
    def __init__(self,
                 model, 
                 batch_size,
                 num_classes,
                 alpha=1.0,
                 neg_pos_ratio=3.0,
                 background_label_id=0,
                 negatives_for_hard=100.0,
                 use_focal_loss = False,
                 anchor_file=None,):
        
        self.model = model
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
        self.use_focal_loss = use_focal_loss
        self.anchor_file = anchor_file
        if self.anchor_file:
            self.mbox_priorbox = pickle.load(open(anchor_file, 'rb'))

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_true, y_pred = y_true[:, :, 4:-8], y_pred[:, :, 4:-8]
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def _focal_loss(self, y_true, y_pred, alpha=1, gamma=2.0):
        y_true, y_pred = y_true[:, :, 4:-8], y_pred[:, :, 4:-8]
        
        y_pred = tf.maximum(y_pred, 1e-7)
        
        ones = tf.ones_like(y_true)
        
        # alpha should be a step function as paper mentioned, but ut doesn't matter ()
        
        alpha_t = tf.where(tf.equal(y_true, 1), alpha*ones, 1 - alpha*ones)
        
        # ce = - y_true * tf.math.log(y_pred)
        
       # weight = (1 - y_pred)** gamma
        
        focal_loss = -tf.reduce_sum(alpha_t*y_true *(1 - y_pred)** gamma* tf.math.log(y_pred), axis=-1)
        return focal_loss

    def compute_loss(self, y_true, y_pred):
        
        '''
            Add the pkl anchors
        '''
        y_true=tf.cast(y_true, tf.float32)
        y_pred=tf.cast(y_pred, tf.float32)

        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.cast(tf.shape(y_true)[1], tf.float32)

        if self.anchor_file:

            mbox_priorbox_batch = []
            for i in range(self.batch_size):
                mbox_priorbox_batch.append(self.mbox_priorbox[0])
            mbox_priorbox_batch = np.array(mbox_priorbox_batch)
            y_pred = concatenate([y_pred, mbox_priorbox_batch], axis=2)

        else:
            pass
        
        # print('SDk predictions: {0}'.format(np.shape(y_pred)))
        # 计算所有的loss
        # SSD 输出
        # mbox_loc_final: (batch_size, 2278, 4)
        # mbox_conf_final: (batch_size, 2278, 9)
        # mbox_priorbox: (batch_size, 2278, 8)
        # 分类的loss
        # batch_size,8732,21 -> batch_size,8732
        if self.use_focal_loss:
            conf_loss = self._focal_loss(y_true, y_pred)
        else:
            conf_loss = self._softmax_loss(y_true, y_pred)


        # 框的位置的loss
        # batch_size,8732,4 -> batch_size,8732
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # 获取所有的正标签的loss
        # 每一个batch的pos的个数
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        # 每一个batch的pos_loc_loss
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                     axis=1)
        # 每一个batch的pos_conf_loss
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                      axis=1)

        # 获取一定的负样本
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos,
                             num_boxes - num_pos)

        # 找到了哪些值是大于0的
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 获得一个1.0
        has_min = tf.cast(tf.reduce_any(pos_num_neg_mask), tf.float32)
        num_neg = tf.concat(axis=0, values=[num_neg,
                                            [(1 - has_min) * self.negatives_for_hard]])
        # 求平均每个图片要取多少个负样本
        num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg,
                                                       tf.greater(num_neg, 0)))
        num_neg_batch = tf.cast(num_neg_batch, tf.int32)

        # conf的起始
        confs_start = 4 + self.background_label_id + 1
        # conf的结束
        confs_end = confs_start + self.num_classes - 1

        # 找到实际上在该位置不应该有预测结果的框，求他们最大的置信度。
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)

        # 取top_k个置信度，作为负样本
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)

        # 找到其在1维上的索引
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(num_boxes, tf.int32) +
                        tf.reshape(indices, [-1]))

        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                           tf.ones_like(num_pos))
        total_loss = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_loss /= tf.reduce_sum(num_pos)
        total_loss += tf.reduce_sum(self.alpha * pos_loc_loss) / tf.reduce_sum(num_pos)

        # Restrain spikes
        # spikes, _ = self.punish_FP_density()
        # total_loss += spikes
        
        pos_loss = tf.reduce_sum(pos_conf_loss) / tf.reduce_sum(num_pos)
        neg_loss = tf.reduce_sum(neg_conf_loss) / tf.reduce_sum(num_pos)
        conf_loss = (tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss))/tf.reduce_sum(num_pos)
        loc_loss_positive = tf.reduce_sum(self.alpha * pos_loc_loss) / tf.reduce_sum(num_pos)
        
        return total_loss, neg_loss, pos_loss, loc_loss_positive

    def punish_FP_density(self, verbose=True):
        # self.model.summary()
        delta_spikes = 0
        delta_spikes_total = 0

        _table = PrettyTable()
        _table.field_names = ['layer name',
                            'spikes(K)',
                            'total spikes(K)',
                            'spike sparsity(%)']

        # Collect layer infos
        count = 0
        for i, layer in enumerate(self.model.layers):

            if hasattr(layer, 'get_spikes'):
                delta_spikes += layer.get_spikes()[0]/10**6/self.batch_size
                delta_spikes_total += layer.get_spikes()[1]/10**6/self.batch_size
    
                if verbose:
                   
                    spikes_per_layer = layer.get_spikes()[0] / 10 ** 3/self.batch_size # k
                    total_spikes_per_layer = layer.get_spikes()[1] / 10 ** 3/self.batch_size # k
                    spike_density = (1- spikes_per_layer / total_spikes_per_layer) * 100#%

                    # _table.add_row([layer.name,   
                    #                 spikes_per_layer,
                    #                 total_spikes_per_layer,
                    #                 spike_density])
                    tf.print([layer.name,   
                            spikes_per_layer,
                            total_spikes_per_layer,
                            spike_density])
                    
        # if self.verbose:
        #     print(_table)
        # if verbose:
        #     print(_table)
        
        return delta_spikes, delta_spikes_total
    
    def l1_loss(self, y_true, y_pred):
        return tf.reduce_sum(self.model.losses)

    def total_loss(self, y_true, y_pred):
        self.loss_out = self.compute_loss(y_true, y_pred)
        return self.loss_out[0]
    
    def negatives_loss(self, y_true, y_pred):
        return self.loss_out[1]

    def positives_loss(self, y_true, y_pred):
        return self.loss_out[2]

    def location_loss_positive(self, y_true, y_pred):
        return self.loss_out[3]
    
    def nb_spikes(self, y_true, y_pred):
        spikes, _ = self.punish_FP_density(verbose=False)
        return spikes
    
    def nb_total_spikes(self, y_true, y_pred):
        spikes, total_spikes = self.punish_FP_density(verbose=False)
        return total_spikes

