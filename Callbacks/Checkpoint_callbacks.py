#!/usr/bin/env python
# encoding: utf-8

'''
@author: Zeqi@@
@file: Checkpoint_callbacks.py
@time: 1/12/20 5:02
'''

import os
import warnings
import tensorflow as tf
import numpy as np

class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, 
                filepath, 
                model_name, 
                monitor='val_loss', 
                verbose=0,
                save_best_only=False, 
                save_weights_only=False,
                mode='auto', 
                period=1, 
                add_nb_spikes=False):

        super(ModelCheckpoint, self).__init__()
        self.model_name = model_name
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.add_nb_spikes = add_nb_spikes

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(logs)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # model_sparsity = measure_weights_sparsity(self.model)
            if self.add_nb_spikes:
                filepath = os.path.join(self.filepath, self.model_name%(epoch + 1, 
                                                                        logs.get('loss'), 
                                                                        logs.get('val_loss'), 
                                                                        logs.get('val_total_loss'), 
                                                                        logs.get('val_l1_loss'), 
                                                                        logs.get('mAP'), 
                                                                        logs.get('val_nb_spikes')))
            else:
                filepath = os.path.join(self.filepath, self.model_name%(epoch + 1, logs.get('loss'), logs.get('val_loss'), logs.get('mAP')))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            print('save weights only')
                            # or model_for_pruning.save(keras_model_file, include_optimizer=True)
                            model_for_export = self.model
                            model_for_export.save_weights(filepath, overwrite=True)
                        else:
                            print('Not save weights only')
                            model_for_export = self.model
                            print('Best model weights sparsity:')
                            # measure_weights_sparsity(model_for_export)
                            model_for_export.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    print('saving the quantized model! save_weight_only')
                    model_for_export = self.model
                    model_for_export.save_weights(filepath, overwrite=True)
                else:

                    model_for_export = self.model
                    #model_for_export.summary()
                    # measure_weights_sparsity(model_for_export)
                    self.model.save(filepath, overwrite=True)
                    # print('')
                    # print('saving the quantized model!')
                    # print('')