#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@time      : 1/1/21 1:11 PM
@fileName  : train.py
'''

import os
import sys
sys.path.append('../../Kitti-Object-Detection')
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from Models.ResNet18_SSD_sparse import SSD_detector
from tensorflow.keras.optimizers import Adam

from Preprocess.TFData_Loader import build_input

from Loss.Loss_Sparsity import MultiboxLoss
from Preprocess.anchors import get_pruned_mobilenetv2_anchors

from Callbacks.utils import BBoxUtility
from Callbacks.mAP_callbacks import VOC2012mAP_Callback 
from Callbacks.Checkpoint_callbacks import ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Main-train')

ex_index = 2

config = {'Training mode': True,
          'plot':False,
          'mix_precision float16':False,
          'consecutive_frames': False,
          
          # Hyperparameters
          'lr': 1e-3,
          'epochs': 10000,
          'batch_size': 128,
          'num_classes': 9,
          'input_shape':(160, 480, 3),
          'input_dtype': np.float32,
          'reset':1,
          'add_anchors':False,
          'plot_prediction':False,
          'val_data_split':0.05,

          # Model
          'Fold_model': False, # BN fold or not

          # Regularization
          'l1' : 1e-5,
          'aug_rate': 5,
          'expectation': 74,

          # loss
          'alpha': 1.0,

          # checkpoint
          'checkpoint_dir':'logs_v1/train_resnet18',

          # Quantization
          'quantization' : False,
          'quantize_mode': 'AdaptivFloat',
          'bits': 8,
          'per_channel': False,

          # Path
          'anchor_path': '../Pretrained_models/original/Mobilenetv1_ssd160_480_anchors_pruned.pkl',
          'data_path': "/nas/datasets/kitti_dataset/2D_objects/kitti_voc/",
          'annotation_path': '../preparation/data_txt',
          'command_line':"python3 ../Callbacks/get_map.py --ex_index {} --GT_PATH ../Trainer_tfdata/input_{}/ground-truth_{} --DR_PATH ../Trainer_tfdata/input_{}/detection-results_{}".format(ex_index, ex_index, ex_index, ex_index, ex_index),
          }


if __name__ == "__main__":
    # Logs directory
    log_dir = config['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    annotation_path = os.path.join(config['annotation_path'], 'kitti_obj_trainval.txt')
    command_line = config['command_line']
    num_classes = config['num_classes']
    input_shape = config['input_shape']
    
    # Load the data
    logger.info('Building data loader')
    h, w, c = config['input_shape']
    priors = get_pruned_mobilenetv2_anchors()
    bbox_util = BBoxUtility(num_classes, priors)

    file = os.path.join('../preparation/data_txt', 'kitti_obj_trainval.txt')
    train_dataset = build_input(file, config['batch_size'], is_train=True)

    file = os.path.join('../preparation/data_txt', 'kitti_obj_test.txt')
    val_dataset = build_input(file, 1, is_train=False)
    
    # Build model
    input_tensor = Input(shape=input_shape)

    # mirrored_strategy = tf.distribute.MirroredStrategy()

    # with mirrored_strategy.scope():

    model = SSD_detector(num_classes=config['num_classes'],
                        encoder_bias=True, 
                        decoder_bias=True,
                        use_bn=False,
                        anchor_file=None)

    BN_model = SSD_detector(num_classes=config['num_classes'],
                            encoder_bias=False, 
                            decoder_bias=True,
                            use_bn=True,
                            anchor_file=None)

    if config['Fold_model']:
        Conv_SSD = model(input_tensor, input_shape=input_shape, structured=True)
        # Conv_SSD.load_weights(config['pretrained_model'], by_name=True, skip_mismatch=True)
        logger.info('Build SSD model - BN {} - bias {}'.format(False, False))
        SSDBN_model = Conv_SSD
    else:
        Conv_SSD = model(input_tensor, input_shape=input_shape, structured=True)
        ConvBN_SSD = BN_model(input_tensor, input_shape=input_shape, structured=True)
        # ConvBN_SSD.load_weights(config['pretrained_model'], by_name=True, skip_mismatch=True)
        logger.info('Build SSD model - BN {} - bias {}'.format(True, False))
        # SSDBN_model = Fold_model(verbose=True).BN_merge_into_Conv(Conv_SSD, ConvBN_SSD)
        SSDBN_model = ConvBN_SSD

    # Set Callbacks
    checkpoint = ModelCheckpoint(log_dir,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                save_weights_only=False,
                                verbose=1,
                                add_nb_spikes=True,
                                model_name='v1_ep_%05d-valloss_%0.2f-valtloss_%0.2f-vall1loss_%0.2f-mAP_%0.2f-spikes_%0.2f.h5')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)

    map_callbacks = VOC2012mAP_Callback(ex_index=ex_index,
                                        visual=config['plot_prediction'],
                                        data_path=config['data_path'],
                                        reset_freq=config['reset'],
                                        add_anchors=config['add_anchors'],
                                        anchor_file=config['anchor_path'],
                                        annotation_path=config['annotation_path'],
                                        command_line=config['command_line'])

    multibox_loss = MultiboxLoss(model=SSDBN_model,
                                batch_size=1,
                                num_classes=config['num_classes'],
                                alpha=config['alpha'],
                                neg_pos_ratio=3.0,
                                background_label_id=0,
                                negatives_for_hard=100.0,
                                use_focal_loss=False,
                                anchor_file=False)

    SSDBN_model.compile(optimizer=Adam(lr=config['lr']), 
                        loss=multibox_loss.total_loss,
                        metrics=[multibox_loss.total_loss,
                                multibox_loss.location_loss_positive,
                                multibox_loss._softmax_loss])

    if config['Training mode']:
        # SSDBN_model.evaluate(val_dataset, verbose=1)
        # SSDBN_model.predict(val_dataset, verbose=1, callbacks=[weight_sparsity_callback, map_callbacks, update_l1_callback])

        SSDBN_model.fit(train_dataset,
                    validation_data=val_dataset,
                    initial_epoch=0,
                    epochs=config['epochs'],
                    verbose=1,
                    callbacks=[reduce_lr, map_callbacks, checkpoint])
    else:

        SSDBN_model.evaluate(val_dataset, verbose=1)
        SSDBN_model.predict(val_dataset, verbose=1, callbacks=[map_callbacks])
