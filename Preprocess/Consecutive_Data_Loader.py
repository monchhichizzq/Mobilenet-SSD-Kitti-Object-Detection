#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 1/12/20 1:48 PM
@fileName  : Consecutive_Data_Loader.py
'''

import sys
sys.path.append('../../Kitti-Object-Detection')

import os
import cv2
from PIL import Image
from tqdm import  tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from Callbacks.plot_utils import plot_one_box

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import Sequence
from Preprocess.anchors import BBoxUtility, get_mobilenet_anchors

import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Tensorflow-2 Consecutive Data Loader')


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_class(classes_path='../preparation/data_txt/kitti_classes.txt'):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_input_sparsity(input_tensor_sequence):
    input_previous, input_current = input_tensor_sequence
    delta_frame = input_current - input_previous
    non_zero_input = np.count_nonzero(delta_frame)
    total = np.prod(np.shape(delta_frame)) - 8*480*2
    
    logger.info('Non zero: {}, Total: {}'.format(non_zero_input, total))
    
    sparsity = (total - non_zero_input)/total * 100
    logger.info('Input sparsity: {} %'.format(np.round(sparsity, 2)))
    return delta_frame


class Kitti_DataGenerator(Sequence):
    def __init__(self, train, batch_size, plot = False, **kwargs):
        self.plot = plot
        self.train_mode = train
        self.consecutive_frames = kwargs.get('consecutive_frames', False)

        self.batch_size = batch_size
        self.input_dtype = kwargs.get('input_dtype', np.float32)
        self.num_classes = kwargs.get('num_classes', 9 ) - 1
        self.model_image_size = kwargs.get('input_shape', (160, 480, 3))
        
        self.bbox_utils = kwargs.get('bbox_utils', None)
        self.data_path = kwargs.get('data_path', None)
        self.annotation_path =  kwargs.get('annotation_path', None) 
        
        
        if train:
            self.shuffle = True
            self.train_txt = kwargs.get('train_txt', None)
            self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/trainval.txt')).read().strip().split()
            self.lines = open(os.path.join(self.annotation_path, self.train_txt)).readlines()
            logger.info('Number of training samples: {}'.format(len(self.lines)))

        else:
            self.shuffle = False
            self.val_txt = kwargs.get('val_txt', None)
            if self.consecutive_frames:
                image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
                self.generate_consecutive_ids(image_ids)
            else:
                image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
                self.lines = open(os.path.join(self.annotation_path,  self.val_txt)).readlines()
                logger.info('Number of validation samples: {}'.format(len(self.lines)))
                self.generate_consecutive_ids(image_ids)

        self.on_epoch_end()


    def generate_consecutive_ids(self, image_ids):

        subframe_nums = ['_03.png', '_02.png', '_01.png', '.png']
        # subframe_nums = ['.png', '.png', '.png', '.png']
        self.image_path_list = []
        self.consecutive_image_path_list = []

        if not self.consecutive_frames:

            logger.info('Generate random frame path')
            
            for i, image_id in enumerate(image_ids):
                image_path = os.path.join(self.data_path, "final/" + image_id + '.png')
                self.image_path_list.append(image_path)

        else:

            logger.info('Generate consecutive frame path')

            for i, image_id in enumerate(image_ids):
                for j, name in enumerate(subframe_nums):
                    image_path = os.path.join(self.data_path, "final/" + image_id + name)
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.data_path, "final/" + image_id + subframe_nums[j + 1])
                        if not os.path.exists(image_path):
                            image_path = os.path.join(self.data_path, "final/" + image_id + subframe_nums[j + 2])
                    self.consecutive_image_path_list.append(image_path)

    def get_consecutive_val_data(self, path):
        
        image = Image.open(path)
        iw, ih = image.size
        h, w, channel = self.model_image_size
   
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
    
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

        if self.plot:
            plt.figure("Image_" + str(np.shape(new_image)))
            plt.imshow(new_image)
            plt.pause(0.001)

        x_offset, y_offset = (w - nw) // 2 / iw, (h - nh) // 2 / ih
        image_data = np.array(new_image, self.input_dtype)
    
        return preprocess_input(image_data), x_offset, y_offset


    def get_val_data(self, annotation_line):

        '''No data augmentation'''
        line = annotation_line.split()  # Image path + class and box
        
        image_path_current = line[0]
        current_frame_name = line[0].split('/')[-1].split('.')[0]
        previous_frame_name = current_frame_name + '_01'
        image_path_current = os.path.join(self.data_path, "final/", current_frame_name + '.png')
        image_path_previous = os.path.join(self.data_path, "final/",  previous_frame_name + '.png')
        # logger.info('val - current_frame_name: {}'.format(image_path_current))
        # logger.info('val - previous_frame_name: {}'.format(image_path_previous))
       
        # Get image
        image_pre = Image.open(image_path_previous)
        image_cur = Image.open(image_path_current)
        
        # Input image's height and width
        iw, ih = image_cur.size
        # Model input's height and width
        h, w, channel = self.model_image_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # Resize image
        # Scale factor
        scale = min(w / iw, h / ih)
        # New scaled size
        nw = int(iw * scale)
        nh = int(ih * scale)
        # New scaled image
        image_pre = image_pre.resize((nw, nh), Image.BICUBIC)
        image_cur = image_cur.resize((nw, nh), Image.BICUBIC)

        # Create a new RGB image
        new_image_pre = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_cur = Image.new('RGB', (w, h), (128, 128, 128))
        
        # Crop the image
        dx, dy = (w - nw) // 2, (h - nh) // 2
        new_image_pre.paste(image_pre, (dx, dy))
        new_image_cur.paste(image_cur, (dx, dy))
        
        pre_image_data = np.array(new_image_pre, self.input_dtype)
        cur_image_data = np.array(new_image_cur, self.input_dtype)
    
        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return pre_image_data, cur_image_data, []

        if (box_data[:, :4] > 0).any():
            return pre_image_data, cur_image_data, box_data
        else:
            return pre_image_data, cur_image_data, []

    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''Data augmentation'''
        # logger.info('Preprocess on Validation data with Data Augmentation')

        line = annotation_line.split()  # Image path + class and box
        
        image_path_current = line[0]
        current_frame_name = line[0].split('/')[-1].split('.')[0]
        previous_frame_name = current_frame_name + '_01'
        image_path_current = os.path.join(self.data_path, "final/", current_frame_name + '.png')
        image_path_previous = os.path.join(self.data_path, "final/",  previous_frame_name + '.png')
        # logger.info('train - current_frame_name: {}'.format(image_path_current))
        # logger.info('train - previous_frame_name: {}'.format(image_path_previous))
        
        # Get image
        image_pre = Image.open(image_path_previous)
        image_cur = Image.open(image_path_current)
        
        # Original image size
        iw, ih = image_cur.size
        # Model input shape
        h, w, channel = self.model_image_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # Resize image
        # Choose a random value between 0.7. and 1.3
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        # Choose a random value between .5 and 1.5
        scale = rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image_cur = image_cur.resize((nw, nh), Image.BICUBIC)
        image_pre = image_pre.resize((nw, nh), Image.BICUBIC)

        # Place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image_cur = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_pre = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_cur.paste(image_cur, (dx, dy))
        new_image_pre.paste(image_pre, (dx, dy))
        
        image_cur = new_image_cur
        image_pre = new_image_pre

        # flip image or not
        flip = rand() < .5
        if flip: 
            # print('filp')
            image_cur = image_cur.transpose(Image.FLIP_LEFT_RIGHT)
            image_pre = image_pre.transpose(Image.FLIP_LEFT_RIGHT)
            
        # distort image
        hue = rand(-hue, hue) # -0.1, 0.1
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat) # 1, 1.5
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val) # 1, 1.5
        
        # previous
        x = cv2.cvtColor(np.array(image_pre, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        pre_image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        
        # current
        x = cv2.cvtColor(np.array(image_cur, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        cur_image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # Set input dtype
        pre_image_data = np.array(pre_image_data, self.input_dtype)
        cur_image_data = np.array(cur_image_data, self.input_dtype)

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        if len(box) == 0:
            return pre_image_data, cur_image_data, []

        if (box_data[:, :4] > 0).any():
            return pre_image_data, cur_image_data, box_data
        else:
            return pre_image_data, cur_image_data, []

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.consecutive_frames:
            self.indexes = np.arange(len(self.consecutive_image_path_list))
        else:
            self.indexes = np.arange(len(self.lines))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.consecutive_frames:
             return int(np.floor(len(self.consecutive_image_path_list)/self.batch_size))
        else:
            return int(np.floor(len(self.lines) / self.batch_size))

    def process_box(self, labels):
        # If objects are labeled in the input image
        if len(labels) != 0:
            height, width, channel = self.model_image_size
            # Get the box coordinates, left, top, right, bottom

            boxes = np.array(labels[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / width
            boxes[:, 1] = boxes[:, 1] / height
            boxes[:, 2] = boxes[:, 2] / width
            boxes[:, 3] = boxes[:, 3] / height
            one_hot_label = np.eye(self.num_classes)[np.array(labels[:, 4], np.int32)]
            if ((boxes[:, 3] - boxes[:, 1]) <= 0).any() and ((boxes[:, 2] - boxes[:, 0]) <= 0).any():
                pass

            labels = np.concatenate([boxes, one_hot_label], axis=-1)
            # print('y', tf.shape(y))
        train_labels = self.bbox_utils.assign_boxes(labels)
        return train_labels


    def __run_video(self, image_path_temp):

        # Generate data
        image_batch = np.empty((self.batch_size, self.model_image_size[0], self.model_image_size[1], 3))

        for i, image_path in enumerate(image_path_temp):
            # image_path = os.path.join(self.data_path, "JPEGImages/" + image_id + ".png")
            input, _, _ = self.get_consecutive_val_data(image_path)
            image_batch[i] =input

        return image_batch


    def __data_generation(self, image_ids_temp):

        # Generate data
        cur_image_batch = []
        pre_image_batch = []
        label_batch = []
        
        for i, annotation_line in enumerate(image_ids_temp):
            if self.train_mode:
                # pre_input, cur_input, labels = self.get_random_data(annotation_line)
                pre_input, cur_input, labels = self.get_val_data(annotation_line)
            else:
                pre_input, cur_input, labels = self.get_val_data(annotation_line)

            anno_labels = labels

            labels = self.process_box(labels)
            
            
            if self.plot:
                
                delta_frame = get_input_sparsity([cur_input, pre_input])
                class_names = get_class()
                # input = cv2.cvtColor(np.array(input, np.uint8), cv2.COLOR_RGB2BGR)
                cur_in = np.array(cur_input, np.uint8)
                pre_in = np.array(pre_input, np.uint8)

                for label in anno_labels:
                    # img, coord, class_names,  label=None, color=None, line_thickness=1
                    plot_one_box(img=cur_in, coord=label[:4], label=class_names[int(label[-1])], color=(0, 0, 255))

                delta_frame = np.array(delta_frame, np.uint8)
                
                logger.info('current_frame_name: {}'.format(np.shape(cur_in)))
                logger.info('previous_frame_name: {}'.format(np.shape(pre_in)))
        
                img =np.concatenate((pre_in, cur_in), axis=0)
                img =np.concatenate((img, delta_frame), axis=0)
                logger.info('Concatenation: {}'.format(np.shape(img)))

                w, h, c = img.shape
                img = cv2.resize(img, (h*4, w*4))
                cv2.imshow('image', img)
                # cv2.imshow('delta', delta_frame)
                cv2.waitKey(10000)
                
            pre_input = preprocess_input(pre_input)
            cur_input = preprocess_input(cur_input)
            
            cur_image_batch.append(cur_input)
            pre_image_batch.append(pre_input)
            label_batch.append(labels)
        
        cur_image_batch, pre_image_batch, label_batch = np.array(cur_image_batch), np.array(pre_image_batch), np.array(label_batch)
        
        # print(np.min(pre_image_batch), np.max(pre_image_batch))

        return [pre_image_batch, cur_image_batch], np.array(label_batch)


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        if self.consecutive_frames:
            consecutive_image_batch = [self.consecutive_image_path_list[k] for k in indexes]
            image_batch = self.__run_video(consecutive_image_batch)
            return image_batch
        
        else:
            annotation_lines_batch = [self.lines[k] for k in indexes]
            image_batch, label_batch = self.__data_generation(annotation_lines_batch)
            
            # logger.info('Image batch: {} {}'.format(np.shape(image_batch[0]), np.shape(image_batch[1])))
            # logger.info('label_batch: {}'.format(np.shape(label_batch)))
       
            return image_batch, label_batch


if __name__ == '__main__':
    NUM_CLASSES = 9
    priors = get_mobilenet_anchors(img_size=(480, 160))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    train_params = {'train': True,
                    'batch_size': 2000,
                    'num_classes': 9,
                    'input_shape': (160, 480, 3),
                    'bbox_utils': bbox_util,
                    'consecutive_frames': False,
                    'data_path': '/home/zzhu/kitti_depth_dataset/kitti_voc',
                    'annotation_path': '../preparation/data_txt',
                    'train_txt': 'kitti_obj_trainval.txt',
                    'plot': True}

    train_generator = Kitti_DataGenerator(**train_params)
    image_batch, label_batch = next(iter(train_generator))
    print(image_batch[0].shape, image_batch[1].shape)

    

"""
caffe mode: convert the images from RGB to BGR, then will zero-center each color channel respect to the ImageNet dataset, without scaling
tf mode: scale pixels between -1 and 1, sample-wise
torch mode: will scale pixels between 0 and 1 nad then will normalize each channel with respecet ot the Imagenet dataset            
"""