#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 1/12/20 1:48 PM
@fileName  : Data_Loader.py
'''

import sys
sys.path.append('../../Kitti-Object-Detection')

import os
import cv2
from PIL import Image
from tqdm import  tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import Sequence
from Preprocess.anchors import BBoxUtility, get_mobilenet_anchors
from Callbacks.plot_utils import plot_one_box

import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Tensorflow-2 Data Loader')

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_class(classes_path='../preparation/data_txt/kitti_classes.txt'):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


class Kitti_DataGenerator(Sequence):
    def __init__(self, train, batch_size, plot = False, **kwargs):
        self.plot = plot
        self.train_mode = train
        self.consecutive_frames = kwargs.get('consecutive_frames', False)

        self.batch_size = batch_size
        self.input_dtype = kwargs.get('input_dtype', np.float16)
        self.num_classes = kwargs.get('num_classes', 9 ) - 1
        self.model_image_size = kwargs.get('input_shape', (160, 480, 3))
        
        self.bbox_utils = kwargs.get('bbox_utils', None)
        self.data_path = kwargs.get('data_path', None)
        self.annotation_path =  kwargs.get('annotation_path', None) 
  

        if train:
            self.shuffle = True
            self.train_txt = kwargs.get('train_txt', 'kitti_obj_trainval.txt')
            self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/trainval.txt')).read().strip().split()
            self.lines = open(os.path.join(self.annotation_path, self.train_txt)).readlines()
            logger.info('Number of training samples: {}'.format(len(self.lines)))

        else:
            self.shuffle = False
            self.val_txt = kwargs.get('val_txt', 'kitti_obj_test.txt')
            if self.consecutive_frames:
                logger.info('Consecutive mode: {}'.format(self.consecutive_frames))
                image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
                self.generate_consecutive_ids(image_ids)
            else:
                image_ids = open(os.path.join(self.data_path,  'ImageSets/Main/test.txt')).read().strip().split()
                self.lines = open(os.path.join(self.annotation_path, self.val_txt)).readlines()
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
                image_path = os.path.join(self.data_path, "JPEGImages/" + image_id + '.png')
                self.image_path_list.append(image_path)

        else:

            logger.info('Generate consecutive frame path')

            for i, image_id in enumerate(image_ids):
                for j, name in enumerate(subframe_nums):
                    image_path = os.path.join(self.data_path, "JPEGImages/" + image_id + name)
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.data_path, "JPEGImages/" + image_id + subframe_nums[j + 1])
                        if not os.path.exists(image_path):
                            image_path = os.path.join(self.data_path, "JPEGImages/" + image_id + subframe_nums[j + 2])
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
        # logger.info('Preprocess on Validation data without Data Augmentation')
        line = annotation_line.split()  # Image path + class and box

        # Get image
        image = Image.open(line[0])
        # Input image's height and width
        iw, ih = image.size
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
        image = image.resize((nw, nh), Image.BICUBIC)

        # Create a new RGB image
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        # Crop the image
        dx, dy = (w - nw) // 2, (h - nh) // 2
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, self.input_dtype)
    
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
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''Data augmentation'''
        # logger.info('Preprocess on Validation data with Data Augmentation')

        line = annotation_line.split()  # Image path + class and box

        # Get image
        image = Image.open(line[0])
        # Original image size
        iw, ih = image.size
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
        image = image.resize((nw, nh), Image.BICUBIC)

        # Place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        # image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue) # -0.1, 0.1
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat) # 1, 1.5
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val) # 1, 1.5
        x = cv2.cvtColor(np.array(new_image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # Set input dtype
        image_data = np.array(image_data, self.input_dtype)

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
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

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
        image_batch = []
        label_batch = []
        for i, annotation_line in enumerate(image_ids_temp):
            if self.train_mode:
                input, labels = self.get_random_data(annotation_line)
            else:
                input, labels = self.get_val_data(annotation_line)
        
            anno_labels = labels

            labels = self.process_box(labels)

            if self.plot:
                class_names = get_class()
                # img = cv2.cvtColor(np.array(input, np.uint8), cv2.COLOR_RGB2BGR)
                img = np.array(input, np.uint8)
                # [x_min, y_min, x_max, y_max]
               
                for label in anno_labels:
                    # img, coord, class_names,  label=None, color=None, line_thickness=1
                    plot_one_box(img=img, coord=label[:4], label=class_names[int(label[-1])])
                plt.figure("Image_" + str(np.shape(img)))
                plt.imshow(img)
                plt.pause(0.1)

            image_batch.append(input)
            label_batch.append(labels)

        image_batch, label_batch = np.array(image_batch), np.array(label_batch)
        image_batch = preprocess_input(image_batch)

        return np.array(image_batch), np.array(label_batch)

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
            # print(np.shape(label_batch))
            return image_batch, label_batch


if __name__ == '__main__':
    NUM_CLASSES = 9
    priors = get_mobilenet_anchors(img_size=(480, 160))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    train_params = {'train': True,
                    'batch_size': 20,
                    'num_classes': 9,
                    'input_shape': (160, 480, 3),
                    'bbox_utils': bbox_util,
                    'plot': True}

    train_generator = Kitti_DataGenerator(**train_params)
    image_batch = next(iter(train_generator))
    print(np.shape(image_batch))


"""
caffe mode: convert the images from RGB to BGR, then will zero-center each color channel respect to the ImageNet dataset, without scaling
tf mode: scale pixels between -1 and 1, sample-wise
torch mode: will scale pixels between 0 and 1 nad then will normalize each channel with respecet ot the Imagenet dataset            
"""