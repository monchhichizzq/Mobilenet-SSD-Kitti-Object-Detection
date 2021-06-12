#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 1/25/21 9:38 PM
@fileName  : mAP_callbacks.py
'''

import cv2
import os
import time
import pickle
import shutil
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from subprocess import call
import xml.etree.ElementTree as ET
from Callbacks.Decoder_predictions import BBoxUtility
from Callbacks.plot_utils import plot_one_box, generate_colors
from Preprocess.Data_Loader import Kitti_DataGenerator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Mean average precision')

class get_predictions():
    def __init__(self, **kwargs):
        self.write_down = kwargs.get('write_down', True)
        self.gap_time = kwargs.get('gap_time', 1000)
        self.visual = kwargs.get('visual', True)
        self.nms_thresh = kwargs.get('nms_thresh', 0.5)
        self.IOU_thresh = kwargs.get('IOU_thresh', 0.5)
        self.confidence = kwargs.get('confidence', 0.01)
        self.model_image_size = kwargs.get('input_shape', (160, 480, 3))
        self.classes_path = kwargs.get('class_path', '../preparation/data_txt/kitti_classes.txt')
        self.anchor_file = kwargs.get('prior_file', 'SDK_models/Mobilenetv2_ssdLite160_480_anchors.pkl')
        self.pre_path = kwargs.get('pre_path', 'input/detection-results')

        self.class_names = self._get_class()
        self.colors = generate_colors(self.class_names)
        self.decoded_predictions = []

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        self.num_classes = len(class_names) + 1
        return class_names

    def get_Priorbox_predictions(self, predictions):
        mbox_priorbox = pickle.load(open(self.anchor_file, 'rb'))  # (1, 9646, 8)
        predictions = np.concatenate([predictions,
                                      mbox_priorbox],
                                     axis=2)
        # print('SDk predictions: {0}'.format(np.shape(predictions)))
        return predictions

    def ssd_correct_boxes(self, top, left, bottom, right, input_shape, image_shape):

        new_shape = image_shape * np.min(input_shape / image_shape)

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
        box_hw = np.concatenate((bottom - top, right - left), axis=-1)

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([
            box_mins[:, 0:1],
            box_mins[:, 1:2],
            box_maxes[:, 0:1],
            box_maxes[:, 1:2]
        ], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def run(self, orignal, outputs, image_id ,add_anchors):
        if self.write_down:
            self.detect_txtfile = open(os.path.join(self.pre_path, image_id + ".txt"), "w")

        prediction = []
        if add_anchors:
            for output in outputs:
                output = np.expand_dims(output, axis=0)
                anchor_prediction = self.get_Priorbox_predictions(output)
                prediction.append(anchor_prediction[0])
            prediction = np.array(prediction)
        else:
            prediction = outputs

        self.bbox_util = BBoxUtility(self.num_classes,
                                     overlap_threshold=self.IOU_thresh,
                                     nms_thresh=self.nms_thresh)

        results = self.bbox_util.detection_out(prediction,
                                               background_label_id=0, 
                                               keep_top_k=200,
                                               confidence_threshold=self.confidence)

        if len(np.shape(results)) < 3:
            pass
            # print('results:', np.shape(results))
        else:
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)

            # Remove the gray column
            image_shape = np.array(np.shape(orignal)[0:2])
            boxes = self.ssd_correct_boxes(top_ymin,
                                           top_xmin,
                                           top_ymax,
                                           top_xmax,
                                           np.array([self.model_image_size[0],self.model_image_size[1]]),
                                           image_shape)

            for i, c in enumerate(top_label_indices):
                predicted_class = self.class_names[int(c)-1]
                score = str(top_conf[i])

                top, left, bottom, right = boxes[i]
                if self.write_down:
                    self.detect_txtfile.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

                if self.visual:
                    self.decoded_predictions.append([predicted_class, score[:6], int(left), int(top), int(right), int(bottom)])
                    plot_one_box(orignal, [int(left), int(top), int(right), int(bottom)],
                                  label=predicted_class + ', {:.2f}%'.format(np.round(float(score) * 100,2)),
                                  color=self.colors[int(c)-1])

        if self.visual:
            # ti = end - start
            # cv2.putText(orignal, 'FPS: {0} Execution time: {1}ms'.format(str(int(1000/ti)), str(int(ti))), (20, 50), 0, 0.5, [20, 150, 255], thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow('Plot', orignal)
            cv2.waitKey(self.gap_time)

            # os.makedirs(self.plot_path, exist_ok=True)
            # cv2.imwrite(os.path.join(self.plot_path, image_id + ".png", ), bgr_img)

        if self.write_down:
            self.detect_txtfile.close()

        return self.decoded_predictions

class get_gt():
    def __init__(self, data_path, gt_path):
        self.data_path = data_path
        self.gt_path = gt_path
        #os.makedirs(self.gt_path, exist_ok=True)

    def run(self):
        ground_truth = []
        image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
        for image_id in tqdm(image_ids):
            with open(os.path.join(self.gt_path, image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(self.data_path, "Annotations/" + image_id + ".xml")).getroot()
                ground_truth_per_frame=[]
                for obj in root.findall('object'):
                    # if obj.find('difficult') != None:
                    #     difficult = obj.find('difficult').text
                    #     if int(difficult) == 1:
                    #         continue
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    ground_truth_per_frame.append([obj_name, left, top, right, bottom])
                # print('\n')
                # print(ground_truth_per_frame)
                ground_truth.append(ground_truth_per_frame)
        return ground_truth

def read_map(path):
    fp = open(path)
    lines = fp.readlines()
    for i, line in enumerate(lines):
        # print(i, line)
        if i == 11:
            mAP = float(line.split()[-1].split('%')[0])
    fp.close()
    return mAP


class VOC2012mAP_Callback(tf.keras.callbacks.Callback):

    def __init__(self, data_path, 
                 annotation_path, 
                 command_line, 
                 anchor_file,  
                 reset_freq=4, 
                 visual=True, 
                 **kwargs):
        
        super(VOC2012mAP_Callback, self).__init__()
        self.visual = visual
        self.reset_freq = reset_freq
        
        experiment_index=kwargs.get('ex_index', 0)
        self.gt_save = "input_{}/ground-truth_{}".format(experiment_index, experiment_index)
        self.detection_save = "input_{}/detection-results_{}".format(experiment_index, experiment_index)
        self.prediction_save = "input_{}/".format(experiment_index)
        self.command_line = command_line

        self.data_path = data_path
        self.annotation_path = annotation_path
        self.image_ids = open(os.path.join(data_path, 'ImageSets/Main/test.txt')).read().strip().split()

        self.anchor_file = anchor_file

        self.initialize_folder()
        ground_truth = get_gt(data_path=self.data_path, gt_path=self.gt_save).run()

        self.consecutive_frames = kwargs.get('consecutive_frames', False)
        self.generate_image_path_list()
        
        # self.remove_first_channel = kwargs.get('remove_first_channel', True)
        self.add_anchors = kwargs.get('add_anchors', False)
        self.map_read_path = kwargs.get('map_read_path', '../Callbacks/results/results_{}.txt'.format(experiment_index))


    def initialize_folder(self):
        if os.path.exists(self.prediction_save):
            shutil.rmtree(self.prediction_save)
            logger.info('Initialize the save folder: {}'.format(self.prediction_save))
        os.makedirs(self.prediction_save)

        os.makedirs(self.gt_save, exist_ok=True)
        os.makedirs(self.detection_save, exist_ok=True)

        logger.info('Makedirs (ground truth): {}'.format(self.gt_save))
        logger.info('Makedirs (detection): {}'.format(self.detection_save))


    def generate_image_path_list(self):
        consecutive_params = {'train': False,
                                'batch_size': 1,
                                'num_classes': 9,
                                'input_shape': (160, 480, 3),
                                'consecutive_frames':self.consecutive_frames,
                                'input_dtype': np.float32,
                                'data_path': self.data_path,
                                'annotation_path': self.annotation_path,
                                'plot': False}

        self.consecutive_generator = Kitti_DataGenerator(**consecutive_params)
        
        if self.consecutive_frames:
            self.consecutive_image_path_list = self.consecutive_generator.consecutive_image_path_list
            logger.info('Generate consecutive_image_path_list')
        else:
            self.image_path_list = self.consecutive_generator.image_path_list
            logger.info('Generate image path list')

    def on_predict_batch_end(self, batch, logs=None):
        # logger.info('on_predict_batch_end')
        if self.reset_freq == 1:
            prediction = logs['outputs'][0] if len(np.shape(logs['outputs'])) >3 else logs['outputs']
            image_ids = self.image_ids[batch]
            frame_path = self.image_path_list[batch]
            raw_image = cv2.imread(frame_path)
            decoded_prediction = get_predictions(pre_path=self.detection_save,
                                                 visual=self.visual,
                                                 prior_file=self.anchor_file).run(raw_image,
                                                                                  prediction,
                                                                                  image_ids,
                                                                                  self.add_anchors)
        else:
            if (batch+1) % self.reset_freq == 0:
                prediction = logs['outputs'][0] 
                image_ids = self.image_ids[int(batch//self.reset_freq)]
                frame_path = self.consecutive_image_path_list[batch]
                raw_image = cv2.imread(frame_path)
                decoded_prediction = get_predictions(pre_path=self.detection_save,
                                                     visual=self.visual,
                                                     prior_file=self.anchor_file).run(raw_image,
                                                                                      prediction,
                                                                                      image_ids,
                                                                                      self.add_anchors)

    def on_predict_end(self, logs=None):
        call(
                self.command_line,
                shell=True
            )
        logger.info('Run {}'.format(self.command_line))
        mAP = read_map(self.map_read_path)
        logs['mAP'] = mAP
        logger.info('mAP: {}%'.format(mAP))

    def on_test_batch_end(self, batch, logs=None):
        # logger.info('on_predict_batch_end')
        outputs = get_model_output(self.model)
        outputs = outputs.numpy()
        prediction = outputs[0] if len(np.shape(outputs)) >3 else outputs
        # print('\noutputs', type(prediction), np.shape(prediction))
        image_ids = self.image_ids[batch]
        frame_path = self.image_path_list[batch]
        raw_image = cv2.imread(frame_path)
        decoded_prediction = get_predictions(pre_path=self.detection_save,
                                                visual=self.visual,
                                                prior_file=self.anchor_file).run(raw_image,
                                                                                prediction,
                                                                                image_ids,
                                                                                self.add_anchors)                                                                                  
                                                                                      

    def on_epoch_end(self, epoch, logs=None):
        print('')
        logger.info('on_{}_epoch_end'.format(epoch))
        call(
            self.command_line,
            shell=True
        )
        mAP = read_map(self.map_read_path)
        logs['mAP'] = mAP
        logger.info('mAP: {}%'.format(mAP))


def get_model_output(model):
    layer = model.get_layer('prediction_holder')
    prediction = layer.get_value()
    # print(np.shape(prediction), type(prediction))
    return prediction
            

if __name__ == '__main__':
   pass
