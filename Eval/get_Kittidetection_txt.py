#----------------------------------------------------#
#   Get detection-result amd images-optional
#----------------------------------------------------#
# import sys
# sys.path.append("..")
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from Eval.nets_good import ssd
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from SSD_body.utils import BBoxUtility,letterbox_image,ssd_correct_boxes
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf
import time
import colorsys
import random

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class BBoxUtility(object):
    # def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
    #              nms_thresh=0.45, top_k=400):
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.3, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为ssd预测结果的格式

        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取ssd应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        # 除以0.1
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        # 除以0.2
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def detection_out(self, predictions, background_label_id=0, keep_top_k=200,
                      confidence_threshold=0.5):
        # 网络预测的结果 loc+conf+priorbox+variances
        mbox_loc = predictions[:, :, :4]
        # 0.1，0.1，0.2，0.2
        variances = predictions[:, :, -4:]
        # 先验框
        mbox_priorbox = predictions[:, :, -8:-4]
        # 置信度
        mbox_conf = predictions[:, :, 4:-8]
        results = []
        # 对每一个特征层进行处理
        for i in range(len(mbox_loc)):
            results.append([])
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])

            for c in range(self.num_classes):
                if c == background_label_id:
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > confidence_threshold
                if len(c_confs[c_confs_m]) > 0:
                    # 取出得分高于confidence_threshold的框
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    # 进行iou的非极大抑制
                    idx = tf.image.non_max_suppression(tf.cast(boxes_to_process, tf.float32),
                                                       tf.cast(confs_to_process, tf.float32),
                                                       self._top_k,
                                                       iou_threshold=self._nms_thresh).numpy()
                    # 取出在非极大抑制中效果较好的内容
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]
                    # 将label、置信度、框的位置进行堆叠。
                    labels = c * np.ones((len(idx), 1))
                    c_pred = np.concatenate((labels, confs, good_boxes),
                                            axis=1)
                    # 添加进result里
                    results[-1].extend(c_pred)
            if len(results[-1]) > 0:
                # 按照置信度进行排序
                results[-1] = np.array(results[-1])
                argsort = np.argsort(results[-1][:, 1])[::-1]
                results[-1] = results[-1][argsort]
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k]
        # 获得，在所有预测结果里面，置信度比较高的框
        # 还有，利用先验框和ssd的预测结果，处理获得了真实框（预测框）的位置
        return results


class SSD_predictions():
    #---------------------------------------------------#
    #   Detection
    #---------------------------------------------------#
    def __init__(self, gap=100):
        self.gap_time = gap
        self.backbone = 'mobilenet'
        self.model_image_size = (160, 480, 3)
        self.confidence = 0.3
        self.nms_thresh = 0.3
        self.IOU_thresh = 0.5
        self.model_path = '../Trainer/logs/ep242-loss2.113-val_loss2.084.h5'
        self.classes_path = '../preparation/data_txt/kitti_classes.txt'
        self.class_names = self._get_class()
        self.generate()  # Launch model

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)) # crop the image
        x_offset, y_offset = (w - nw) // 2 / iw, (h - nh) // 2 / ih
        return new_image, x_offset, y_offset

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算总的种类
        self.num_classes = len(self.class_names) + 1
        self.ssd_model = ssd.SSD300(self.model_image_size, self.backbone, self.num_classes)
        self.ssd_model.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Different color bounding box
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

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

    def detect_image(self, image_id, image, bgr_img, write_down=False):
        # img_ori = image
        if write_down:
            self.detect_txtfile = open("./input/detection-results/" + image_id + ".txt", "w")
        image_shape = np.array(np.shape(image)[0:2])
        crop_img, x_offset, y_offset = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        photo = np.array(crop_img,dtype = np.float64)

        # Image preprocessing
        photo = preprocess_input(np.reshape(photo,[1,self.model_image_size[0],self.model_image_size[1],3]))
        start = time.time()
        preds = self.ssd_model.predict(photo)
        end = time.time()
        ti = np.round((end - start) * 1000)
        print('Execution time: {} ms'.format(ti))

        # Decode the predictions
        self.bbox_util = BBoxUtility(self.num_classes, overlap_threshold=self.IOU_thresh,
                                     nms_thresh=self.nms_thresh)  # (self, num_classes, priors=None, overlap_threshold=0.5, nms_thresh=0.3, top_k=400)
        results = self.bbox_util.detection_out(preds, background_label_id=0, keep_top_k=200, confidence_threshold=self.confidence)
        
        if len(results[0])<=0:
            if write_down:
                self.detect_txtfile.close()
            return bgr_img

        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin, det_ymin, det_xmax, det_ymax = results[0][:, 2], results[0][:, 3], results[0][:, 4], results[0][:, 5]
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        # Remove the gray column
        boxes = self.ssd_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)


        # img_ori = np.array(img_ori)
        # img_ori = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)
        img_ori = bgr_img
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)-1]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            if write_down:
                self.detect_txtfile.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            self.plot_one_box(img_ori, [int(left), int(top), int(right), int(bottom)],
                              label=predicted_class + ', {:.2f}%'.format(np.round(float(score) * 100,2)),
                              color=self.colors[int(c)-1])
        cv2.putText(bgr_img, 'FPS: {0} Execution time: {1}ms'.format(str(int(1000/ti)), str(int(ti))), (20, 50), 0, 0.5, [20, 150, 255], thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow('draw', img_ori)
        cv2.waitKey(self.gap_time)

        if write_down:
            self.detect_txtfile.close()
        return bgr_img

    def plot_one_box(self, img, coord, label=None, color=None, line_thickness=None):
        '''
        coord: [x_min, y_min, x_max, y_max] format coordinates.
        img: img to plot on.
        label: str. The label name.
        color: int. color index.
        line_thickness: int. rectangle line thickness.
        '''

        tl = line_thickness or int(round(0.0005 * max(img.shape[0:2])))  # line thickness
        color = color or [random.randint(0, 255) for _ in range(len(self.class_names))]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__=='__main__':
    ssd = SSD_predictions()
    kitti_test_path = '../preparation/kitti_voc'
    image_ids = open(os.path.join(kitti_test_path, 'ImageSets/Main/test.txt')).read().strip().split()

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-results"):
        os.makedirs("./input/detection-results")
    if not os.path.exists("./input/images-optional"):
        os.makedirs("./input/images-optional")

    for image_id in tqdm(image_ids):
        image_path = os.path.join(kitti_test_path, "JPEGImages/"+image_id+".png")
        image = Image.open(image_path)
        # 开启后在之后计算mAP可以可视化
        image.save("./input/images-optional/"+image_id+".jpg")

        ssd.detect_image(image_id,image, write_down=True)


    print("Conversion completed!")
