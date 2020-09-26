#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from Eval.get_Kittidetection_txt import SSD_predictions
import tensorflow as tf
from PIL import Image
import os
import cv2
import numpy as np

FPS = 20
height = 480
width = 1280

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
out = cv2.VideoWriter('0047_drive.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, (width, height))

path = '2011_09_26_drive_0005_extract/2011_09_26/2011_09_26_drive_0005_extract/image_02/data'
path = '2011_10_03_drive_0047_extract/2011_10_03/2011_10_03_drive_0047_extract/image_03/data'
ssd = SSD_predictions(gap=2)
for image_name in os.listdir(path):
    image_path = os.path.join(path, image_name)
    bgr_img=cv2.imread(image_path)
    print(np.shape(bgr_img))
    image = Image.open(image_path)
    img = ssd.detect_image(image_name, image, bgr_img)
    img = cv2.resize(img, (width, height))
    print(np.shape(img))
    out.write(img)
out.release()
cv2.destroyAllWindows()
