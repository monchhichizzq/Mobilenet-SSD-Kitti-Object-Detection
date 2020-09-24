import numpy as np
import pickle
import matplotlib.pyplot as plt
# from SSD_body.anchors import PriorBox, get_mobilenet_anchors, get_vgg16_anchors

def decode_boxes(input_shape, mbox_loc, mbox_priorbox, variances):
    # modification
    # mbox_priorbox = mbox_priorbox / 300
    #
    # 获得先验框的宽与高
    # left, top, rigth, bottom
    img_width = input_shape[0]
    img_height = input_shape[1]
    prior_width = (mbox_priorbox[:, 2] - mbox_priorbox[:, 0])/img_width
    prior_height = (mbox_priorbox[:, 3] - mbox_priorbox[:, 1])/img_height
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
    # modification
    # decode_bbox = decode_bbox * 300
    # 回到原图的大小比
    decode_bbox[0] *= img_width
    decode_bbox[2] *= img_width
    decode_bbox[1] *= img_height
    decode_bbox[3] *= img_height

    return decode_bbox

#
class PriorBox():
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):

        self.waxis = 1
        self.haxis = 0

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True

    def compute_output_shape(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def call(self, input_shape, mask=None):

        # 获取输入进来的特征层的宽与高
        # 3x3
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        # 获取输入进来的图片的宽和高
        # 300x300
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # 获得先验框的宽和高
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        print("box_widths:",box_widths)
        print("box_heights:",box_heights)

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)


        print("linx:",linx)
        print("liny:",liny)
        centers_x, centers_y = np.meshgrid(linx, liny)
        # 计算网格中心
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)


        fig = plt.figure(figsize=(img_width/100, img_height/100))
        ax = fig.add_subplot(111)
        plt.ylim(0,img_height*2)
        plt.xlim(0,img_width)
        plt.scatter(centers_x,centers_y)

        num_priors_ = len(self.aspect_ratios)
        print('priors number', num_priors_)
        # 4 或 6 个框
        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        print('prior_box_centers', np.shape(prior_boxes), num_priors_)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        # prior_boxes: 300, 24 意味着 从中心向四个方向扩展4 * 提出框6*


        # 获得先验框的左上角和右下角
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        prod_prior = int(np.prod(np.shape(prior_boxes))/4)
        new = decode_boxes(input_shape=self.img_size,
                            mbox_loc=np.random.randn(prod_prior,4),
                           mbox_priorbox=prior_boxes.reshape([prod_prior,4]),
                           variances=np.tile(np.expand_dims(self.variances,axis=0),prod_prior))
        prior_boxes = new.reshape([np.shape(prior_boxes)[0],-1])
        print('prior_box_centers', np.shape(prior_boxes))
        print('box_widths', np.shape(box_widths))
        print('box_heights', np.shape(box_heights))

        # 对于4 这个点总共有24框 （num_prior_*2）
        c = 4
        for i in range(int(np.shape(prior_boxes)[1])):
            try:
                rect = plt.Rectangle([prior_boxes[c, i], prior_boxes[c, 1 + i]], box_widths[int(i/4)] * 2, box_heights[int(i/4)] * 2,
                                      color="r", fill=False)
                ax.add_patch(rect)
            except (IndexError):
                pass

        plt.show()
        # 变成小数的形式
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)

        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        return prior_boxes

if __name__ == '__main__':
    net = {} 
    #-----------------------将提取到的主干特征进行处理---------------------------#
    img_size = (480,160)
    # img_size = (300, 300)
    # img_size, min_size, max_size=None, aspect_ratios=None, flip=True, variances=[0.1], clip=True
    # (11,11), (14, 37), (20, 15), (37, 23), (57, 42), (111, 74)
    # (7, 21), (9, 9), (14, 13)
    print('\nconv4_3_norm_mbox_priorbox 20,60')
    priorbox = PriorBox(img_size, 10.0, max_size=21.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox.call([20, 60])
    # (14, 37), (17, 16), (25, 12), (28, 14)
    print('\nfc7_mbox_priorbox 10,30')
    priorbox = PriorBox(img_size, 21.0, max_size=45.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox.call([10, 30])

    # (30, 70), (41, 23), (56, 34)
    print('\nconv6_2_mbox_priorbox 5,15')
    priorbox = PriorBox(img_size, 45.0, max_size=99.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox.call([5, 15])

    # (77, 53), (127, 77)
    print('\nconv7_2_mbox_priorbox 3,8')
    priorbox = PriorBox(img_size, 99.0, max_size=153.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox.call([3, 8])

    print('\nconv8_2_mbox_priorbox 2,4')
    priorbox = PriorBox(img_size, 153.0, max_size=207.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox.call([2, 4])

    print('\nconv9_2_mbox_priorbox 1,1')
    priorbox = PriorBox(img_size, 207.0, max_size=261.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')

    net['conv9_2_mbox_priorbox'] = priorbox.call([1, 1])
    #
    priorbox = PriorBox(img_size, 261.0, max_size=315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')

    net['pool6_mbox_priorbox'] = priorbox.call([1,1])
    #
    net['mbox_priorbox'] = np.concatenate([net['conv4_3_norm_mbox_priorbox'],
                                    net['fc7_mbox_priorbox'],
                                    net['conv6_2_mbox_priorbox'],
                                    net['conv7_2_mbox_priorbox'],
                                    net['conv8_2_mbox_priorbox'],
                                    net['pool6_mbox_priorbox']],
                                    axis=0)
    print(np.shape(net['mbox_priorbox']))








