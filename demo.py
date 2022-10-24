import os
import paddle
import numpy as np
import cv2

from ext_op import RoIAlignRotated


def main():
    img_root = './data'
    # batch_ind, x, y, w, h, angle
    rois = [[0, 760, 467, 740, 114, 0.319553],
             [0, 940, 780, 500, 140, 0.379553],]
    imgs = [cv2.imread('data/IMG_1539.JPG')] # 读取输入图像
    # 设置gpu或cpu
    device = 'gpu:0' # cpu or gpu:0
    paddle.set_device(device)
    # 将数据迁移到对应的设备上
    imgs = paddle.to_tensor(imgs, dtype=paddle.float32, stop_gradient=False).transpose([0, 3, 1, 2])
    rois = paddle.to_tensor(rois, dtype=paddle.float32, stop_gradient=False)
    # 初始化RoIAlignRotated， 输出大小设置为h=100, w=800
    pool_fun = RoIAlignRotated(out_size=(100, 800), spatial_scale=1.0)
    # 前向推理
    output = pool_fun(imgs, rois)
    # 将输出结果保存
    output = output.transpose([0, 2, 3, 1])
    for i, out in enumerate(output):
        out = out.numpy().astype(np.uint8)
        cv2.imwrite('./data/result_{}.jpg'.format(i), out)

if __name__ == '__main__':
    main()