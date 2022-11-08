# -*- coding: utf-8 -*-
"""
乐乐感知学堂公众号
@author: https://blog.csdn.net/suiyingy
"""

from mayavi import mlab
import numpy as np
import cv2
import os
def viz_mayavi(points, vals="distance"):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mlab.points3d(x, y, z,
                  z,  # Values used for Color
                  mode="point",
                  colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                  # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                  figure=fig,
                  )
#   路径
root_path = '/home/zwh/work_space/18xx/Complex-YOLOv4-Pytorch/dataset/mini kitti/'
velodyne_path = root_path+'kitti mini data object veloyne/training/velodyne/000001.bin'
image_path = root_path+'kitti mini data object image 2/training/image_2/000001.png'

if __name__ == '__main__':
    assert(os.path.exists(velodyne_path) and os.path.exists(image_path)),"velodyne_path='{}' or image_path='{}' is not exist".format(velodyne_path,image_path)
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape([-1, 4])
    viz_mayavi(points)

    cv2.imshow('img',cv2.imread(image_path))
    cv2.waitKey()
    mlab.show()
