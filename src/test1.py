#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default='/home/zwh/work_space/18xx/Complex-YOLOv4-Pytorch/checkpoints/complexer_yolo/Model_complexer_yolo_epoch_10000.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

configs = parse_test_configs()
configs.distributed = False  # For testing

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())
# print(torch.cuda.get_device_capability(0))

model = create_model(configs)
model.print_network()
print('\n\n' + '-*=' * 30 + '\n\n')

device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)

assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

# pretrained_dict = torch.load(configs.pretrained_path, map_location=device_string)
model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

configs.device = torch.device(device_string)
model = model.to(device=configs.device)

out_cap = None

model.eval()

def pc_tf(pc):
    # extrinsic 中旋转的表达形式为旋转矩阵
    # pc shape (n , 3)
    # extrinsic shape (4, 4)
    R = np.array([[0.000796274,-1,0,0],[1,0.000796274,0,0],[0,0,1,0],[0,0,0,1]],dtype='float32')#yaw=1.57 hs
    # R = np.array([[0.000796274,1,0,0],[-1,0.000796274,0,0],[0,0,1,0],[0,0,0,1]],dtype='float32')#yaw=-1.57 ls
    # R = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype='float32')

    T = np.array([0,0,-0.8,0],dtype='float32')
    pc = (R @ pc.T).T + T # 注意 R 需要左乘, 右乘会得到错误结果
    return pc


def callback(data):
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "intensity"))
    pc_list = []
    for p in pc:
        pc_list.append([p[0], p[1], p[2], p[3]/255.0])
    lidarData = np.array(pc_list)

    lidarData = pc_tf(lidarData)

    b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
    rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

    with torch.no_grad():
        imgs_bev = torch.from_numpy(rgb_map).unsqueeze(0)
        input_imgs = imgs_bev.to(device=configs.device).float()
        t1 = time_synchronized()
        outputs = model(input_imgs)
        t2 = time_synchronized()
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

        cv2.imshow('test-img1', img_bev)
        print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
        cv2.waitKey(10)


        print('\tDone testing sample size {}, time: {:.1f}ms, speed {:.2f}FPS'.format(len(lidarData),(t2 - t1) * 1000,
                                                                                       1 / (t2 - t1)))

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/pointcloud_Pandar640", PointCloud2, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
