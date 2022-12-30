import os
import time
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import load_depth, get_bbox, compute_mAP, plot_mAP


parser = argparse.ArgumentParser()

parser.add_argument('--result_dir', type=str, default='', help='result directory')
opt = parser.parse_args()

def evaluate():
    # metric
    result_dir = opt.result_dir
    with open(result_dir,'rb') as f:
        res = cPickle.load(f)

    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    plot_mAP(res["iou_aps"],res["pose_aps"],'/'.join(result_dir.split("/")[:-1]),iou_thres_list,degree_thres_list,shift_thres_list)

if __name__ == '__main__':
    print('Evaluating ...')
    evaluate()