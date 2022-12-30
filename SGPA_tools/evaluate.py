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

xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
norm_color = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def evaluate():
    result_dir = opt.result_dir
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    plot_mAP(iou_aps,pose_aps,result_dir,iou_thres_list,degree_thres_list,shift_thres_list)
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()

if __name__ == '__main__':
    print('Evaluating ...')
    evaluate()