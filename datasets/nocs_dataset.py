from operator import truediv
import os
from cv2 import dilate
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from torch.utils import data
import random
import time
import itertools
import json
import open3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sciR
import logging
from PIL import Image
import re
import cv2
from NOCS_tools.utils import backproject, get_bbox
import torchvision.transforms as transforms
from scipy import ndimage
from visualization_utils import render_window,draw_points

def init_np_seed(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed()
    np.random.seed(seed%4294967296) # numpy seed must be between 0 and 2**32-1
    
def np_collate(batch):
    batch_z = zip(*batch)
    return [torch.stack([torch.from_numpy(b) for b in batch_z_z], 0) for batch_z_z in batch_z]

class PinMemDict:
    def __init__(self, data):
        self.data = data
    # custom memory pinning method on custom type
    def pin_memory(self):
        out_b = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                out_b[k] = v.pin_memory()
            else:
                out_b[k] = v
        return out_b
        
def np_collate_dict(batch):
    b_out = {}
    for k in batch[0].keys():
        c = []
        for b in batch:
            c.append(b[k])
        if type(c[0]) is np.ndarray:
            c = torch.from_numpy(np.stack(c, axis=0))
        else:
            pass
        b_out[k] = c
    # return PinMemDict(b_out)
    return b_out
    
class ShuffleWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        self.data_source = data_source
        self.n_repeats = n_repeats
        print('[ShuffleWarpSampler] Expanded data size: {}'.format(len(self)))
        
    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = torch.randperm(len(self.data_source)).tolist()
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)
    
    def __len__(self):
        return len(self.data_source) * self.n_repeats

class SequentialWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        self.data_source = data_source
        self.n_repeats = n_repeats
        print('[SequentialWarpSampler] Expanded data size: {}'.format(len(self)))
        
    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = list(range(len(self.data_source)))
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)
    
    def __len__(self):
        return len(self.data_source) * self.n_repeats

def compute_state(attrs, points):
    attrs = torch.tensor(attrs).cuda()
    points = torch.tensor(points).cuda()
    attrs = attrs.unsqueeze(1)
    points = points.unsqueeze(0)
    dists_all = torch.sum((points - attrs[...,1:])**2, dim=-1)
    dist_min, points_label = torch.min(dists_all, dim=-2)
    
    return points_label.cpu().numpy().astype(np.int32)

def compute_state_sym(attrs, points):
    attrs = torch.tensor(attrs).cuda()
    points = torch.tensor(points).cuda()
    attrs = attrs.unsqueeze(-3)
    points = points.unsqueeze(-2)
    # print(attrs.shape, points.shape)
    dists_all = torch.sum((points - attrs)**2, dim=-1)
    # print(dists_all.shape)
    dist_min, points_label = torch.min(dists_all, dim=-1)
    return points_label.cpu().numpy().astype(np.int32)

class NOCS_dataset(torch.utils.data.Dataset,):
    # args.data_dir, args.data_type, args.data_lists, args.attrs_dir, args.categories, args.names, args.num_segs, args.sym
    def __init__(self, cfg):
        self.onehot_dict = {'02876657': [1,0,0,0,0,0],
                            '02880940': [0,1,0,0,0,0],
                            '02942699': [0,0,1,0,0,0],
                            '02946921': [0,0,0,1,0,0],
                            '03642806': [0,0,0,0,1,0],
                            '03797390': [0,0,0,0,0,1]}
        self.sym = cfg.sym if(hasattr(cfg, 'sym')) else 0
        self.sym_dict = {   '02876657': self.sym,
                            '02880940': self.sym,
                            '02942699': 0,
                            '02946921': self.sym,
                            '03642806': 0,
                            '03797390': 0}

        self.obj_labels = []
        self.obj_ids = []
        self.obj_scales = []
        self.obj_Qs = []
        self.obj_ts = []
        self.mask_ids = []
        self.file_paths = []
        self.attrs = {}
        self.data_lists = cfg.data_lists
        self.categories = cfg.categories
        self.names = cfg.names
        self.num_segs = cfg.num_segs
        self.data_dir = cfg.data_dir
        self.data_type = cfg.data_type
        self.attrs_dir = cfg.attrs_dir
        
        self.jitter = cfg.jitter if(hasattr(cfg, 'jitter')) else 1
        self.num_points = cfg.num_points if(hasattr(cfg, 'num_points')) else 1024
        # self.dust_type = cfg.dust_type if(hasattr(cfg, 'dust_type')) else None
        self.bg_aug = cfg.bg_aug if(hasattr(cfg, 'bg_aug')) else False
        self.composed_dir = cfg.composed_dir if(hasattr(cfg, 'composed_dir')) else None
        self.norm_scale = 1000.0
        self.shift_range = 0.01
        
        if self.data_type in ['real_train', 'real_test']:
            self.intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
            assert(0), "not add suport for training on real data"
        else: ## CAMERA data
            self.intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])
        
        if('real' in self.data_type):
            f = open(self.data_lists,'r')
            lines = f.readlines()
            for i, (category, name) in enumerate(zip(self.categories, self.names)):
                for line in lines:
                    line_parse = re.split(r'[ ]+', line.strip('\n'))
                    if(name not in line_parse[3]):
                        continue
                    self.file_paths.append(line_parse[0])
                    self.mask_ids.append(int(line_parse[1]))
                    self.obj_ids.append(line_parse[3])
                    self.obj_scales.append(float(line_parse[4]))
                    self.obj_Qs.append(list(map(float,line_parse[5:9])))
                    self.obj_ts.append(list(map(float,line_parse[9:12])))
                    self.obj_labels.append(category)
            with open(os.path.join(self.attrs_dir, f'attrs_real_nocs.json'), 'r') as f:
                self.attrs=json.load(f)
        else:
            for i, (data_list, category, name) in enumerate(zip(self.data_lists, self.categories, self.names)):
                with open(data_list,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_parse = re.split(r'[ ]+', line.strip('\n'))
                        if(float(line_parse[4]) > 10000):
                            continue
                        self.file_paths.append(line_parse[0])
                        self.mask_ids.append(int(line_parse[1]))
                        self.obj_ids.append(line_parse[3])
                        self.obj_scales.append(float(line_parse[4]))
                        self.obj_Qs.append(list(map(float,line_parse[5:9])))
                        self.obj_ts.append(list(map(float,line_parse[9:12])))
                        self.obj_labels.append(category)
                with open(os.path.join(self.attrs_dir, f'attrs_{category}_{name}_{self.num_segs}.json'), 'r') as f:
                        self.attrs[category]=json.load(f)
    
    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)
        if(depth is None):
            return None

        if len(depth.shape) == 3:
            # This is encoded depth image, let's convert
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth

        return depth16
    
    def __getitem__(self, index):
        file_path =  self.file_paths[index]
        
        depth_path = os.path.join(self.data_dir, file_path+'_depth.png')
        mask_path = os.path.join(self.data_dir, file_path+'_mask.png')
        coord_path = os.path.join(self.data_dir, file_path+'_coord.png')
        # print(depth_path)

        if(self.bg_aug): # data augmentation for imperfect segmentation
            aug_type = np.random.randint(0,2)
            comp_depth_path = os.path.join(self.composed_dir, file_path+'_composed.png')
            comp_depth = self.load_depth(comp_depth_path)
        
        # print(depth_path,mask_path)
        try:
            assert os.path.exists(depth_path)
            assert os.path.exists(mask_path)
        except Exception as e:
            print("missing depth", depth_path)
            print(e)
            exit()
    
        depth = self.load_depth(depth_path)
        mask = cv2.imread(mask_path)[:,:,2]
        coord = cv2.imread(coord_path)[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = coord[:, :, 2]
        
        bbox = None
        if('real' in self.data_type):
            attrs = np.array(self.attrs[self.obj_ids[index]+f"_{self.num_segs}"])
            bbox = np.array(self.attrs[self.obj_ids[index]+f"_{self.num_segs}_bbox"])
        else:
            attrs = np.array(self.attrs[self.obj_labels[index]][self.obj_ids[index]])

        seg_mask = (mask==self.mask_ids[index])
        horizontal_indicies = np.where(np.any(seg_mask, axis=0))[0]
        vertical_indicies = np.where(np.any(seg_mask, axis=1))[0]
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
        rmin, rmax, cmin, cmax = get_bbox([y1, x1, y2, x2])
    
        outlier_points = None

        is_mask_hole = 0
        if(is_mask_hole):
            c_x, c_y = (rmax+rmin)//2, (cmax+cmin)//2
            r = np.random.randint(5,(rmax-rmin)//3+5)
            l_x, l_y = max(c_x-r, rmin), max(c_y-r, cmin)
            # r = (rmax-rmin)//2
            r_x, r_y = min(c_x+r, rmax), min(c_y+r, cmax)
            # print('cropped size', r_x-l_x, r_y-l_y)
            if(comp_depth is not None):
                comp_depth[l_x:r_x, l_y:r_y] = 0
            seg_mask[l_x:r_x, l_y:r_y] = 0

        if(self.bg_aug and comp_depth is not None and aug_type == 0):
            lm = rmax-rmin
            cut_lim = min(lm, 25)
            x_cut = np.random.randint(1, cut_lim)
            y_cut = np.random.randint(1, cut_lim)
            x_offset = np.random.randint(0, x_cut)
            y_offset = np.random.randint(0, y_cut)
            x_l = rmin + x_offset
            x_r = x_l + lm - x_cut
            y_l = cmin + y_offset
            y_r = y_l + lm - y_cut
            bbox_mask = np.zeros_like(seg_mask)
            bbox_mask[x_l:x_r, y_l:y_r] = 1
            seg_mask = np.logical_and(bbox_mask, seg_mask)
            noise_mask = np.logical_xor(bbox_mask, seg_mask)
            if(np.sum(noise_mask) != 0 and np.sum(np.logical_and(noise_mask, comp_depth>0))>0):
                outlier_points, _ = backproject(comp_depth, self.intrinsics, noise_mask)

        if(np.sum(seg_mask) == 0):
            seg_mask = (mask==self.mask_ids[index])
        depth = depth.astype(np.float32)
        points,idxs = backproject(depth, self.intrinsics, seg_mask)
        nocs_points = coord[idxs[0], idxs[1], :] - 0.5
        
        scale = self.obj_scales[index]
        R = sciR.from_quat(self.obj_Qs[index]).as_matrix()
        t = self.obj_ts[index]
        choose = np.random.choice(points.shape[0], self.num_points, True)
        points = points[choose]
        trans_points = (R.T @ (points - t).T/scale).T
        trans_points[:,2] *= -1
        nocs_points = nocs_points[choose]
        cate_sym = self.sym_dict[self.obj_labels[index]]

        if(cate_sym):
            all_rots = []
            for d in range(0,360,cate_sym):
                all_rots.append(sciR.from_euler('xyz',[0,d,0],degrees=True).as_matrix())
            all_rots = np.stack(all_rots, axis=0)
            n_attrs = np.transpose(all_rots @ attrs[:,1:].T, (0,2,1))
            n_attrs = n_attrs.reshape((-1, attrs.shape[0], 3))
            n_rs = np.zeros((n_attrs.shape[0], n_attrs.shape[1])) + np.expand_dims(attrs[:,0], 0)
            gt_label = compute_state_sym(n_attrs, np.expand_dims(nocs_points,0))
            nocs_scale = 1
        else:  
            gt_label = compute_state(attrs, nocs_points)
            nocs_scale = 1

        if(self.bg_aug and comp_depth is not None and aug_type == 1):
            dilate_times = np.random.randint(0,6)
            dilate_mask = [[False, True, False], [True, True, True], [False, True, False]]
            o_mask = (mask==self.mask_ids[index])
            for i in range(dilate_times):
                o_mask = ndimage.binary_dilation(o_mask, dilate_mask)
            outlier_mask = np.logical_xor(o_mask, mask==self.mask_ids[index])
            if(np.sum(np.logical_and(outlier_mask, comp_depth>0))!=0):
                outlier_points,_ = backproject(comp_depth, self.intrinsics, outlier_mask)

        if(outlier_points is not None):
            choose = np.random.choice(outlier_points.shape[0], self.num_points//5, True)
            outlier_points = outlier_points[choose]

            points = np.concatenate((points,outlier_points), axis=0)
            # print(gt_label.shape, (attrs.shape[0]+np.zeros_like(choose)).shape)
            if(cate_sym):
                gt_label = np.concatenate((gt_label, attrs.shape[0]+np.zeros((gt_label.shape[0], choose.shape[0]))), axis=1)
            else:
                gt_label = np.concatenate((gt_label, attrs.shape[0]+np.zeros_like(choose)))
            choose = np.random.choice(points.shape[0], self.num_points, True)
            points = points[choose]
            if(cate_sym):
                gt_label = gt_label[:,choose]
            else:
                gt_label = gt_label[choose]
            gt_label = gt_label.astype(np.int32)
                
        if(self.sym == 0):
            gt_label = np.expand_dims(gt_label, 0)
        if(cate_sym == 0 and self.sym!=0):
            gt_label = np.expand_dims(gt_label, 0)
            gt_label = gt_label.repeat(360//self.sym,axis=0)

        points /= self.norm_scale
        is_jitter = self.jitter
        if 'train' in self.data_type and is_jitter:
            # point shift
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            add_t = add_t + np.clip(0.001*np.random.randn(points.shape[0], 3), -0.005, 0.005)
            points = np.add(points, add_t)
        points = points.astype(np.float32)
        return {'points': points, 'points_label': gt_label, 'onehot':np.array(self.onehot_dict[self.obj_labels[index]]), 'cate_sym':np.array(cate_sym), 'category':np.array(self.obj_labels[index], dtype=np.int32)}

    def __len__(self):
        return len(self.obj_ids)
        
        
def get_data_loaders(args):
    print(args.categories)
    train_dataset = NOCS_dataset(args)
    train_sampler = ShuffleWarpSampler(train_dataset, n_repeats=args.train.num_repeats)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train.batch_size, sampler=train_sampler, shuffle=False, num_workers=args.train.num_workers, drop_last=True, collate_fn=np_collate_dict, worker_init_fn=init_np_seed, persistent_workers=True)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train.batch_size, sampler=train_sampler, shuffle=False, num_workers=args.train.num_workers, pin_memory=True, drop_last=True, collate_fn=np_collate_dict, worker_init_fn=init_np_seed)

    loaders = {
        'train_loader': train_loader,
        'train_dataset': train_dataset,
    }
    
    return loaders