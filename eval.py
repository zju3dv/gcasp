import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import cv2
import sys
import time
import yaml
import json
import tqdm
import torch
import open3d
import datetime
import argparse
import importlib
import numpy as np
import _pickle as cPickle
from NOCS_tools import utils
from NOCS_tools.dataset import NOCSDataset
from NOCS_tools.train import ScenesConfig
from NOCS_tools.aligning import estimateSimilarityTransform
from NOCS_tools.utils import backproject, get_bbox, get_3d_bbox, transform_coordinates_3d
from scipy.spatial.transform.rotation import Rotation as sciR
import DualSDFHandler

parser = argparse.ArgumentParser( description='eval nocs')
parser.add_argument('config', type=str,help='The configuration file.')
parser.add_argument('--pretrained', default=None, type=str,help='pretrained model checkpoint')
parser.add_argument('--data', type=str, help="val/real_test", default='real_test')
parser.add_argument('--draw', dest='draw', action='store_true', help="whether draw and save detection visualization")

args = parser.parse_args()
data = args.data

onehot_dict = { 1: [1,0,0,0,0,0],
                2: [0,1,0,0,0,0],
                3: [0,0,1,0,0,0],
                4: [0,0,0,1,0,0],
                5: [0,0,0,0,1,0],
                6: [0,0,0,0,0,1]}

def get_args():
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    
    return config

def load_model(args, cfg):
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)

    if args.pretrained is not None:
        start_epoch = trainer.resume(args.pretrained)

    trainer.eval()
    
    return trainer

def inference(model, depth, mask, bb, K, shape_handler, class_id):
    def return_with_zero():
        return 1,np.eye(3),np.array([0,0,0]),np.eye(4), np.array([1,1,1]), np.zeros((128))
    
    valid = np.logical_and(mask[:,:]>0, depth>0)
    if(np.sum(valid) == 0):
        return return_with_zero()
    
    points,_ = backproject(depth, K, valid)
    if(points.shape[0] < 200):
        return return_with_zero()

    points = points[np.random.choice(points.shape[0], 1024, True)]
    data = {'points':torch.from_numpy(points).cuda().unsqueeze(0)}

    pred,_ = model.classification(data, np.array(onehot_dict[class_id]))
    pred = torch.exp(pred)[0]
    _, pred_labels = torch.topk(pred,3,dim=1)
    pred_labels = pred_labels.cpu().numpy()
    pred_label = pred_labels[:,0]
    pred_valid = pred_label != num_segs # filter out noise points
    points = points[pred_valid]
    pred_label = pred_label[pred_valid]

    labels = np.unique(pred_label)
    nlabel = min(100,len(labels))

    if(nlabel <= 3):
        return return_with_zero()
    
    choose_labels = labels[np.random.choice(labels.shape[0],nlabel,replace=False)]
    center_points = np.zeros((nlabel, 3))
    for j, label in enumerate(choose_labels):
        center_points[j] = np.mean(points[pred_label==label], axis=0)

    shapecode, attrs, timeused = shape_handler.optimize_shape_ransac(choose_labels, center_points, 1000, 1, num_step=30)

    attrs = process_attrs(attrs, class_id)

    s, R, t, T = estimateSimilarityTransform(points.copy(), attrs[pred_label,1:], verbose=False)
    T[:3,:3] = T[:3,:3].T
        
    z_180_RT = np.zeros((4, 4), dtype=np.float32)
    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
    z_180_RT[3, 3] = 1
    
    Rt = z_180_RT @ np.linalg.inv(T)
    Rt[:3,:] /= 1000

    bbox = np.amax(attrs[:,1:], axis=0) - np.amin(attrs[:,1:], axis=0)

    return s, R, t, Rt, bbox, shapecode

def process_attrs(attrs, class_id=-1): # normalize the estimated shape
    attrs[:,3] *= -1
    attrs[:,2] *= -1
    if(class_id == 5): # rotate the laptop to normalize
        y_ord = np.argsort(attrs[:,1])[-50:]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(attrs[y_ord,1:])
        (a,b,c,d), _ = pcd.segment_plane(0.05, 5, 100)
        if(np.dot([a,b,c], [0,1,0])<0):
            a,b,c = -a,-b,-c
        deg = np.arccos(np.dot([a,b,c], [0,1,0])/np.linalg.norm([a,b,c]))
        R = sciR.from_euler('z',deg).as_matrix()
        attrs[:,1:] = (R @ attrs[:,1:].T).T
    centers1 = attrs[:,1:] - np.exp(attrs[:,0:1])
    centers2 = attrs[:,1:] + np.exp(attrs[:,0:1])
    bb1 = np.min(centers1, axis=0)
    bb2 = np.max(centers2, axis=0)
    center = (bb1+bb2)/2
    attrs[:,1:] -= center
    return attrs

def load_full_depth(image_path):
    depth = cv2.imread(image_path, -1)
    if(depth is None):
        return None
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2]) # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

class InferenceConfig(ScenesConfig): # use nocs config to load test dataset
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    COORD_USE_REGRESSION = False
    COORD_NUM_BINS = 32
    COORD_USE_DELTA = False
    USE_SYMMETRY_LOSS = True
    TRAINING_AUGMENTATION = False
    OBJ_MODEL_DIR = os.path.join('./eval_datas','obj_models')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    cfg = get_args()

    config = InferenceConfig()

    # Load experimental settings
    num_segs = cfg.data.num_segs

    # dataset directories
    root_dir = './eval_datas'
    camera_dir = os.path.join('./', 'eval_datas')
    real_dir = os.path.join(root_dir, 'real')
    coco_dir = os.path.join(root_dir, 'coco')

    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]

    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug', # or can ?
        'laptop': 'laptop',
    }

    to_eval_ids = [] # check which category to evaluate, default: all
    shape_handlers = [] # load dualsdf pretrained models

    for eval_name in cfg.data.names:
        for i,name in enumerate(synset_names):
            if(name == eval_name):
                to_eval_ids.append(i)
                break

    for eval_name in cfg.data.names:
        for i,name in enumerate(synset_names):
            if(name == eval_name):
                shape_handlers.append(DualSDFHandler.get_instance({
                    'config': f"./config/dualsdf{num_segs}.yaml",
                    'pretrained': f"./eval_datas/DualSDF_ckpts/{eval_name}/{num_segs}/epoch_9999.pth"
                }))
                break

    model = load_model(args, cfg)
    # Recreate the model in inference mode

    gt_dir = os.path.join(root_dir,'gts', data)
    mask_dir = os.path.join('./eval_datas/deformnet_eval/mrcnn_results/', data)
    depth_dir = os.path.join('./camera_full_depths/', data)
    
    if data == 'val':
        dataset_val = NOCSDataset(synset_names, 'val', config)
        dataset_val.load_camera_scenes(camera_dir)
        dataset_val.prepare(class_map)
        dataset = dataset_val
    elif data == 'real_test':
        dataset_real_test = NOCSDataset(synset_names, 'test', config)
        dataset_real_test.load_real_scenes(real_dir)
        dataset_real_test.prepare(class_map)
        dataset = dataset_real_test
    elif data == 'real_train':
        dataset_real_train = NOCSDataset(synset_names, 'train', config)
        dataset_real_train.load_real_scenes(real_dir)
        dataset_real_train.prepare(class_map)
        dataset = dataset_real_train
    else:
        assert False, "Unknown data resource."

    image_ids = dataset.image_ids
    now = datetime.datetime.now()
    save_dir = os.path.join('eval_logs', "{}_{:%Y%m%dT%H%M}".format(data, now))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data in ['real_train', 'real_test']:
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    else: ## CAMERA data
        intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

    elapse_times = []

    for i, image_id in enumerate(tqdm.tqdm(image_ids)):
        image_start = time.time()
        image_path = dataset.image_info[image_id]["path"]
        
        path_parse = image_path.split('/')
        mesh_class_ids = []

        image_short_path = '_'.join(path_parse[-3:])
        save_path = os.path.join(save_dir, 'results_{}.pkl'.format(image_short_path))
        mask_path = os.path.join(mask_dir,'results_{}.pkl'.format(image_short_path))
        
        if(not os.path.exists(mask_path)):
            continue
        with open(mask_path, 'rb') as f:
            mrcnn_result = cPickle.load(f)
        # mrcnn_result = {}

        # record results
        result = {}

        # loading ground truth
        image = dataset.load_image(image_id)
        if(data=='val'):
            depth = load_full_depth(os.path.join(depth_dir, path_parse[-2], path_parse[-1]+"_composed.png")) #dataset.load_depth(image_id)
            if(depth is None):
                continue
        else:
            depth = dataset.load_depth(image_id)
        gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label, _ = dataset.load_mask(image_id)
        gt_bbox = utils.extract_bboxes(gt_mask)

        result['image_id'] = image_id
        result['image_path'] = image_path

        result['gt_class_ids'] = gt_class_ids
        result['gt_bboxes'] = gt_bbox
        result['gt_RTs'] = None
        result['gt_scales'] = gt_scales

        image_path_parsing = image_path.split('/')
        gt_pkl_path = os.path.join(gt_dir, 'results_{}_{}_{}.pkl'.format(data, image_path_parsing[-2], image_path_parsing[-1]))

        if (os.path.exists(gt_pkl_path)):
            with open(gt_pkl_path, 'rb') as f:
                gt = cPickle.load(f)
            result['gt_RTs'] = gt['gt_RTs']
            if 'handle_visibility' in gt:
                result['gt_handle_visibility'] = gt['handle_visibility']
                assert len(gt['handle_visibility']) == len(gt_class_ids)
                # print('got handle visibiity.')
            else: 
                result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
        else:
            assert 0, "cannot find gt pose provided by nocs"
            # align gt coord with depth to get RT
            if not data in ['coco_val', 'coco_train']:
                if len(gt_class_ids) == 0:
                    print('No gt instance exsits in this image.')

                print('\nAligning ground truth...')
                start = time.time()
                result['gt_RTs'], _, error_message, _ = utils.align(gt_class_ids, 
                                                                    gt_mask, 
                                                                    gt_coord, 
                                                                    depth, 
                                                                    intrinsics, 
                                                                    synset_names, 
                                                                    image_path,
                                                                    )
                print('New alignment takes {:03f}s.'.format(time.time() - start))
                
                np.save(save_dir+'/'+'{}_{}_{}_gt_pose.npy'.format(data, image_path_parsing[-2], image_path_parsing[-1]), result['gt_RTs'])
                
            result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
        ## detection
        start = time.time()
        elapsed = time.time() - start
        elapse_times = elapsed
        # print('\nDetection takes {:03f}s.'.format(elapsed))

        result['pred_class_ids'] = mrcnn_result['class_ids']
        result['pred_bboxes'] = mrcnn_result['rois']
        result['pred_RTs'] = np.ones((mrcnn_result['rois'].shape[0],4,4))
        result['pred_scales'] = np.ones((mrcnn_result['rois'].shape[0],3))
        result['pred_scores'] = mrcnn_result['scores']
        result['class_ids'] = mrcnn_result['class_ids']
        
        if len(result['class_ids']) == 0:
            print('No instance is detected.')

        # print('Aligning predictions...')
        start = time.time()
        
        for i, eval_id in enumerate(to_eval_ids):
            mask_ids = np.where(result['pred_class_ids']==eval_id)[0].tolist()
            for j,mask_id in enumerate(mask_ids):
                bbox = mrcnn_result['rois'][mask_id]
                bb = get_bbox(bbox)

                s, R, t, T, pred_scale,shapecode = inference(model, depth, mrcnn_result['masks'][...,mask_id], bb, intrinsics, shape_handlers[i], eval_id)
                result['pred_RTs'][mask_id] = T
                result['pred_scales'][mask_id] = pred_scale
        
        elapsed = time.time() - start
        
        # print('New alignment takes {:03f}s.'.format(time.time() - start))
        elapse_times += elapsed

        if args.draw:
            draw_rgb = False
            utils.draw_detections(image, save_dir, data, image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics, synset_names, draw_rgb,
                                    gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                    mrcnn_result['rois'], mrcnn_result['class_ids'], mrcnn_result['masks'], gt_coord, result['pred_RTs'], mrcnn_result['scores'], result['pred_scales'])

        
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)
        # print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))

        # elapsed = time.time() - image_start
        # print('Takes {} to finish this image.'.format(elapsed))
        # print('Alignment average time: ', np.mean(np.array(elapse_times)))
        # print('\n')
