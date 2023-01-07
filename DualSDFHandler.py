import time
import yaml
import math
import torch
import open3d
import argparse
import importlib
import numpy as np
from skimage import measure
from scipy.spatial.transform import Rotation as R

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from visualization_utils import draw_points

#marching cube
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out=None,
    scale = None,
    offset = None
):
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().numpy()

    level = 0.03
    # print(numpy_3d_sdf_tensor.shape)
    # print(np.max(numpy_3d_sdf_tensor), np.min(numpy_3d_sdf_tensor))
    level = min(level, numpy_3d_sdf_tensor.max())
    level = max(level, numpy_3d_sdf_tensor.min())
    verts, faces, normals, values = measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3, method='lewiner',step_size=1
    )

    # print(verts.shape, faces.shape, normals.shape)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    mesh = open3d.geometry.TriangleMesh(open3d.utility.Vector3dVector(mesh_points), open3d.utility.Vector3iVector(faces))
    mesh.vertex_normals = open3d.utility.Vector3dVector(normals)
    mesh.compute_triangle_normals()
    # open3d.visualization.draw_geometries([mesh])
    return mesh

class DualSDFHandler:
    def __init__(self, args, cfg):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda:0')
        
        trainer_lib = importlib.import_module(cfg.trainer.type)
        self.trainer = trainer_lib.Trainer(cfg, args, self.device)
    
        if args.pretrained is not None:
            self.trainer.resume_demo(args.pretrained)
        else:
            self.trainer.resume_demo(cfg.resume.dir)
        self.trainer.eval()
    
    def sample(self, id, is_std = False, feature = None):
        if(is_std):
            feature = torch.zeros((1,128)).cuda()
        elif feature is not None:
            feature = torch.from_numpy(feature).cuda()
        else:
            feature = self.trainer.get_known_latent(id)
        attrs = self.trainer.prim_attr_net(feature.detach()).detach().cpu().numpy()
        attrs = attrs.reshape(-1, 4)
        return attrs
    
    def export_mesh(self, feature, N = 256):
        feature = torch.from_numpy(feature).float().cuda()
        points, dists = self.trainer.extract_sdf(feature, N)
        pts = points[torch.abs(dists[:,0])<0.003]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts.cpu().numpy())
        dists = dists.reshape((N,N,N))
        f = open3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh = convert_sdf_samples_to_ply(dists, [-1,-1,-1], 2/(N-1))
        # mesh.translate([0,0,0])
        # open3d.visualization.draw_geometries([pcd, mesh ,f])
        return mesh

    def visualize(self, attrs, colors, path):
        def render(pcds, save_dir):
            gui.Application.instance.initialize()
            render = rendering.OffscreenRenderer(640, 480)

            mat1 = rendering.MaterialRecord()
            mat1.shader = 'defaultLit'
            mat1.base_color = np.array([1, 1, 1, 1])*0.85
            
            mat2 = rendering.MaterialRecord()
            mat2.shader = 'defaultLitTransparency'
            mat2.base_color = [0.5,0.5,0.5,0.5]

            render.scene.camera.look_at([0,0,0], [0,0.8,1.7], [0,1,0])
            render.scene.scene.set_sun_light([-1,-1,-1], [1,1,1], 70000)
            # render.scene.scene.enable_sun_light(False)
            # render.scene.show_axes(True)

            for i,pcd in enumerate(pcds):
                render.scene.add_geometry('ellipsoid{}'.format(i), pcd, mat1)

            image = render.render_to_image()
            open3d.io.write_image(save_dir, image, 9)
        
        pcds = draw_points(attrs, colors)
        render(pcds, path)

    def optimize_shape_ransac(self, labels, centers, n_feature, n_ransac = 1, num_step = 100):
        def generate_all_pairs(n):
            x,y = np.meshgrid(range(n), range(n))
            x = x.reshape(-1)
            y = y.reshape(-1)
            valid = (x!=y)
            all_pairs = np.stack((x[valid], y[valid]), axis=1)
            return all_pairs

        def compute_invariant_feature(points, pairs):
            # origins = points[:,pairs[:,0],:]
            # points_a = points[:,pairs[:,1],:]
            # points_b = points[:,pairs[:,2],:]
            # vectors_a = points_a-origins
            # vectors_b = points_b-origins
            # norms_a = torch.norm(vectors_a, dim=-1)
            # norms_b = torch.norm(vectors_b, dim=-1)
            # vectors_dots = torch.sum(vectors_a*vectors_b, dim=-1)
            # feature = vectors_dots / (norms_a * norms_b)

            vectors = points[:, pairs[:,1]] - points[:, pairs[:,0]]
            vectors = vectors.view(vectors.shape[0], -1, vectors.shape[-1])
            vector_norms = torch.norm(vectors, dim=-1)
            vector_norms = vector_norms.unsqueeze(1) * vector_norms.unsqueeze(2)
            vector_dots = vectors @ vectors.permute(0,2,1)
            feature = vector_dots / (vector_norms+1e-8)
            
            return feature.view(feature.shape[0],-1)

        shape_loss = torch.nn.MSELoss()
        nlabel = labels.shape[0]
        all_pairs = generate_all_pairs(nlabel)
        centers = torch.from_numpy(centers).type(torch.FloatTensor).cuda()
        # print(all_pairs.shape)
        all_feature = compute_invariant_feature(centers.unsqueeze(0), all_pairs)
        best_feature = None
        best_ratio = None

        start = time.time()
        while(n_ransac>0):
            n_ransac -= 1
            
            n_pairs = min(n_feature, all_pairs.shape[0])
            choose_idx = np.random.choice(range(all_pairs.shape[0]), n_pairs)

            pairs = all_pairs[choose_idx]
            pairs = torch.from_numpy(pairs).cuda()
            target_feature = compute_invariant_feature(centers.unsqueeze(0), pairs)
            # print("feature_shape", target_feature.shape)

            feature = torch.zeros((1,128)).type(torch.FloatTensor).cuda()
            feature = torch.autograd.Variable(feature, requires_grad=True)

            optimizer = torch.optim.Adam([{'params':feature, 'lr':0.3}])
            for step in range(num_step):
                attrs = self.trainer.prim_attr_net(feature).view(-1,4)
                current_centers = attrs[labels, 1:]
                current_feature = compute_invariant_feature(current_centers.unsqueeze(0), pairs)
                loss = shape_loss(current_feature, target_feature) + 0.0001*torch.sqrt(torch.sum(feature**2)+1e-8)
                if(loss<1e-3):
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            opt_feature = compute_invariant_feature(current_centers.unsqueeze(0), all_pairs)
            # total_loss = shape_loss(all_feature, opt_feature)
            ratio = torch.abs(all_feature-opt_feature)/all_feature
            ratio = torch.sum(ratio<0.05)/ratio.shape[1]

            if(best_feature is None or ratio>best_ratio):
                best_feature, best_ratio = feature, ratio

        end = time.time()

        return best_feature.detach().cpu().numpy(), self.trainer.prim_attr_net(best_feature).detach().view(-1,4).cpu().numpy(), end-start

    # optimize fine-level shape, not used in gcasp project
    def optimize_fine(self, points, pcd=None, init_T = torch.eye(4)):
        def rot_euler2rot(euler):
            assert euler.shape[0] == 3
            R = torch.zeros((euler.shape[0], euler.shape[0]))
            euler = euler * (math.pi / 180)

            cx = torch.cos(euler[0])
            sx = torch.sin(euler[0])
            cy = torch.cos(euler[1])
            sy = torch.sin(euler[1])
            cz = torch.cos(euler[2])
            sz = torch.sin(euler[2])

            R[0, 0] = cz * cy
            R[0, 1] = -sz * cy
            R[0, 2] = sy
            R[1, 0] = cz * sy * sx + sz * cx
            R[1, 1] = -sz * sy * sx + cz * cx
            R[1, 2] = -cy * sx
            R[2, 0] = -cz * sy * cx + sz * sx
            R[2, 1] = sz * sy * cx + cz * sx
            R[2, 2] = cy * cx
            
            return R

        def up(x):
            x_ = torch.zeros((x.shape[0], 3, 3)).cuda()
            x_[:,0,1] = -x[:,2]
            x_[:,0,2] = x[:,1]
            x_[:,1,0] = x[:,2]
            x_[:,1,2] = -x[:,0]
            x_[:,2,0] = -x[:,1]
            x_[:,2,1] = x[:,0]
            return x_
        
        def exp_(x):
            theta = torch.norm(x, 2)
            a = (x/theta)
            I = torch.eye(3).cuda()
            cos_ = torch.cos(theta).cuda()
            sin_ = torch.sin(theta).cuda()
            return cos_*I + (1-cos_)*a @ a.T + sin_*up(a.T)[0]
            
        def getJ(x):
            theta = torch.norm(x, 2)
            a = (x/theta)
            I = torch.eye(3).cuda()
            cos_ = torch.cos(theta).cuda()
            sin_ = torch.sin(theta).cuda()
            return sin_/theta*I + (1-sin_/theta)*a @ a.T + (1-cos_)/theta*up(a.T)[0]
        
        num_step = 3000
        
        feature = torch.zeros((1,128)).cuda()
        feature = torch.autograd.Variable(feature, requires_grad=True)
        
        t = torch.zeros((1,3)).cuda()
        t = torch.autograd.Variable(t, requires_grad=True)
        
        s = torch.tensor(1.0).cuda()
        s = torch.autograd.Variable(s, requires_grad=False)
        
        gt_dist = torch.from_numpy(points[:,3]).type(torch.FloatTensor).cuda().unsqueeze(-1)
        points = torch.from_numpy(points[:,:3]).type(torch.FloatTensor).cuda()
        lr = 0.1
            
        i = torch.tensor([0,0,0,0,0,0,1]).cuda()
        P = torch.zeros((3,4))
        P[:3,:3] = torch.eye(3)
        P = P.unsqueeze(0).cuda()
        T = init_T.cuda()
        # T[2,3] = -0.4
        s = torch.tensor(1.0).cuda()
        
        gtT = torch.eye(4)
        gtT[:3,:3] = rot_euler2rot(torch.tensor([10,0,0]))
        gtT[:3,3] = torch.tensor([0,0,0.1])
        gtT = gtT.cuda()
        
        points = torch.cat((points, torch.ones((points.shape[0], 1)).cuda()), dim=1).unsqueeze(-1)
        # points = gtT @ points
        # print(gtT)
        
        optimizer = torch.optim.Adam([{'params':feature}])
        
        for step in range(num_step):
            
            X = (T @ points)[:,:3,0]
            X = torch.autograd.Variable(X, requires_grad=True)
            
            dists, prim_dists = self.trainer.compute_dists(feature, s * X)
            
            sum_dists = torch.sum(dists)
            
            sum_dists.backward(retain_graph=True)
            
            x_ = torch.zeros((X.detach().shape[0], 4, 7)).cuda()
            x_[:,:3,:3] = torch.eye(3).cuda()
            x_[:,:3,3:6] = -up(X.detach())
            x_[:,:3,6] = X.detach()
            
            err = self.clamped_l1_correct(dists, gt_dist)
            
            dfdx = X.grad
            
            grad = s*i*dists.detach() + (s*dfdx.unsqueeze(-1).transpose(-1,-2) @ P @ x_).squeeze(1)
            
            r = dists.detach() - gt_dist.detach()
            
            grad = torch.sign(r)*grad
            grad = grad[r[:,0] != 0]
            
            grad = -0.01 * torch.mean(grad, dim=0).unsqueeze(-1)
            
            dT = torch.eye(4).cuda()
            dT[:3,:3] = exp_(grad[3:6])
            dT[:3,3:4] = getJ(grad[3:6]) @ grad[:3]
            
            s = s * torch.exp(grad[6,0])

            T = dT @ T
            
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
        return s, T, feature.detach(), err
        # self.visualize(feature.data, gt=pcd,t=t.detach().cpu().numpy()[0], s=s.detach().cpu().numpy())

def get_args(args):
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)

            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    

    args = dict2namespace(args)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config = dict2namespace(config)
    return args, config

def get_instance(args):
    args, cfg = get_args(args)
    runner = DualSDFHandler(args, cfg)
    return runner

if __name__ == "__main__":
    # command line args
    args, cfg = get_args()
    runner = DualSDFHandler(args, cfg)
    
    for i in range(100):
        feature = runner.trainer.get_known_latent(i)
        attrs = runner.trainer.prim_attr_net(feature.detach()).view(-1,4).cpu().numpy()
        
        rot_matrix = R.from_euler('xyz', np.random.rand(3)*360).as_matrix()
        trans_matrix = np.random.rand(3)
        # attrs[:,1:] = (rot_matrix @ attrs[:,1:].T).T + trans_matrix
        
        labels = np.random.choice(attrs.shape[0], 100, replace=False)
        centers = attrs[labels,1:]
        
        pred_feature, pred_attrs = runner.optimize_shape(labels, centers)
        
        pcds = []
        for j in range(attrs.shape[0]):
            pcd = open3d.geometry.TriangleMesh.create_sphere()
            pcd.compute_vertex_normals()
            pcd.scale(np.exp(attrs[j,0]), [0,0,0])
            pcd.translate(attrs[j,1:])
            pcds.append(pcd)
            
            pcd = open3d.geometry.TriangleMesh.create_sphere()
            pcd.compute_vertex_normals()
            pcd.scale(np.exp(pred_attrs[j,0]), [0,0,0])
            pcd.translate(pred_attrs[j,1:]+[3,0,0])
            pcds.append(pcd)
        open3d.visualization.draw_geometries(pcds)