import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDF2Points():
    def __init__(self, device):
        self.device = device

    def generate_points_within_sphere(self, num_sphere_samples = 5192):
        points = torch.empty([num_sphere_samples,3], dtype=torch.float32, device=self.device) # x y z
        r = torch.empty([num_sphere_samples,1], dtype=torch.float32, device=self.device) # x y z
        points.normal_(0, 1)
        points = points / torch.sqrt(torch.sum(points**2, dim=-1, keepdim=True))
        r.uniform_(0, 1) # r = 1
        points = points * (r**(1/3))
        return points

    def extract(self, sdf_fun, coloridx=None, colorcoord=None):
        with torch.no_grad():
            pts = self.generate_points_within_sphere()
            dists = sdf_fun(pts)
            points = pts[torch.abs(dists['dists'][:,0])<0.02]

        return points

    def extract_sdf(self, sdf_fun, N = 64, coloridx=None, colorcoord=None):
        with torch.no_grad():
            x = (torch.tensor(range(N))-N/2)/(N/2)*1.1
            x,y,z = torch.meshgrid(x,x,x)
            step = 5096*2
            pts = torch.cat((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))), dim=1).cuda()
            # pts = torch.tensor(torch.meshgrid(x,x,x), device = self.device)
            # print(pts.shape)
            res = []
            for i in tqdm(range(0, pts.shape[0], step), desc="exporting mesh"):
                r = i + step
                if(r > pts.shape[0]):
                    r = pts.shape[0]
                res.append(sdf_fun(pts[i: r])['dists'])
            dists = torch.cat(res, dim = 0)
        return pts, dists
