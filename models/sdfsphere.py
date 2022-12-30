import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rot_euler2rot(euler):
    """
    rotation_euler2matrix
    :param euler: 3x1 vector containing the angles in degreee  
    :return: R the 3x3 rotation matrix
    """
    assert euler.shape[-1] == 3
    R = torch.zeros((euler.shape[0], euler.shape[1], euler.shape[2], 3)).cuda()
    euler = euler * (math.pi / 180)

    cx = torch.cos(euler[:,:,0])
    sx = torch.sin(euler[:,:,0])
    cy = torch.cos(euler[:,:,1])
    sy = torch.sin(euler[:,:,1])
    cz = torch.cos(euler[:,:,2])
    sz = torch.sin(euler[:,:,2])

    R[:,:,0, 0] = cz * cy
    R[:,:,0, 1] = -sz * cy
    R[:,:,0, 2] = sy
    R[:,:,1, 0] = cz * sy * sx + sz * cx
    R[:,:,1, 1] = -sz * sy * sx + cz * cx
    R[:,:,1, 2] = -cy * sx
    R[:,:,2, 0] = -cz * sy * cx + sz * sx
    R[:,:,2, 1] = sz * sy * cx + cz * sx
    R[:,:,2, 2] = cy * cx
    return R

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k*a, dim=dim, keepdim=keepdim) / k
    return dmix

class SDFFun(nn.Module):
    def __init__(self, cfg):
        super(SDFFun, self).__init__()
        self.return_idx = cfg.return_idx
        self.smooth = cfg.smooth
        self.smooth_factor = cfg.smooth_factor
        print('[SdfSphere] return idx: {}; smooth: {}'.format(self.return_idx, self.smooth))
        
    # assume we use Sphere primitive for everything
    # parameters: radius[r], center[xyz]
    # def prim_sphere_batched_smooth(self, x, p):
    #     device = x.device
    #     # print(x.shape, p.shape)
    #     # x = x.unsqueeze(-2) # B N 1 3
    #     # p = p.unsqueeze(-3) # B 1 1 3
    #     logr = p[:,:,0]
    #     d = torch.sqrt(torch.sum((x-p[:,:,1:4])**2, dim=-1)) - torch.exp(logr) # B N M 
    #     # d = torch.relu(torch.sum((x**2)/(p[:,:,:]**2), dim=-1)-1) + torch.sqrt(torch.sum(p**2, dim=-1)) # B N M
    #     # center = p[:,:,:3].unsqueeze(-3)
    #     # axes = p[:,:,3:6]
    #     # euler = p[:,:,6:]
    #     # rot = rot_euler2rot(euler)
    #     # axes = torch.diag_embed(axes)
    #     # # print((x-center).shape, rot.shape, axes.shape)
    #     # # print(((x - center) @ rot @ axes).shape)
    #     # d = (x - center) @ axes @ torch.transpose((x-center),-1,-2) - 1
    #     d = d.reshape((d.shape[0], -1))
    #     # print(d.shape)
    #     # d = torch.sum(d, dim=-1, keepdim=True)
    #     # print(d.shape)
    #     return d
    #     if self.return_idx:
    #         d, loc = torch.min(d, dim=-1, keepdim=True)
    #         return d, loc
    #     else:
    #         if self.smooth and False:
    #             d = bsmin(d, dim=-1, k=self.smooth_factor, keepdim=True)
    #         else:
    #             d, _ = torch.min(d, dim=-1, keepdim=True)
    #         print(d.shape)
    #         return d

    def prim_sphere_batched_smooth(self, x, p):
        device = x.device
        # x = x.unsqueeze(-2) # B N 1 3
        # p = p.unsqueeze(-3) # B 1 M 4
        # logr = p[:,:,:,0]
        # center = p[:,:,0:3]
        # t = x-center
        axes = p[:,:,0:3]
        i_axes = 1/axes
        d = torch.norm(i_axes * x, 2, -1)*(torch.norm(i_axes * x, 2, -1)-1)/(torch.norm((i_axes**2) * x, 2, -1)+1e-8)
        # print(d.shape)
        # d = torch.sqrt(torch.sum(((x-p[:,:,:,0:3])**2)/(p[:,:,:,3:6]**2), dim=-1)) - 1 # B N M 
        return d
        if self.return_idx:
            d, loc = torch.min(d, dim=-1, keepdim=True)
            return d, loc
        else:
            if self.smooth:
                d = bsmin(d, dim=-1, k=self.smooth_factor, keepdim=True)
            else:
                d, _ = torch.min(d, dim=-1, keepdim=True)
            return d

    # a: [B M 4]; 
    # x: [B N 3]; x, y, z \in [-0.5, 0.5]
    def forward(self, a, x):
        a = a.reshape(a.size(0), -1, 3)
        out = self.prim_sphere_batched_smooth(x, a)
        return out
