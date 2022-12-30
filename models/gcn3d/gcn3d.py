import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./models/gcn3d")
import gcn3d_utils as gcn3d

class get_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        class_num = cfg.num_classes
        support_num = cfg.support_num
        neighbor_num = cfg.neighbor_num
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        dim_fuse = sum([128, 128, 256, 256, 512, 512, 6])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 1024, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self, 
                vertices: "tensor (bs, vetice_num, 3)", 
                onehot: "tensor (bs, cat_num)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)
        fm_1 = F.relu(self.conv_1(neighbor_index, vertices, fm_0), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)

        fm_2 = F.relu(self.conv_2(neighbor_index, v_pool_1, fm_pool_1), inplace= True)
        fm_3 = F.relu(self.conv_3(neighbor_index, v_pool_1, fm_2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global, onehot], dim= 2)

        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input) 
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch)
        pred = F.log_softmax(pred, dim=-1)
        return pred

class get_loss(torch.nn.Module):
    def __init__(self, cfg):
        super(get_loss, self).__init__()
        self.cate_dict = {2876657: "bottle",
                          2880940: "bowl",
                          2942699: "camera",
                          2946921: "can",
                          3642806: "laptop",
                          3797390: "mug"}

    def forward(self, pred, target, cate_sym, category):
        # print(pred.shape, target.shape)
        nsym_batch = (cate_sym == 0)
        nsym_pred = pred[nsym_batch]
        nsym_target = target[nsym_batch]
        sym_pred = pred[~nsym_batch]
        sym_target = target[~nsym_batch]
        sym = target.shape[1]
        loss = torch.zeros((0)).cuda()
        if(nsym_pred.shape[0]):
            bs = nsym_pred.shape[0]
            nsym_pred = nsym_pred.contiguous().view(-1, nsym_pred.shape[-1])
            nsym_target = nsym_target[:,0,:].contiguous().view(-1)
            nsym_loss = F.nll_loss(nsym_pred, nsym_target, reduction='none').view(bs,-1)
            nsym_loss = torch.mean(nsym_loss, dim=-1)
            loss = torch.cat((loss, nsym_loss), dim=-1)
        if(sym_pred.shape[0]):
            bs = sym_pred.shape[0]
            sym_pred = sym_pred.unsqueeze(1).repeat(1,sym,1,1).view(-1,sym_pred.shape[-1])
            sym_loss = F.nll_loss(sym_pred, sym_target.view(-1), reduction='none')
            # print(loss.shape)
            sym_loss = sym_loss.view(bs,sym,-1)
            sym_loss = torch.mean(sym_loss, dim=-1)
            # print(loss.shape)
            sym_loss = torch.min(sym_loss, dim=-1)[0]
            loss = torch.cat((loss, sym_loss), dim=-1)
            # print(loss.shape)
        mean_loss = torch.mean(loss)
        losses = {'loss':mean_loss}
        for key,val in self.cate_dict.items():
            cate_loss = loss[category==key]
            if(cate_loss.shape[0] == 0):
                continue
            losses[f'{val}_loss'] = torch.mean(cate_loss)

        return losses

def test():
    model = get_model(class_num= 50, support_num= 1, neighbor_num= 50)

if __name__ == "__main__":
    test()