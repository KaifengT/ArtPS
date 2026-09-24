from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from model_utils import *

# from utils.mm3d_pn2 import furthest_point_sample, gather_points
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1, num_parts=2):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.num_parts = num_parts
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        
        self.conv_x = nn.Conv1d(3*self.num_parts, 64, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        

        self.channel = channel

        
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        
        
        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.conv_out = nn.Conv1d(64, 3*self.num_parts, kernel_size=1)


    def forward(self, x, coarse, feat_g):
        batch_size, _, P, N = coarse.size()
        coarse = coarse.reshape(batch_size, -1, N).contiguous()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1)

        y1 = self.sa1(y0, y0)
        y2 = self.sa2(y1, y1)
        y3 = self.sa3(y2, y2)
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio)

        y_up = y.repeat(1,1,self.ratio)
        y_cat = torch.cat([y3,y_up],dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio)

        x = x.reshape(batch_size, 3, self.num_parts, N*self.ratio).contiguous()

        return x, y3, y4

class PCT_encoder(nn.Module):
    def __init__(self, channel=64, num_parts=2):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.num_parts = num_parts
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3*self.num_parts, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        # SFA
        x1 = self.sa1_1(x1,x1).contiguous()
        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N
        x2 = torch.cat([x_g1, x2], dim=1)
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()
        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4
        x3 = torch.cat([x_g2, x3], dim=1)
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sa0_d(x, x))
        x1_d = (self.sa1_d(x0_d, x0_d))
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))

        fine = fine.reshape(batch_size, 3, self.num_parts, -1).contiguous()

        return x_g, fine

class Model(nn.Module):
    def __init__(self, args=None, num_parts=0, r1=2, r2=2):
        super(Model, self).__init__()

        self.num_parts = num_parts

        self.encoder = PCT_encoder(num_parts=self.num_parts)

        self.refine = PCT_refine(ratio=r1, num_parts=self.num_parts)
        self.refine1 = PCT_refine(ratio=r2, num_parts=self.num_parts)
        

    def forward(self, x, x_cls, gt=None, is_training=True, only_test=False):
        '''
        X(B, N, 3)
        x_cls (B, N) 0~num_parts-1
        
        '''
        centroid = torch.mean(x, dim=1, keepdim=True)
        x = (x - centroid).transpose(2, 1).contiguous()
        
        
        feat_g, coarse = self.encoder(x)

        B, _, P, N = coarse.shape
        fused = torch.zeros(B, 3, P, 512, device=coarse.device)
        for b in range(coarse.shape[0]):
            for part_id in range(self.num_parts):
                this_part_id = (x_cls[b] == part_id).nonzero().squeeze(-1)
                this_part_x = x[b, :, this_part_id]
                this_part_cat = torch.cat([this_part_x, coarse[b, :, part_id, :]], dim=-1).unsqueeze(0)
                this_part_fused = gather_points(this_part_cat, furthest_point_sample(this_part_cat.transpose(1, 2).contiguous(), 512))
                fused[b, :, part_id, :] = this_part_fused.squeeze(0)


        fine, feat_fine, f00 = self.refine(None, fused, feat_g)
        fine1, feat_fine1, f11 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).transpose(2, 3).contiguous()
        fine = fine.transpose(1, 2).transpose(2, 3).contiguous()
        fine1 = fine1.transpose(1, 2).transpose(2, 3).contiguous()
        
    

        if only_test:
            return fine1 + centroid.unsqueeze(1)

        gt = gt - centroid.unsqueeze(1)
        pbatch_fine1 = fine1.reshape(-1, fine1.shape[-2], 3).contiguous()
        pbatch_fine = fine.reshape(-1, fine.shape[-2], 3).contiguous()
        pbatch_coarse = coarse.reshape(-1, coarse.shape[-2], 3).contiguous()
        pbatch_gt = gt.reshape(-1, gt.shape[-2], 3).contiguous()

        if is_training:
            
            loss3, _ = calc_cd(pbatch_fine1, pbatch_gt)
            
            gt_fine1 = gather_points(pbatch_gt.transpose(1, 2).contiguous(), furthest_point_sample(pbatch_gt, pbatch_fine.shape[1])).transpose(1, 2).contiguous()
            loss2, _ = calc_cd(pbatch_fine, gt_fine1)
            
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, pbatch_coarse.shape[1])).transpose(1, 2).contiguous()
            loss1, _ = calc_cd(pbatch_coarse, gt_coarse)
            
            # dist = torch.cdist(fine1, gt, p=2)
            # min_dist1, id1 = torch.min(dist, dim=-1)
            # min_dist2, id2 = torch.min(dist, dim=-2)
            # min_dist, id = torch.min(min_dist1 + min_dist2, dim=-1)
            # fine_gt_cls = torch.gather(gt_cls, 1, id1).detach()
            
            # cls_loss = torch.mean(F.cross_entropy(dense_seg_score.view(-1, dense_seg_score.shape[-1]), fine_gt_cls.view(-1)))

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return fine + centroid.unsqueeze(1), loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(pbatch_fine1, pbatch_gt)
            cd_p_coarse, cd_t_coarse = calc_cd(pbatch_coarse, pbatch_gt)

            # seg_mask = torch.max(dense_seg_score, dim=-1)[1].unsqueeze(-1)

            return {'out1': coarse + centroid.unsqueeze(1), 
                    'out2': fine1 + centroid.unsqueeze(1), 
                    'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}

