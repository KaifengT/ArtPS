
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import optimizer

import torch.nn.functional as F
from backbone.flow import GlowRmtPredictor
from backbone.pointnet import PointNet2, FixedPointNetFPModule, PointnetSAModule
from backbone.conv import EquivariantLayer
from backbone.mamba3d.mamba3d import Mamba3D, Mamba3d_config
from tools.utils import *


print('-'*50)
VERSION = __file__.split('/')[-1].split('.')[0]
print('Model Version: ', VERSION)



class JointBranch(nn.Module):
    """MLP-like structure"""
    def __init__(self, in_channels, out_channels_list, activation, normalization, momentum=0.1,
                 bn_momentum_decay_step=None, bn_momentum_decay=1, output_init_radius=None):
        super(JointBranch, self).__init__()

        self.layers = nn.ModuleList()
        previous_out_channels = in_channels
        for i, c_out in enumerate(out_channels_list):
            if i != len(out_channels_list) - 1:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, activation, normalization,
                                                    momentum, bn_momentum_decay_step, bn_momentum_decay))
            else:
                self.layers.append(EquivariantLayer(previous_out_channels, c_out, None, None))
            previous_out_channels = c_out

        if output_init_radius is not None:
            self.layers[len(out_channels_list) - 1].conv.bias.data.uniform_(-1 * output_init_radius, output_init_radius)

    def forward(self, x, epoch=None):
        for layer in self.layers:
            x = layer(x, epoch)
        return x
        


class base_pem(nn.Module):
    def __init__(self, 
                    
                    mamba_config:dict,
                    glow_config:dict,
                    

                    use_pretrain:bool = True,
                    num_parts:int = 3,
                    base_sample:int = 1024,
                    joint_type:list = [],
                    beta_dim=64,
                    
                    reverse_on_training=False,
                    
                    sdf_decoder=None,
                    
                    
                    
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.version = VERSION
        self.num_parts = num_parts
        
        self.joint_type = joint_type
        
        self.base_sample = base_sample
        self.num_joints = num_parts - 1
        
        self.context_features = glow_config.context_features
        
        self.beta_dim = beta_dim
        
        self.sdf_decoder = sdf_decoder
        
        
        self.reverse_on_training = reverse_on_training
        
        self.backbone = Mamba3D(mamba_config, dense=True)
        
        if use_pretrain:
            print('Load pretrain model')
            self.backbone.load_model_from_ckpt('backbone/mamba3d/ckpt-last.pth')
        
        self.glow = GlowRmtPredictor(context_features=glow_config.context_features, 
                                     hidden_features=glow_config.hidden_features, 
                                     z_dim=9+1*(num_parts-1) + self.beta_dim + 1, 
                                     noise_scale=glow_config.noise_scale,
                                     num_layers=glow_config.num_layers,
                                     num_blocks_per_layer=glow_config.num_blocks
                                     )

        
        self.fp_module_1 = FixedPointNetFPModule(mlp=[384+384, 384, 384])
        self.fp_module_2 = FixedPointNetFPModule(mlp=[384+384, 384, 192])
        self.fp_module_3 = FixedPointNetFPModule(mlp=[192, 192, 192, 192])


        self.cls_1 = EquivariantLayer(192, 128, activation='relu', normalization='batch')
        self.cls_2 = EquivariantLayer(128, self.num_parts, activation=None, normalization=None)
        
        
        self.joint_loc_layer_1 = EquivariantLayer(384, 256, activation='relu', normalization='batch')
        self.joint_loc_layer_2 = EquivariantLayer(256, 128, activation='relu', normalization='batch')
        self.joint_loc_layer_3 = EquivariantLayer(128, self.num_joints * 3, activation=None, normalization=None)
        
        self.joint_loc_att_1 = EquivariantLayer(384, 128, activation='relu', normalization='batch')
        self.joint_loc_att_2 = EquivariantLayer(128, self.num_joints, activation='sigmoid', normalization=None)
        
        self.joint_axis_layer_1 = EquivariantLayer(384, 256, activation='relu', normalization='batch')
        self.joint_axis_layer_2 = EquivariantLayer(256, 128, activation='relu', normalization='batch')
        self.joint_axis_layer_3 = EquivariantLayer(128, self.num_joints * 3, activation=None, normalization=None)
        
        self.joint_axis_att_1 = EquivariantLayer(384, 128, activation='relu', normalization='batch')
        self.joint_axis_att_2 = EquivariantLayer(128, self.num_joints, activation='sigmoid', normalization=None)
        
                
        
        
        
        
        



        
    def forward(self, x, **kwargs):
        self.sdf_decoder.eval()
        rdata = {}
        global_feature, block_features, group_input_tokens, center, center_idx = self.backbone(x)
        
        # block_features_0 = block_features[0].clone()
        # block_features_1 = block_features[1].clone()
        ''' --------- Dense feature propagation --------- '''
        block_features[1] = self.fp_module_1(center, center, block_features[1], block_features[2])
        block_features[0] = self.fp_module_2(center, center, block_features[0], block_features[1])
        dense_feature = self.fp_module_3(x, center, None, block_features[0])#.transpose(1, 2).contiguous()
        
        ''' --------- Segmentation --------- '''
        dense_seg_score = self.cls_2(self.cls_1(dense_feature)).transpose(1, 2).contiguous() # (B, N, num_parts)
        
        B, N, _ = dense_seg_score.shape
        base_idx = 0
        
        
        seg_detached = torch.max(dense_seg_score.clone().detach(), dim=-1)[1]
        
        base_cls_idxss = seg_detached == base_idx
        
        # solution 1
        # base_feat = torch.where(base_cls_idxss[:, None, :].repeat(1, dense_feature.shape[1], 1), dense_feature, torch.zeros_like(dense_feature)) # B, 192, N
        # base_global_feature = torch.sum(base_feat, dim=-1, keepdim=True) / base_cls_idxss.sum(-1, keepdim=True).unsqueeze(-1) # B, 192, 1
        # # because there may be 0 in base_cls_idxss.sum(-1, keepdim=True)
        # base_global_feature = torch.nan_to_num(base_global_feature, nan=0.0, posinf=0.0, neginf=0.0).squeeze(-1) # (B, 192)
        
        
        # solution 2
        base_feat = torch.where(base_cls_idxss[:, None, :].repeat(1, dense_feature.shape[1], 1), dense_feature, torch.ones_like(dense_feature, device=dense_feature.device)) # B, 192, N
        base_global_feature = base_feat.mean(dim=-1) # B, 192
        
        
        if self.context_features == 384:
            feature_for_glow = torch.cat((base_global_feature, dense_feature.mean(dim=-1)), dim=-1)
        elif self.context_features == 1216:
            feature_for_glow = torch.cat((base_global_feature, global_feature), dim=-1)
        elif self.context_features == 704:
            feature_for_glow = torch.cat((base_global_feature, self.get_ri_feature(x)), dim=-1)
        
        
        
        ''' --------- split input point (x) for each part --------- '''
        
        
        ''' --------- KeyPoint for all Parts  --------- '''
        
        feature_for_kp = block_features[1].clone()
        
        
        
        ''' --------- Observed space joint loc and axis regress  --------- '''
        NF = feature_for_kp.shape[-1]
        joint_loc_offset = self.joint_loc_layer_3(self.joint_loc_layer_2(self.joint_loc_layer_1(feature_for_kp))).transpose(1, 2).contiguous()
        joint_loc_offset = joint_loc_offset.reshape(B, joint_loc_offset.shape[1], self.num_joints, 3)
        joint_loc_reg = center.clone().detach().unsqueeze(-2) - joint_loc_offset
        
        joint_loc_score = self.joint_loc_att_2(self.joint_loc_att_1(feature_for_kp)).transpose(1, 2).reshape(B, NF, self.num_joints, 1)
        joint_loc_prob = F.softmax(joint_loc_score, dim=1)
        joint_loc_reg = torch.sum(joint_loc_reg * joint_loc_prob, dim=1)
        
        joint_axis_reg = self.joint_axis_layer_3(self.joint_axis_layer_2(self.joint_axis_layer_1(feature_for_kp))).transpose(1, 2).contiguous()
        joint_axis_reg = joint_axis_reg.reshape(B, joint_axis_reg.shape[1], self.num_joints, 3)
        
        joint_axis_score = self.joint_axis_att_2(self.joint_axis_att_1(feature_for_kp)).transpose(1, 2).reshape(B, NF, self.num_joints, 1)
        joint_axis_prob = F.softmax(joint_axis_score, dim=1)
        joint_axis_reg = torch.sum(joint_axis_reg * joint_axis_prob, dim=1)
        
        rdata.update({
            'joint_loc_reg':joint_loc_reg,
            'joint_axis_reg':joint_axis_reg,
            }
        )
        ''' --------- Shape deformation Beta  --------- '''
        # beta = self.beta_3(self.beta_2(self.beta_1(global_feature.unsqueeze(-1)))).squeeze(-1)
        # beta = self.beta_3(self.beta_2(self.beta_1(rif))).squeeze(-1)
        # canonical_global_kp = self.get_norm_keypoints(beta)
        
        ''' --------- Joint Params form Beta  --------- '''
        # norm_joint_loc, norm_joint_axis = self.get_norm_joint_params(beta)
        
        
        ''' --------- Glow Prediction  --------- '''
        if self.training:
            
            out = self.flow_reverse(feature_for_glow, kwargs['tgt_distrub'])
                        
            rdata.update(out)
            
            if self.reverse_on_training:
                out = self.flow_forward(feature_for_glow, x, dense_seg_score.clone().detach(), kwargs['num_sample'])
                rdata.update(out)
                            
            
        else:
            
            out = self.flow_forward(feature_for_glow, x, dense_seg_score, kwargs['num_sample'])
            
            rdata.update(out)
       
        
        rdata.update({

            'global_feature':global_feature,
            'base_global_feature':base_global_feature,
            'dense_seg_score':dense_seg_score,
        })
            
        
        return rdata
    
    
    
    def get_ri_feature(self, x):
        
        
        # rif = torch.zeros(x.shape[0], 512, 1, device=x.device)
        # for part_id in range(self.num_parts):
        #     this_part_idx = seg == part_id
        #     this_part_point = torch.where(this_part_idx.unsqueeze(-1), x, torch.nan)
            
        #     # this_part_point.nan
        #     # point_max = torch.max(torch.nan_to_num(this_part_point, nan=-torch.inf), dim=-2, keepdim=True)[0]
        #     # point_min = torch.min(torch.nan_to_num(this_part_point, nan=torch.inf), dim=-2, keepdim=True)[0]
        #     # this_part_center = (point_max + point_min) / 2
            
        #     this_part_center = torch.nanmean(this_part_point, dim=-2, keepdim=True,)
        #     this_part_point_decenter = this_part_point - this_part_center
        #     this_part_point_decenter = torch.nan_to_num(this_part_point_decenter, nan=0.0, posinf=0.0, neginf=0.0)
        #     this_part_norm = compute_LRA(this_part_point_decenter, True)
            
        #     # data.update({
        #     #     f'this_part_point_decenter_{part_id}':this_part_point_decenter.clone().detach().cpu().numpy(),
        #     #     f'this_part_norm_line{part_id}':torch.stack((this_part_point_decenter, this_part_point_decenter+this_part_norm), dim=-2).clone().detach().cpu().numpy(),
        #     # })
            
        #     rif += self.ricore[part_id](this_part_point_decenter.detach(), this_part_norm.detach())
            
        # with open('tempfile/test.pkl', 'wb') as f:
        #     import pickle
        #     pickle.dump(data, f)
        
        norm = compute_LRA(x, True)
        
        rif = self.ricore(x.detach(), norm.detach())
        
        return rif.squeeze(-1)
        
    def flow_reverse(self, condition, tgt_distrub):
        rdata = {}
        log_p, z = self.glow.get_inputs_log_prob(condition, tgt_distrub,)
        
        rdata.update({
            'loss_gen':-log_p.mean(),
        })
                    
        return rdata

    def flow_forward(self, condition, cloud, dense_seg_score, top_k_sample):
        rdata = {}
        NP = self.num_parts
        B = condition.shape[0]
        
        
        pred_token, log_p = self.glow.generate_random_samples(condition, num_samples=self.base_sample)
        
        pred_token = pred_token[:, :top_k_sample]
        log_p = log_p[:, :top_k_sample]
        
        base_pose, beta, joint_state, scale = self.DEtokenlize_articulation_2(pred_token, cloud=cloud, part_cls=dense_seg_score)
        
        # base_pose B,S,4,4
        # beta B,S,beta_dim
        # joint_state B,S,P
        # scale B,S,1
        
        joint_loc, joint_axis = self.sdf_decoder.forward_joint(beta.reshape(-1, self.beta_dim))
        
        # joint_loc/axis B, S, (P-1), 3
        
        
        joint_loc = joint_loc.reshape(B, top_k_sample, -1, 3)
        joint_axis = joint_axis.reshape(B, top_k_sample, -1, 3)
        
        # joint_axis_len = torch.linalg.norm(joint_axis, dim=-1, keepdim=True)
        # joint_axis = joint_axis / joint_axis_len
        
        # FIXME joint_loc is in normalized space, need to divid scale befrore decode_articulation
    
        part_pose = self.decode_articulation(base_pose,
                                             joint_loc / scale.unsqueeze(-2),
                                             joint_axis,
                                             joint_state,
                                             joint_type=self.joint_type)
        pred_pose = torch.cat([base_pose.unsqueeze(-3), part_pose], dim=-3)
        
        rdata.update({
            'base_pose':base_pose,
            'pred_pose':pred_pose,
            'pred_token':pred_token,
            'log_p':log_p,
            'beta':beta,
            'joint_loc':joint_loc,
            'joint_axis':joint_axis,
            'joint_state':joint_state,
            'scale':scale,
        })
        
        return rdata



    @staticmethod
    def get_reflection_operator(n_pl):
        """ 
        Modified from OMAD
        The reflection operator is parametrized by the normal vector
        of the plane of symmetry passing through the origin. """
        norm_npl = torch.norm(n_pl, 2)
        n_x = n_pl[0, 0] / norm_npl
        n_y = torch.tensor(0.0, device=n_pl.device)
        n_z = n_pl[0, 1] / norm_npl
        refl_mat = torch.stack(
            [
                1 - 2 * n_x * n_x,
                -2 * n_x * n_y,
                -2 * n_x * n_z,
                -2 * n_x * n_y,
                1 - 2 * n_y * n_y,
                -2 * n_y * n_z,
                -2 * n_x * n_z,
                -2 * n_y * n_z,
                1 - 2 * n_z * n_z,
            ],
            dim=0,
        ).reshape(1, 3, 3)

        return refl_mat

    @staticmethod
    def cls_loss(input, target, weight=1.0):
        cls_loss = weight * torch.mean(F.cross_entropy(input.view(-1, input.shape[-1]), target.view(-1)))
        return cls_loss
    
    
    @staticmethod
    def gen_loss(log_p):
        return -log_p.mean()

    @staticmethod        
    def kp_loss(pred_canonical_kp:torch.Tensor,
                pred_observed_kp:torch.Tensor,
                pred_prob_kp:torch.Tensor,
                gt_rt:torch.Tensor,
                gt_canonical_kp:torch.Tensor,
                kp_weight:dict,
                kp_vis:torch.Tensor=None,
                kp_seg_mask:torch.Tensor=None,
                ):

        _, NSKP, NP, _, _ = pred_observed_kp.shape
        inv_gt_per_part_rt = torch.linalg.inv(gt_rt).unsqueeze(1)
        
        ''' project observed kp to canonical space '''
        pred_observed_global_kp = pred_observed_kp
        pred_observed_global_kp_in_canonical = transform_points(inv_gt_per_part_rt, pred_observed_global_kp)
        
        ''' calculate dis between projected observed kp and gt kp in canonical space '''
        # pred_weighted_kp_in_canonical = transform_points(inv_gt_per_part_rt, pred_weighted_kp)
        observed_global_kp_dis = torch.norm(gt_canonical_kp.unsqueeze(1) - pred_observed_global_kp_in_canonical, dim=-1)
        
        ''' mask the dis use visibility '''
        if kp_vis is not None:
            observed_global_kp_dis = observed_global_kp_dis * kp_vis.detach().unsqueeze(1)
        
        
        '''mask the dis use seg '''
        if kp_seg_mask is not None:
            observed_global_kp_dis = observed_global_kp_dis * kp_seg_mask.unsqueeze(-1)
        
        observed_global_kp_loss = observed_global_kp_dis.mean()
        
        
        ''' calculate loss between predicted gt kp in canonical space '''
        canonical_global_kp_loss = torch.mean(torch.mean(torch.norm(gt_canonical_kp - pred_canonical_kp, dim=-1), dim=-1), dim=-2).mean()
        consistent_kp_loss = torch.norm(pred_observed_global_kp_in_canonical - pred_canonical_kp.detach().unsqueeze(1), dim=-1).mean()
        
        
        # confi_loss = torch.norm(gt_canonical_kp.unsqueeze(1) - pred_observed_global_kp_in_canonical.detach(), dim=-1, keepdim=True) * pred_prob_kp - 0.01 * torch.log(pred_prob_kp)
        # confi_loss = torch.mean(confi_loss)
        confi_loss = 0
        # weighted_kp_loss = torch.mean(torch.mean(torch.norm(gt_canonical_kp.unsqueeze(1) - pred_weighted_kp_in_canonical, dim=-1), dim=-1), dim=-2).mean()

        kpo_loss = kp_weight['weight_kpo'] * observed_global_kp_loss
        kpc_loss = kp_weight['weight_kpc'] * canonical_global_kp_loss
        kpco_loss = kp_weight['weight_kpco'] * consistent_kp_loss
        confi_loss = kp_weight['weight_cfi'] * confi_loss
        
        return kpo_loss + kpc_loss + kpco_loss + confi_loss

    @staticmethod
    def joint_loss(pred_joint_loc:torch.Tensor, 
                   pred_joint_axis:torch.Tensor, 
                   gt_joint_loc:torch.Tensor, 
                   gt_joint_axis:torch.Tensor, 
                   joint_type:list):
        '''
        Args:
            pred_joint_loc: [B, num_joints, 3]
            pred_joint_axis: [B, num_joints, 3]
            gt_joint_loc: [B, num_joints, 3]
            gt_joint_axis: [B, num_joints, 3]
            joint_type: list of joint types, e.g., ['revolute', 'prismatic']
        Return:
            loss_joint_param: scalar tensor, the joint loss
        
        '''
        
        
        loss_joint_param_list = []
        
        num_parts = gt_joint_loc.shape[-2] + 1
        
        eps = 1e-8
        
        for part_idx in range(1, num_parts): # Only count for child part
            
            jtype = joint_type[part_idx - 1]
            
            loss_joint_axis = 1 - F.cosine_similarity(pred_joint_axis[:, part_idx - 1, :],
                                                gt_joint_axis[:, part_idx - 1, :], dim=-1).mean(dim=-1).mean(dim=-1)
            norm_gt_joint_axis = gt_joint_axis[:, part_idx - 1, :] / torch.norm(gt_joint_axis[:, part_idx - 1, :],
                                                                                dim=-1, keepdim=True)
            if jtype == 'revolute':
                p = gt_joint_loc[:, part_idx - 1, :]
                q = gt_joint_loc[:, part_idx - 1, :] + norm_gt_joint_axis
                r = pred_joint_loc[:, part_idx - 1, :]
                x = p - q
                loss_joint_loc = torch.norm(
                    ((r - q) * x).sum(-1, keepdim=True) / ((x * x).sum(-1, keepdim=True) + eps) * (p - q) + (q - r),
                    dim=-1).mean(-1).mean(-1)
                loss_joint_param = loss_joint_loc + loss_joint_axis
            elif jtype == 'prismatic':
                loss_joint_param = loss_joint_axis
            else:
                raise NotImplementedError(f"Joint type {jtype} is not implemented.")
            loss_joint_param_list.append(loss_joint_param)
            
        return torch.mean(torch.stack(loss_joint_param_list))


        
    @staticmethod
    def decode_articulation(base_pose, norm_joint_loc, norm_joint_axis, joint_state, with_base=True, joint_type=[]):
        '''
        Args:
            base_pose (torch.Tensor) (B, 4, 4): base part pose in world space
            norm_joint_loc (torch.Tensor) (B, P-1, 3): joint pivot in normalized space
            norm_joint_axis (torch.Tensor) (B, P-1, 3): joint axis in normalized space
            joint_state (torch.Tensor) (B, P): joint state, e.g., angle for revolute joint, distance for prismatic joint
            with_base (bool): if True, the base part is included in the output
            joint_type (list): list of joint types, e.g., ['revolute', 'prismatic', ...]

        assumption: all parts connected to the base part
        
        Returns:
            part_pose (torch.Tensor) (B, P, 4, 4)
        '''
        B, J = norm_joint_loc.shape[0], norm_joint_loc.shape[-2]
        

        
        part_pose_list = []
        for joint_id in range(J):
            
            jtype = joint_type[joint_id]
        
            tmat1 = torch.eye(4, device=base_pose.device).expand_as(base_pose).clone()
            tmat1[..., :3, 3] = -norm_joint_loc[..., joint_id, :] # B 4 4
            
            
            aug_matrix = torch.eye(4, device=base_pose.device).expand_as(base_pose).clone() # B 4 4
            
            if jtype == 'revolute':
                # FIXME: norm_joint_axis in OMAD dataset is negative
                aug_matrix[..., :3, :3] = axis_angle_to_matrix( - norm_joint_axis[..., joint_id, :] * joint_state[..., joint_id+1:joint_id+2]) # B 4 4
                # aug_matrix[..., :3, :3] = axis_angle_to_matrix( norm_joint_axis[..., joint_id, :] * joint_state[..., joint_id+1:joint_id+2]) # B 4 4
            elif jtype == 'prismatic':
                aug_matrix[..., :3, 3] = - norm_joint_axis[..., joint_id, :] * joint_state[..., joint_id+1:joint_id+2]
            else:
                raise NotImplementedError
            
            tmat2 = torch.eye(4, device=base_pose.device).expand_as(base_pose).clone()
            tmat2[..., :3, 3] = norm_joint_loc[..., joint_id, :] # B 4 4
            
            if with_base:
                part_pose = base_pose @ tmat2 @ aug_matrix @ tmat1
            else:
                part_pose = tmat2 @ aug_matrix @ tmat1
            
            part_pose_list.append(part_pose[..., None, :, :])
        
        part_pose = torch.cat(part_pose_list, dim=-3)
        
        return part_pose

    def tokenlize_articulation_2(self, rt, beta, joint_state, scale, **kwargs):
        '''
        rt (B, 4, 4)
        joint_state (B, P)
        beta (B, 10)
        '''
        B = rt.shape[0]
        base_token = tokenlize_pose(rt) # base rt B 4 4 -> B 9
        # joint_loc_token = joint_loc.reshape(B, -1)
        # joint_axis_token = joint_axis.reshape(B, -1)
        joint_state_token = joint_state[:, 1:]
        
        token = torch.cat([base_token, beta, joint_state_token, scale], dim=-1) # B 9 + 1 * (P-1) + 1
        
        return token.detach()
    
    def DEtokenlize_articulation_2(self, token, **kwargs):
        '''
        token (B, 9+1*(P-1)+1)
        '''
        B, L = token.shape[0], token.shape[-1]
        batch_like_shape = token.shape[:-1]
        
        base_token = token[..., :9]
        beta_end = 9 + self.beta_dim
        beta = token[..., 9:beta_end]
        # joint_loc_token = token[..., 9:9+3*(num_parts-1)]
        # joint_axis_token = token[..., 9+3*(num_parts-1):9+6*(num_parts-1)]
        joint_state_token = token[..., beta_end:beta_end+(self.num_parts-1)]
        scale = token[..., beta_end+(self.num_parts-1):beta_end+(self.num_parts-1)+1]
        # beta = token[..., 9+(num_parts-1):]
        # beta = torch.zeros((*batch_like_shape, 10), device=token.device)
        
        base_pose = DEtokenlize_pose(base_token)
        # joint_loc = joint_loc_token.reshape(*batch_like_shape, -1, 3)
        # joint_axis = joint_axis_token.reshape(*batch_like_shape, -1, 3)
        joint_state = torch.zeros((*batch_like_shape, self.num_parts), device=token.device)
        joint_state[..., 1:] = joint_state_token
        
        return base_pose, beta, joint_state, scale



    
    
    
    def get_observed_kp_mean(self, pred_observed_global_kp, center_seg_idx=None):

        if center_seg_idx is None:
            pred_observed_global_kp_mean = pred_observed_global_kp.mean(dim=1)
            return pred_observed_global_kp_mean
        
        
        pred_observed_global_kp_vaild_mask = center_seg_idx == 1
        
        pred_observed_global_kp_vaild = torch.where(pred_observed_global_kp_vaild_mask[..., None, None], pred_observed_global_kp, torch.nan)
        pred_observed_global_kp_mean = pred_observed_global_kp_vaild.nanmean(dim=1) # B, N, KP/P, 3
        pred_observed_global_kp_mean = torch.nan_to_num(pred_observed_global_kp_mean, 0)
        
        return pred_observed_global_kp_mean



class base_pem_rel_trans(base_pem):

    def tokenlize_articulation_2(self, rt, joint_state, beta, scale, cloud, part_cls, ):
        
        part_cls_detach = part_cls.detach().clone()
        cloud_detach = cloud.detach().clone()

        
        gt_seg_onehot = F.one_hot(part_cls_detach.long(), num_classes=self.num_parts)
        mask_splits = torch.split(gt_seg_onehot, split_size_or_sections=1, dim=2)
        
        cloud_base_masked = cloud_detach * mask_splits[0].float()
        centroid = cloud_base_masked.sum(dim=-2, keepdim=True) / mask_splits[0].sum(dim=-2, keepdim=True)
        
        centroid = torch.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0).detach()
        
        centroid_rt = rt.clone()
        centroid_rt[..., :3, 3] -= centroid[..., 0, :]
        
        return super().tokenlize_articulation_2(rt=centroid_rt, beta=beta, joint_state=joint_state, scale=scale)

    def DEtokenlize_articulation_2(self, token, cloud, part_cls, ):
                
        part_cls_detach = part_cls.detach().clone()
        cloud_detach = cloud.detach().clone()
        
        base_cls_idxss = torch.max(part_cls_detach.long(), dim=-1)[1] == 0
        cloud_base_masked = torch.where(base_cls_idxss.unsqueeze(-1).expand_as(cloud_detach), cloud_detach, torch.zeros_like(cloud_detach))
        
        
        centroid = cloud_base_masked.sum(dim=-2, keepdim=True) / base_cls_idxss.sum(dim=-1, keepdim=True).unsqueeze(-1)
        centroid = torch.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        
        new_token = token.clone()
        new_token[..., 6:9] += centroid.detach()
        
        return super().DEtokenlize_articulation_2(new_token)