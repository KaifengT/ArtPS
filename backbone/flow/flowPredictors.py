
import os
import sys
sys.path.append(os.path.dirname(__file__))


from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch import Tensor

from .nflows.customs.flows import Flow, Glow
# from tools.rotationConvert import batch_rotMat2R6d_tensor, batch_rotR6d2Mat_tensor
import torch.nn.functional as F

# =====================
class FlowPredictor(nn.Module):
    """Base of flow predictor for hand prediction"""
    flow : Flow

    def __init__(self) -> None:
        super().__init__()
    
    # ------------
    def get_inputs_log_prob(self, ctx:Tensor, **kwargs
                            ) -> Tuple[Tensor]:
        raise NotImplementedError('.')
    
    # ------------
    def generate_random_samples(self, ctx:Tensor, num_samples:int
                                ) -> Tuple[Tensor]:
        raise NotImplementedError('.')




# =====================
class GlowRmtPredictor(FlowPredictor):
    def __init__(self,
                 context_features:int,
                 hidden_features:int,
                 num_layers: int=4,
                 num_blocks_per_layer: int=2,
                 z_dim: int=7,
                 noise_scale: float=0.025,
                 **kwargs) -> None:
        super().__init__()

        # self.num_points = num_points
        self.z_dim = z_dim
        self.c_dim = context_features
        self.noise_scale = noise_scale

        self.flow = Glow(features=self.z_dim,
                         hidden_features=hidden_features,
                         context_features=context_features,
                         num_layers=num_layers,
                         num_blocks_per_layer=num_blocks_per_layer,
                         **kwargs)


    # -----------------
    def get_inputs_log_prob(self, ctx:Tensor, tgt_space:Tensor,) -> Tuple[Tensor]:
        """
        In:
            ctx: [B, c_dim], input context
            tgt_space: [B, z_dim], rotmat of mano hand, (..., n_samples, n_j, rotmat_3x3)
        Out:
            log_p: [B,]
            z: [B, z_dim] 
        """
        
        # inputs check
        # B, N, J = rmt.shape[:3]
        # assert J == self.num_joints, f'Expect {self.num_joints} joints, but get {J}'
        # assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        # ctx = ctx.reshape(B, 1, self.c_dim).repeat(1, N, 1) # [B, N, c_dim]
        # r6d = batch_rotMat2R6d_tensor(rmt).reshape(B, N, -1) # [B, N, J*6]
        B, _ = tgt_space.shape
        

        if self.training: # add noise to relax overfit
            tgt_space = tgt_space + torch.randn_like(tgt_space) * self.noise_scale
            
        # if usepool:
        #     ctx = F.max_pool1d(ctx.permute(0, 2, 1), self.num_points).permute(0, 2, 1).repeat(1, N, 1)
        # else:
        # ctx = ctx.repeat(1, N, 1)
        # ctx = ctx.reshape(B, 1, self.c_dim).repeat(1, N, 1) # [B, N, c_dim]
        # point = point[:, None, :, :].repeat(1, N, 1, 1).reshape(B, N, -1)
        

        log_p, z = self.flow.log_prob(tgt_space,
                                      ctx)
        
        log_p = log_p / self.z_dim # mean the prob to each compenent.
        z = z.reshape(B, self.z_dim)

        return log_p, z
    

    # ----------------
    def generate_random_samples(self, ctx:Tensor, num_samples:int, usepool=True) -> Tuple[Tensor]:
        """
        In:
            ctx: [B, c_dim]
            num_samples: int
        Out:
            rmt: [B, N, J, 3, 3], the first sample is always z0 sample
            r6d: [B, N, J, 6]
            log_p: [B, N], the log_p of generated samples
        """

        # assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

        B = ctx.shape[0]
        

        # randomly generate inputs in flow's distribution, and the first sample is always the z0 sample.
        # zs :Tensor = self.flow._distribution.sample(num_samples-1, ctx).requires_grad_(False) # zs samples
        
        # sort zs by prob
        zs, zs_p = self.flow._distribution.sample_and_log_prob(num_samples-1, ctx)
        zs, zs_p = zs.requires_grad_(False), torch.exp(zs_p.requires_grad_(False))
        indices_dec = torch.argsort(zs_p, dim=1, descending=True)
        zs = torch.gather(zs, dim=1, index=indices_dec.unsqueeze(-1).repeat(1, 1, self.z_dim))
        
        
        z0 :Tensor = self.flow._distribution.mean.reshape(1, 1, self.z_dim).repeat(B, 1, 1) # z0 sample, i.e. mode point.
        z = torch.cat([z0, zs], dim=1).to(device=ctx.device).requires_grad_(True)
        
        point, log_p, _ = self.flow.sample_and_log_prob(num_samples, z, ctx) # [B, N, z_dim], [B, N]
        
        point = point.reshape(B, num_samples, self.z_dim)

        # rmt = batch_rotR6d2Mat_tensor(r6d.clone().reshape(-1, 6)).reshape(B, num_samples, -1, 3, 3)
        # r6d = r6d.reshape(B, num_samples, -1, 6)
        log_p = log_p.reshape(B, num_samples) / self.z_dim

        return point, log_p






# =====================
# class GlowRmtBetaCamPredictor(FlowPredictor):
#     def __init__(self,
#                  context_features:int,
#                  hidden_features:int,
#                  num_layers: int=4,
#                  num_blocks_per_layer: int=2,
#                  num_joints: int=16,
#                  camera_base: Dict={'avg':[1, 1e2, 1e2], # avg of [s, tx, ty]
#                                     'std':[1, 1e2, 1e2]},
#                 noise_scale: float=0.025,
#                  **kwargs) -> None:
#         super().__init__()

#         self.num_joints = num_joints
#         self.z_dim = num_joints * 6 + 10 + 3 # rmt + beta + cam
#         self.c_dim = context_features
#         self.noise_scale = noise_scale

#         self.flow = Glow(features=self.z_dim,
#                          hidden_features=hidden_features,
#                          context_features=context_features,
#                          num_layers=num_layers,
#                          num_blocks_per_layer=num_blocks_per_layer,
#                          **kwargs)

#         self.register_buffer('camera_avg', torch.tensor(camera_base['avg'], dtype=torch.float32).reshape(1, 1, 3)) # expand B,N dim.
#         self.register_buffer('camera_std', torch.tensor(camera_base['std'], dtype=torch.float32).reshape(1, 1, 3))


#     # -----------------
#     def get_inputs_log_prob(self, ctx:Tensor, rmt:Tensor, beta:Tensor, cam:Tensor
#                             ) -> Tuple[Tensor]:
#         """
#         In:
#             ctx: [B, c_dim], input context
#             rmt: [B, N, J, 3, 3], rotmat of mano hand, (..., n_samples, n_j, rotmat_3x3)
#             beta: [B, N, 10], shape of mano hand 
#             cam: [B, N, 3], weakcam of mano hand
#         Out:
#             log_p: [B, N]
#             z: [B, N, z_dim] 
#         """
        
#         # inputs check
#         B, N, J = rmt.shape[:3]
#         assert J == self.num_joints, f'Expect {self.num_joints} joints, but get {J}'
#         assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

#         ctx = ctx.reshape(B, 1, self.c_dim).repeat(1, N, 1) # [B, N, c_dim]
#         r6d = batch_rotMat2R6d_tensor(rmt).reshape(B, N, -1) # [B, N, J*6]
#         cam = (cam - self.camera_avg) / self.camera_std # normalized

#         inputs = torch.cat([r6d, beta, cam], dim=-1) # [B, N, z_dim]
#         if self.training: # add noise to relax overfit
#             inputs = inputs + torch.randn_like(inputs) * self.noise_scale 

#         log_p, z = self.flow.log_prob(inputs.reshape(B*N, -1),
#                                       ctx.reshape(B*N, -1)) # [B*N]
        
#         log_p = log_p.reshape(B, N) / self.z_dim # mean the prob to each compenent.
#         z = z.reshape(B, N, self.z_dim)

#         return log_p, z
    

#     # ----------------
#     def generate_random_samples(self, ctx:Tensor, num_samples:int
#                                 ) -> Tuple[Tensor]:
#         """
#         In:
#             ctx: [B, c_dim]
#             num_samples: int
#         Out:
#             rmt: [B, N, J, 3, 3], the first sample is always z0 sample
#             r6d: [B, N, J, 6]
#             beta: [B, N, 10]
#             cam: [B, N, 3]
#             log_p: [B, N], the log_p of generated samples
#         """

#         assert ctx.shape[1] == self.c_dim, f'Expect context features {self.c_dim}, but get {ctx.shape[1]}'

#         B = ctx.shape[0]

#         # randomly generate inputs in flow's distribution, and the first sample is always the z0 sample.
#         zs :Tensor = self.flow._distribution.sample(num_samples-1, ctx).requires_grad_(False) # zs samples
#         z0 :Tensor = self.flow._distribution.mean.reshape(1, 1, self.z_dim).repeat(B, 1, 1) # z0 sample, i.e. mode point.
#         z = torch.cat([z0, zs], dim=1).to(device=ctx.device).requires_grad_(True)

#         outputs, log_p, _ = self.flow.sample_and_log_prob(num_samples, z, ctx) # [B, N, z_dim], [B, N]

#         idx_beta = self.num_joints * 6
#         idx_cam = idx_beta + 10
#         r6d, beta, cam = outputs[..., :idx_beta], outputs[..., idx_beta:idx_cam], outputs[..., idx_cam:]

#         rmt = batch_rotR6d2Mat_tensor(r6d.clone().reshape(-1, 6)).reshape(B, num_samples, -1, 3, 3)
#         r6d = r6d.reshape(B, num_samples, -1, 6)
#         beta = beta.reshape(B, num_samples, 10)
#         cam  = cam.reshape(B, num_samples, 3) * self.camera_std + self.camera_avg
#         log_p = log_p.reshape(B, num_samples) / self.z_dim

#         return rmt, r6d, beta, cam, log_p











if __name__ == '__main__':

    # model = GlowRmtBetaCamPredictor(
    #     context_features=1024,
    #     hidden_features=512,
    # )

    # B, N = 2, 201

    # ctx = torch.randn([B, 1024])
    # rmt = torch.randn([B, N, 16, 3, 3])
    # beta = torch.randn([B, N, 10])
    # cam = torch.randn([B, N, 3])

    # log_p, z = model.get_inputs_log_prob(ctx, rmt, beta, cam)
    # outs = model.generate_random_samples(ctx, num_samples=123)
    # for o in outs:
    #     print(o.shape)
    

    model = GlowRmtPredictor(
        context_features=128,
        hidden_features=512,
    )

    B, N = 2, 201

    ctx = torch.randn([B, 1024, 128])
    rmt = torch.randn([B, 1024, 3])

    log_p, z = model.get_inputs_log_prob(ctx, rmt, N)
    print(log_p.shape)
    outs = model.generate_random_samples(ctx, num_samples=1)
    for o in outs:
        print(o.shape)
        
    print(outs[1])