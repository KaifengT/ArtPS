import torch.nn as nn
import torch
from typing import Dict, Optional, Tuple
from backbone.conv import EquivariantLayer
import torch.nn.functional as F
from backbone.pointnet import PointnetSAModule
from tools.utils import *
import math
from einops import rearrange, einsum, repeat
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn.modules.activation import MultiheadAttention

from jaxtyping import Float
from einops import rearrange, reduce
from torch import Tensor

from tools.grid_sample_2d import custom_grid_sample_2d

# NOTE: This version includes bounding box prediction head, but layers are smaller than V9.


MODEL_FILE = __file__
VERSION = MODEL_FILE.split('/')[-1].split('.')[0]

try:
    from accelerate.state import AcceleratorState
    if AcceleratorState.is_main_process:
        print('Model Version: ', VERSION)
except:
    print('Model Version: ', VERSION)


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from "Attention Is All You Need".
    
    Adds positional encodings to input embeddings of shape (batch_size, seq_len, d_model).
    The positional encodings are fixed (not learned) and based on sine and cosine functions
    of different frequencies [[3]].
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model (embedding size)
            max_len: Maximum sequence length supported
            dropout: Dropout rate applied after adding positional encoding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model // 2)

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions

        # Register as buffer (not a model parameter, not updated during training)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings of shape (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding, with same shape as input
        """
        # x is (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # add positional encoding up to seq_len
        return self.dropout(x)


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

class SDFEncoder(nn.Module):
    def __init__(self, inscode_size=64):
        super().__init__()
        # Define the shared PN++
        self.sa_module_1 = PointnetSAModule(
            npoint=512,
            radius=0.2,
            nsample=64,
            mlp=[0, 64, 64, 128],
            bn=True,
            use_xyz=True,
        )

        self.sa_module_2 = PointnetSAModule(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            bn=True,
            use_xyz=True,
        )

        self.sa_module_3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256, 256, 512, 1024],
            bn=True,
            use_xyz=True,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(256, inscode_size),
            
        )


    def forward(self, input):
        l0_xyz = input[:, :, :3]
        # Here l0_features should be blank
        # l0_features = input[:, :, 3:].transpose(1, 2) if input.size(-1) > 3 else None
        l0_features = None
                                                                    #  B,  N,  C     B,  C,   N
        l1_xyz, l1_features = self.sa_module_1(l0_xyz, l0_features) # [16, 512, 3], [16, 128, 512]
        l2_xyz, l2_features = self.sa_module_2(l1_xyz, l1_features) # [16, 128, 3], [16, 256, 128]
        l3_xyz, l3_features = self.sa_module_3(l2_xyz, l2_features) # None,         [16, 1024, 1]

        return self.fc_layer(l3_features.squeeze(-1))


class SDFEncoder_plus(nn.Module):
    def __init__(self, codeSize=64):
        super().__init__()

        # Stage 1: 512 points
        self.sa1 = PointnetSAModule(
            npoint=512,
            radius=0.2,
            nsample=64,
            mlp=[0, 128, 128, 256],  # +64 from pos emb
            bn=True,
            use_xyz=True,
        )

        # Stage 2: 128 points
        self.sa2 = PointnetSAModule(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[256, 256, 256, 512],
            bn=True,
            use_xyz=True,
        )

        # Stage 3: Global feature (1 point)
        self.sa3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[512, 512, 1024, 1024],
            bn=True,
            use_xyz=True,
        )

        # Multi-scale feature fusion (optional but powerful)
        self.fuse_layer = nn.Sequential(
            nn.Linear(1024 + 512 + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )

        # Enhanced head with residual blocks
        self.head = nn.Sequential(
            ResMLPBlock(1024, 512, hidden_dim=2048, dropout=0.1),
            ResMLPBlock(512, codeSize, hidden_dim=256, dropout=0.1),
        )

    def forward(self, input):
        """
        input: [B, N, C]
        """

        xyz = input[:, :, :3]  # [B, N, 3]


        l0_xyz = xyz

        # Set abstraction
        l1_xyz, l1_features = self.sa1(l0_xyz, None)  # [B, 512, 3], [B, 256, 512]
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)  # [B, 128, 3], [B, 512, 128]
        _,      l3_features = self.sa3(l2_xyz, l2_features)       # [B, 1024, 1]

        # Global features
        global_feat = l3_features.squeeze(-1)  # [B, 1024]

        # Optional: fuse with lower-level global pooled features (multi-scale)
        l2_global = F.adaptive_max_pool1d(l2_features, 1).squeeze(-1)  # [B, 512]
        l1_global = F.adaptive_max_pool1d(l1_features, 1).squeeze(-1)  # [B, 256]

        fused = torch.cat([global_feat, l2_global, l1_global], dim=1)  # [B, 1024+512+256]
        fused = self.fuse_layer(fused)  # [B, 1024]

        return self.head(fused)

class ResMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int]=2048, nn_activation: nn.Module = nn.GELU, dropout: float = 0.1, eps: float = 1e-5):
        super().__init__()
        '''
        (*, dim) -> (*, dim)
        '''
        if hidden_dim is None:
            hidden_dim = in_dim
        if in_dim != out_dim:
            self.jump = nn.Linear(in_dim, out_dim)
        else:
            self.jump = nn.Identity()
            
        self.activation = nn_activation()
        self.mlp = nn.Sequential(
            
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn_activation(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=eps),
            nn_activation(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim, eps=eps),
            nn_activation(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.activation(self.mlp(x) + self.jump(x))
    
class MLPBlock(nn.Module):    
    def __init__(self, in_dim: int, out_dim: int, 
                 normalization: bool = True,
                 dropout: float = 0.1, 
                 activation: nn.Module = nn.GELU,
                 eps: float = 1e-5):
        super().__init__()
        '''
        (*, in_dim) -> (*, out_dim)
        '''
        if normalization:
            norm = nn.LayerNorm(out_dim, eps=eps)
        else:
            norm = nn.Identity()
            
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm,
            activation(),
            nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
    
    
class ResidualFusionMLP(nn.Module):
    def __init__(self, x_dim: int, memory_dim: int, nhead: int = 8, hidden_dim: int = 2048, dropout: float = 0.1):
        """
        Args:
            memory_dim(int): K, V dim
            x_dim(int): Q dim
            
            (B, S, x_dim) , (B, L, memory_dim) -> (B, S, memory_dim)
        """
        super().__init__()
        
        assert memory_dim % nhead == 0, "memory_dim must be divisible by nhead"
        
        if x_dim != memory_dim:
            self.proj_2_memory = nn.Linear(x_dim, memory_dim)
        else:
            self.proj_2_memory = nn.Identity()
            
        self.mha = MultiheadAttention(
            memory_dim,
            nhead,
            dropout=dropout,
            batch_first=True  #(B, S, E)
        )
        
        self.ffn = ResMLPBlock(memory_dim, memory_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, S1, x_dim)
            memory  : (B, S2, memory_dim)
            
        Returns:
            out: (B, S1, memory_dim)
        """

        # project feat2 to memory_dim
        x = self.proj_2_memory(x)  # (B, S1, memory_dim)
        fused = self.mha(x, memory, memory)[0] # (B, S1, memory_dim)
        fused = fused + x  # Residual connection
        out = self.ffn(fused)  
        return out    

# UnTested
class FrqPositionalEncoding(nn.Module):
    def __init__(self, in_dim=3, num_freqs=10):
        super().__init__()
        self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.out_dim = in_dim * 2 * num_freqs

    def forward(self, x):
        # x: (..., 3)
        x_proj = x.unsqueeze(-1) * self.freq_bands.to(x.device)  # (..., 3, L)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(-2, -1)

# UnTested
class RBFEncoder(nn.Module):
    def __init__(self, num_centers=16, min_val=-0.1, max_val=0.1, gamma=10.0):
        super().__init__()
        self.centers = torch.linspace(min_val, max_val, num_centers)
        self.gamma = gamma
        self.out_dim = num_centers

    def forward(self, x):
        # x: (..., 1)
        x = x.unsqueeze(-1)  # (..., 1, 1)
        centers = self.centers.to(x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        rbf = torch.exp(-self.gamma * (x - centers) ** 2)  # (..., 1, C)
        return rbf.squeeze(-2)  # (..., C)
    
def SimplePositionalEncoding_(pos):
    
    '''
    Args:
        pos: (B, N, 3)
    Return:
        pos_enc: (B, N, 6)
    '''
    sin_pos = torch.sin(pos)  # (B, N, 3)
    cos_pos = torch.cos(pos)  # (B, N, 3)
    r = torch.linalg.norm(pos, dim=-1, keepdim=True)  # (B, N, 1)
    pos_enc = torch.cat([pos, sin_pos, cos_pos, r], dim=-1)  # (B, N, 10)
    return pos_enc


def query_triplane(
    positions: Float[Tensor, "*B N 3"],
    triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    reduction: str = "concat",
) -> Tensor:
    '''
    From https://github.com/VAST-AI-Research/TriplaneGaussian
    '''
    batched = positions.ndim == 3
    if not batched:
        # no batch dimension
        triplanes = triplanes[None, ...]
        positions = positions[None, ...]

    indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
            (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
            dim=-3,
        )
    out: Float[Tensor, "B3 Cp 1 N"] = custom_grid_sample_2d(
        rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
        rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
        align_corners=False,
        mode="bilinear",
    )
    if reduction == "concat":
        out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
    elif reduction == "mean":
        out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
    else:
        raise NotImplementedError
    
    if not batched:
        out = out.squeeze(0)

    return out



class Decoder(nn.Module):
    def __init__(
        self,
        num_catcodes:int,
        catcode_size:int,
        inscode_size:int,
        # num_parts,
        dropout=0.1
    ):
        super(Decoder, self).__init__()

        
        self.num_catcodes = num_catcodes  # e.g., 256
        self.catcode_size = catcode_size  # e.g., 256
        self.inscode_size = inscode_size  # e.g., 64
        
        self.triplane_dim = 32  # H, W = 32
        
        # self.catecode_pos_enc = PositionalEncoding(d_model=catcode_size, max_len=3 * self.triplane_dim**2, dropout=dropout)
        
        self._catcode :Float[Tensor, "1 3 Ct Hp Wp"] = torch.nn.Parameter(
            torch.randn(1, 3, self.catcode_size, self.triplane_dim, self.triplane_dim, dtype=torch.float32) * 1. / np.sqrt(self.catcode_size),
        )

        # Project Func 1
        # self.inscode_proj = ResMLPBlock(self.inscode_size, self.catcode_size, hidden_dim=2048, dropout=dropout)
        
        # Project Func 2
        # self.inscode_proj = nn.Sequential(
        #     ResMLPBlock(self.inscode_size, 256, normalization=True, dropout=dropout, activation=nn.GELU),
        #     ResMLPBlock(256, self.catcode_size * (self.triplane_dim**2), normalization=True, dropout=dropout, activation=nn.GELU),
        # )
        # Project Func 3
        self.inscode_proj = nn.Sequential(
            nn.Linear(self.inscode_size, self.inscode_size),
            nn.LayerNorm(self.inscode_size),
            nn.ReLU(),
            nn.Linear(self.inscode_size, self.inscode_size // 4 * 4 * 4),
            
            # reshape to feature map
            nn.Unflatten(1, (self.inscode_size // 4, 4, 4)),
            
            nn.Conv2d(self.inscode_size // 4, self.inscode_size // 8, 3, padding=1),
            nn.GroupNorm(max(1, self.inscode_size // 64), self.inscode_size // 8), 
            nn.ReLU(),
            
            nn.Upsample(size=(self.triplane_dim, self.triplane_dim), mode='bilinear'),
            
            nn.Conv2d(self.inscode_size // 8, self.catcode_size, 3, padding=1),
            nn.GroupNorm(max(1, self.catcode_size // 8), self.catcode_size),
            nn.ReLU(),
            
            nn.Conv2d(self.catcode_size, 6 * self.catcode_size, 3, padding=1)
        )
        # self.point_proj_1 = nn.Sequential(
        #     # MLPBlock(10, 64, normalization=False, dropout=0.0, activation=nn.GELU()),
        #     ResMLPBlock(10, 64, hidden_dim=64, dropout=dropout),
        #     ResMLPBlock(64, self.catcode_size, hidden_dim=256, dropout=dropout)
        # )

        # self.fuse_1 = ResidualFusionMLP(x_dim=self.catcode_size, memory_dim=self.catcode_size, nhead=8, hidden_dim=2048, dropout=dropout)
        # self.fuse_2 = ResidualFusionMLP(x_dim=self.catcode_size, memory_dim=self.catcode_size, nhead=8, hidden_dim=2048, dropout=dropout)

        
        self.out_project = nn.Sequential(
            ResMLPBlock(in_dim=3*self.catcode_size,  out_dim=512, hidden_dim=1024, dropout=dropout),
            ResMLPBlock(in_dim=512,                  out_dim=256, hidden_dim=1024, dropout=dropout),
            ResMLPBlock(in_dim=256,                  out_dim=64,  hidden_dim=256, dropout=dropout),
            nn.Linear(64, 1)
        )

        # self.num_parts = num_parts

    def forward(self, cloud:torch.Tensor, inscode:torch.Tensor):
        '''
        Args:
            cloud: (B, N, 3) query points
            inscode: (B, I) instance codes
        '''
        
        # Stage1. Merge catcode and inscode.        
        fused_code: Float[Tensor, "B 3 Ct Hp Wp"] = self._get_Fused_code(inscode)
        
        # Stage2. Query features from triplane
        
        qcode: Float[Tensor, "B N (3 Ct)"] = query_triplane(
            positions=cloud,
            triplanes=fused_code,
            reduction="concat",
            )  # (B, N, catcode_size)        
        
        return self.out_project(qcode), fused_code
    
    
    
    def _get_Fused_code(self, inscode:torch.Tensor):
        '''
        Args:
            inscode: (B, inscode_size) instance codes
        Return:
            fused_code: (B, num_catcodes, catcode_size)
        '''
        copied_catcode = repeat(self._catcode, "1 Np Ct Hp Wp -> B Np Ct Hp Wp", B=inscode.shape[0])
        patch_inscode = self.inscode_proj(inscode)

        patch_inscode = rearrange(patch_inscode, "B (Np Ct) Hp Wp -> B Np Ct Hp Wp", Np=3)
        gamma, beta = patch_inscode.chunk(2, dim=2) 
        fused_code = copied_catcode * (1 + gamma) + beta  # FiLM
        
        
        
        
        
        return fused_code
    
    def _proj_code_for_joint(self, fused_code:torch.Tensor):
        '''
        Args:
            fused_code: (B, 3, catcode_size, H, W)
        Return:
            feat_flap: (B, catcode_size)
        '''

        return rearrange(fused_code, "B Np Ct Hp Wp -> B Ct (Np Hp Wp)").mean(dim=-1)
    

class SDFDecoder(nn.Module):
    def __init__(self, num_catcodes, catcode_size, inscode_size, num_parts, joint_from_inscode=False):
        super().__init__()
        
        CodeInitStdDev = 1.0
        CodeBound = 1.0
        
        self.num_parts = num_parts
        self.num_joints = num_parts - 1
        self.num_catcodes = num_catcodes
        self.catcode_size = catcode_size
        self.inscode_size = inscode_size
        self.joint_from_inscode = joint_from_inscode
        
        
        self.decoder = Decoder(
            num_catcodes=self.num_catcodes,
            catcode_size=self.catcode_size, 
            inscode_size=self.inscode_size, 
        )

        self.joint_net = JointBranch(self.inscode_size if joint_from_inscode else self.catcode_size, 
                                    [64, 64, self.num_joints * 6],
                                    activation='relu', 
                                    normalization='batch',
                                    momentum=0.1,
                                    bn_momentum_decay_step=None,
                                    bn_momentum_decay=1.0)

        self.bbox_head = nn.Sequential(
            ResMLPBlock(self.catcode_size, self.catcode_size, hidden_dim=self.catcode_size * 2, dropout=0.1),
            nn.Linear(self.catcode_size, self.num_parts * 6)
        )
        
    def forward(self, inscode, query_point, chunk_size=1):
        '''
        Args:
            inscode: [B, inscode_size]
            query_point: [B, N, 3]
        Return:
            dict:
                sdf: [B, N, 1]
                norm_joint_loc: [B, num_joints, 3]
                norm_joint_axis: [B, num_joints, 3]
                norm_bbox_center: [B, num_parts, 3]
                norm_bbox_size: [B, num_parts, 3]
        '''
        
        sdf, feat = self.decoder(query_point, inscode)
        
        code = self.decoder._proj_code_for_joint(feat)
        
        norm_joint_loc, norm_joint_axis = self._forward_joint(inscode if self.joint_from_inscode else code)
        
        norm_bbox_center, norm_bbox_size = self._forward_bbox(code)

        return {
            'sdf': sdf,
            'norm_joint_loc': norm_joint_loc,
            'norm_joint_axis': norm_joint_axis,
            'norm_bbox_center': norm_bbox_center,
            'norm_bbox_size': norm_bbox_size
        }

    def _forward_joint(self, code:torch.Tensor):
        '''
        Args:
            inscode: [B, inscode_size]
        Return:
            norm_joint_loc: [B, num_joints, 3]
            norm_joint_axis: [B, num_joints, 3]
        '''
        norm_joint_params = self.joint_net(code.clone().unsqueeze(-1)).reshape(-1, self.num_joints, 6)
        norm_joint_loc = norm_joint_params[:, :, :3]  # (bs, K-1, 3)
        norm_joint_axis = norm_joint_params[:, :, 3:]  # (bs, K-1, 3)

        norm_joint_axis = norm_joint_axis / torch.linalg.norm(norm_joint_axis, dim=-1, keepdim=True)

        return norm_joint_loc, norm_joint_axis
    
    def _forward_bbox(self, code:torch.Tensor):
        '''
        Args:
            code: [B, catcode_size]
        Return:
            pred_bbox_center: [B, num_parts, 3]
            pred_bbox_size: [B, num_parts, 3]
        '''
        bbox_params = self.bbox_head(code).reshape(-1, self.num_parts, 6)
        pred_bbox_center = bbox_params[:, :, :3]
        pred_bbox_size = bbox_params[:, :, 3:6]
        
        return pred_bbox_center, pred_bbox_size
    
    def forward_bbox(self, inscode:torch.Tensor):
        feat = self.decoder._get_Fused_code(inscode)
        feat = self.decoder._proj_code_for_joint(feat)
        
        pred_bbox_center, pred_bbox_size = self._forward_bbox(feat)
        return pred_bbox_center, pred_bbox_size
    
    
    def forward_joint(self, inscode:torch.Tensor):
        
        if self.joint_from_inscode:
            return self._forward_joint(inscode)
        
        else:
            feat = self.decoder._get_Fused_code(inscode)
            feat = self.decoder._proj_code_for_joint(feat)
            
            return self._forward_joint(feat)
        
    @staticmethod
    def bbox_loss(pred_center, pred_size, gt_center, gt_size, pred_rot_6d=None, gt_rot=None):
        '''
        Args:
            pred_center: [B, num_parts, 3]
            pred_rot_6d: [B, num_parts, 6]
            pred_size: [B, num_parts, 3]
            gt_center: [B, num_parts, 3]
            gt_rot: [B, num_parts, 3, 3]
            gt_size: [B, num_parts, 3]
        '''
        loss_center = F.smooth_l1_loss(pred_center, gt_center)
        loss_size = F.smooth_l1_loss(pred_size, gt_size)
        
        if pred_rot_6d is None or gt_rot is None:
            return loss_center + loss_size
        
        else:
            pred_rot_mat = rotation_6d_to_matrix(pred_rot_6d)
            # Frobenius norm loss for rotation matrices
            loss_rot = F.mse_loss(pred_rot_mat, gt_rot)
        
            return loss_center + loss_size + loss_rot
        
    

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
                    ((r - q) * x).sum(-1, keepdim=True) / (x * x).sum(-1, keepdim=True) * (p - q) + (q - r),
                    dim=-1).mean(-1).mean(-1)
                loss_joint_param = loss_joint_loc + loss_joint_axis
            elif jtype == 'prismatic':
                loss_joint_param = loss_joint_axis
            else:
                raise NotImplementedError(f"Joint type {jtype} is not implemented.")
            loss_joint_param_list.append(loss_joint_param)
            
        return torch.mean(torch.stack(loss_joint_param_list))




if __name__ == '__main__':
    # Test ResidualFusionMLP
    batch_size = 4
    memory_len = 10
    tgt_len = 5
    memory_dim = 64
    tgt_dim = 32
    out_dim = 128

    # memory = torch.randn(batch_size, memory_len, memory_dim)
    # tgt = torch.randn(batch_size, tgt_len, tgt_dim)

    # fusion_mlp = ResidualFusionMLP(memory_dim, tgt_dim, nhead=8, hidden_dim=2048, dropout=0.1)
    # out = fusion_mlp(tgt, memory)

    # print("Output shape:", out.shape)  # Expected: (batch_size, tgt_len, out_dim)
    
    sdfdecoder = SDFDecoder(num_catcodes=32, catcode_size=256, inscode_size=128, num_parts=2)
    
    inscode = torch.randn(2, 128)
    query_point = torch.randn(2, 1000, 3)
    
    sdfdata = sdfdecoder(inscode, query_point)
    # loss = sdfdecoder.catcode_regularization(l2_weight=1e-4, diversity_weight=1e-3)
    print("SDF Values shape:", sdfdata['sdf'].shape)
    print("Norm Joint Loc shape:", sdfdata['norm_joint_loc'].shape)
    print("Norm Joint Axis shape:", sdfdata['norm_joint_axis'].shape)
    print("Norm BBox Center shape:", sdfdata['norm_bbox_center'].shape)
    print("Norm BBox Size shape:", sdfdata['norm_bbox_size'].shape)
    
    

