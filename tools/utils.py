import os, sys
import torch
import numpy as np
import random
from tools.rotation import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d
# from pytorch3d.ops import knn_points
import pickle
from scipy.optimize import least_squares
import tabulate
import copy
import h5py
import natsort
import torch.utils.data as data
import torch.nn.functional as F
import traceback
from pointnet2_ops import pointnet2_utils
import matplotlib.pyplot as plt
from itertools import cycle
import cv2
# %matplotlib ipympl
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as mplot3d
# fig = plt.figure()
# ax = mplot3d.Axes3D(fig)
# fig.add_axes(ax)
# ax.scatter([0, 1], [0, 1], [0, 1], s = 0.2)
# ax.view_init(elev=80, azim = 0)
# ax.set_xlabel('X label')
# ax.set_ylabel('Y label')
# ax.set_zlabel('Z label')
# plt.show()

def build_line(end, start=None, dim=-2):
    if start is None:
        start = torch.zeros_like(end)
        
    line = torch.stack([start, end], dim=dim)
    return line

def color_map_numpy(x:np.ndarray, cmap='plasma'):
    cmap = plt.get_cmap(cmap)
    batch_like = x.shape[:-1]
    flatx = x.reshape(-1)
    rgba = cmap(flatx)
    rgba = rgba.reshape(*batch_like, 4)
    
    return rgba


def tokenlize_pose(hrt:torch.Tensor, method='r6d'):
    # hrt: (B, 4, 4)
    # token: (B, 9)    
    assert method in ['r6d', 'quat'], "The method should be 'r6d', 'quat'."
    if method == 'r6d':
        r = matrix_to_rotation_6d(hrt[..., :3, :3])#.reshape(-1, 6)
    elif method == 'quat':
        r = matrix_to_quaternion(hrt[..., :3, :3])
    t3d = hrt[..., :3, 3]#.reshape(-1, 3)
    token = torch.cat([r, t3d], dim=-1) # (B, 9/7)
    return token


def DEtokenlize_pose(token:torch.Tensor, method='r6d'):
    # token: (..., 9,)
    # hrt: (..., 4, 4)
    assert method in ['r6d', 'quat'], "The method should be 'r6d', 'quat'."
    batch_like_shape = token.shape[:-1]
    
    if method == 'r6d':
        rot_mat = rotation_6d_to_matrix(token[..., :6])
        trans3d = token[..., 6:9]
    elif method == 'quat':
        rot_mat = quaternion_to_matrix(token[..., :4])
        trans3d = token[..., 4:7]
    
    hrt = torch.zeros([*batch_like_shape, 4, 4], device=token.device)
    hrt[..., :3, :3] = rot_mat
    hrt[..., :3, 3] = trans3d
    hrt[..., 3, 3] = 1.
    
    return hrt 


def transform_points_old(hrt:torch.Tensor, points:torch.Tensor,):
    '''
    hrt: (B, 4, 4)
    points: (B, N, 3)
    
    return: (B, N, 3)
    '''
    if isinstance(hrt, np.ndarray):
        B, N, _ = points.shape
        p = np.concatenate([points, np.ones((B, N, 1))], axis=-1) # (B, N, 4)
        p_T = p.transpose(0, 2, 1) # (B, 4, N)
        p_transformed = hrt @ p_T # (B, 4, N)
        p_transformed = p_transformed.transpose(0, 2, 1)[:, :, :3] # (B, N, 3)
        return p_transformed

    elif isinstance(hrt, torch.Tensor):
    
        B, N, _ = points.shape
        p = torch.cat([points, torch.ones((B, N, 1), device=points.device)], dim=-1) # (B, N, 4)
        p_T = p.transpose(2, 1) # (B, 4, N)
        p_transformed = torch.bmm(hrt, p_T) # (B, 4, N)
        p_transformed = p_transformed.transpose(2, 1)[:, :, :3] # (B, N, 3)
        return p_transformed.contiguous()
    
    else:
        raise ValueError('Unsupported data type')
    

def transform_points(hrt:torch.Tensor, points:torch.Tensor, auto_expand=True):
    '''
    hrt: (..., 4, 4)
    points: (..., N, 3)
    
    return: (..., N, 3)
    '''
    
    batch_shape = hrt.shape[:-2]
    N = points.shape[-2]
    
    assert len(hrt.shape) == len(points.shape), "The hRT and points should have the same number of dimensions."
    assert hrt.shape[-2:] == (4, 4), "The hRT should have shape (..., 4, 4)."
    assert points.shape[-1] == 3, "The points should have shape (..., N, 3)."
    
    # auto expand the points batch dimension
    # if hrt.shape != points.shape and len(hrt.shape) == len(points.shape) and auto_expand:
    #     if sum(hrt.shape) > sum(points.shape):
    #         points = points.expand((*batch_shape, N, 3)).clone()
    #     else:
    #         hrt = hrt.expand((*batch_shape, 4, 4)).clone()
    
    '''ij, jk->ik'''
    pr = torch.einsum('...ij,...kj->...ki', hrt[..., :3, :3], points) # (..., N, 3)
    pr = pr + hrt[..., :3, 3].unsqueeze(-2) # (..., N, 3)
    return pr.contiguous()


def transform_points_33(r:torch.Tensor, points:torch.Tensor, auto_expand=True):
    '''
    hrt: (..., 4, 4)
    points: (..., N, 3)
    
    return: (..., N, 3)
    '''
    
    batch_shape = r.shape[:-2]
    N = points.shape[-2]
    
    # auto expand the points batch dimension
    if r.shape != points.shape and len(r.shape) == len(points.shape) and auto_expand:
        if sum(r.shape) > sum(points.shape):
            points = points.expand((*batch_shape, N, 3)).clone()
        else:
            r = r.expand((*batch_shape, 3, 3)).clone()
    
    '''ij, jk->ik'''
    pr = torch.einsum('...ij,...kj->...ki', r, points) # (..., N, 3)
    return pr.contiguous()


def rot_diff_rad(rot1, rot2):
    theta = torch.clamp((torch.trace(torch.matmul(rot1, rot2.transpose(1, 0))) - 1) / 2, min=-1.0, max=1.0)
    return torch.acos(theta) % (2 * np.pi)

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180

def batch_rot_diff(rot1, rot2, rad=False):
    batch_shape = rot1.shape[:-2]
    delta_rot = torch.einsum('...ij,...kj->...ik', rot1[..., :3, :3], rot2[..., :3, :3]) # include the transpose equal torch.bmm(rot1, rot2.transpose(2,1))
    cos_theta = torch.vmap(lambda x: torch.clamp((torch.trace(x) - 1.) / 2., min=-1.0, max=1.0))(delta_rot.reshape(-1, 3, 3)).reshape(*batch_shape, 1)
    theta_rad = torch.acos(cos_theta) % (2 * np.pi)
    
    if rad:
        return theta_rad
    else:
        return theta_rad / np.pi * 180

def batch_trans_diff(v1, v2):
    '''
    v1: (..., 3)
    v2: (..., 3)
    '''
    batch_shape = v1.shape[:-1]
    return torch.linalg.norm(v1 - v2, dim=-1).reshape(*batch_shape, 1)

def batch_pose_diff(ob_hrt:torch.Tensor, gt_hrt:torch.Tensor, rad=False):
    '''
    ob_hrt: (..., 4, 4)
    gt_hrt: (..., 4, 4)
    '''
    
    # auto expand the batch dimension
    if gt_hrt.shape != ob_hrt.shape and len(gt_hrt.shape) == len(ob_hrt.shape):
        gt_hrt = gt_hrt.expand_as(ob_hrt)
    
    rdiff = batch_rot_diff(ob_hrt, gt_hrt, rad)
    tdiff = batch_trans_diff(ob_hrt[..., :3, 3], gt_hrt[..., :3, 3])
    
    return rdiff, tdiff

def point_to_line_distance(point, line_loc, line_axis):
    """
    计算点到直线的距离(批量版本)
    
    Args:
        point: 点坐标 (B,1,3)
        line_loc: 直线上的点 (B,1,3) 
        line_axis: 直线方向向量 (B,1,3)
        
    Returns:
        distance: 点到直线的距离 (B,1)
    """
    # 计算点到直线上点的向量
    point_to_line = point - line_loc  # (B,1,3)
    
    # 计算叉积
    cross = torch.linalg.cross(point_to_line, line_axis, dim=-1)  # (B,1,3)
    
    # 计算叉积的范数
    cross_norm = torch.linalg.norm(cross, dim=-1)  # (B,1)
    
    # 计算axis的范数
    axis_norm = torch.linalg.norm(line_axis, dim=-1)  # (B,1)
    
    # 计算距离
    distance = cross_norm / (axis_norm + 1e-8)  # 添加epsilon避免除零
    
    return distance


def point_to_line_transform(point, line_loc, line_axis):
    """
    计算将点投影到直线上的变换矩阵
    
    Args:
        point: 点坐标 (...,1,3)
        line_loc: 直线上的点 (...,1,3) 
        line_axis: 直线方向向量 (...,1,3)
        
    Returns:
        transform: 变换矩阵 (...,4,4)
    """
    B = point.shape[:-2]
    
    # 归一化轴向量
    axis_normalized = line_axis / (torch.linalg.norm(line_axis, dim=-1, keepdim=True) + 1e-8)
    
    # 计算点到直线的向量
    point_to_line = point - line_loc  # (B,1,3)
    
    # 计算投影长度 (点在轴上的投影)
    projection_length = torch.sum(point_to_line * axis_normalized, dim=-1, keepdim=True)  # (B,1,1)
    
    # 计算投影点
    projected_point = line_loc + projection_length * axis_normalized  # (B,1,3)
    
    # 计算平移向量 (从原点到投影点的向量)
    translation = projected_point - point  # (...,1,3)
    
    # 构建变换矩阵
    transform = torch.eye(4, device=point.device).reshape(*[1]*len(B), 4, 4).repeat(*B, 1, 1)  # (..., 4, 4)
    transform[..., :3, 3] = translation.squeeze(-2)
    
    return transform


def pose_error(ob_hrt:torch.Tensor, gt_hrt:torch.Tensor,):
    # deprecated
    # hrt: (..., 4, 4)
    batch_shape = ob_hrt.shape[:-2] # (..., 4, 4)
    ob_hrt_re = ob_hrt.reshape(-1, 4, 4)
    gt_hrt_re = gt_hrt.reshape(-1, 4, 4)
    
    ob_rot_mat = ob_hrt_re[:, :3, :3] # (B, 3, 3)
    ob_trans_3d = ob_hrt_re[:, :3, 3] # (B, 3)
    gt_rot_mat = gt_hrt_re[:, :3, :3] # (B, 3, 3)
    gt_trans_3d = gt_hrt_re[:, :3, 3] # (B, 3)
    
    rdiff = batch_rot_diff(ob_rot_mat, gt_rot_mat)
    tdiff = torch.linalg.norm(ob_trans_3d - gt_trans_3d, dim=-1)
    
    return rdiff.reshape(*batch_shape), tdiff.reshape(*batch_shape)



def set_random_seed(seed):
    np.random.seed(seed)
    # torch.set_rng_state(torch.manual_seed(seed).get_state())
    torch.manual_seed(seed)
    random.seed(seed)


def uniform_sample_so3(batch_size, device):
    # 生成一批随机四元数
    # u1 = torch.rand(batch_size, device=device)
    # u2 = torch.rand(batch_size, device=device)
    # u3 = torch.rand(batch_size, device=device)
    
    u = torch.rand((batch_size, 3), device=device)
    u1 = u[:, 0]
    u2 = u[:, 1]
    u3 = u[:, 2]
    
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * torch.pi * u2) # 1
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * torch.pi * u2) # 0
    q3 = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3) # 0
    q4 = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3) # 0
    
    # 将四元数转换为旋转矩阵
    rotation_matrices = torch.zeros((batch_size, 3, 3), device=device)
    rotation_matrices[:, 0, 0] = 1 - 2 * (q3**2 + q4**2)
    rotation_matrices[:, 0, 1] = 2 * (q2 * q3 - q1 * q4)
    rotation_matrices[:, 0, 2] = 2 * (q2 * q4 + q1 * q3)
    rotation_matrices[:, 1, 0] = 2 * (q2 * q3 + q1 * q4)
    rotation_matrices[:, 1, 1] = 1 - 2 * (q2**2 + q4**2)
    rotation_matrices[:, 1, 2] = 2 * (q3 * q4 - q1 * q2)
    rotation_matrices[:, 2, 0] = 2 * (q2 * q4 - q1 * q3)
    rotation_matrices[:, 2, 1] = 2 * (q3 * q4 + q1 * q2)
    rotation_matrices[:, 2, 2] = 1 - 2 * (q2**2 + q3**2)
    
    return rotation_matrices

def mat_draw_plot(rotation_matrices):
    B = rotation_matrices.shape[0]
    point = torch.Tensor([[0., 0., 1.]], device=rotation_matrices.device).repeat(B, 1).unsqueeze(-1)
    tgt = torch.bmm(rotation_matrices, point)
    
    x = tgt[:,0,0].cpu().numpy()
    y = tgt[:,1,0].cpu().numpy()
    z = tgt[:,2,0].cpu().numpy()
    
    
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as mplot3d

    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(x, y, z)
    ax.view_init(elev=80, azim = 0)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    plt.show()

def batch_angle_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        v1: (..., 3)
        v2: (..., 3)
    
    Returns:
        angle (...,)
    """
    # 计算点积
    dot_product = torch.sum(v1 * v2, dim=-1)
    
    # 计算向量的范数
    norm_v1 = torch.norm(v1, dim=-1)
    norm_v2 = torch.norm(v2, dim=-1)
    
    norm = torch.clamp(norm_v1 * norm_v2, min=1e-8, )  # 避免除零
    # 计算夹角的余弦值
    cos_angle = dot_product / norm
    
    cos_angle = torch.clamp(cos_angle, min=-1.0, max=1.0)  # 限制在[-1, 1]范围内
    
    # 计算夹角（弧度）
    angle = torch.acos(cos_angle) / np.pi * 180
    
    return angle


def get_acc_per_threshold(err, start, stop, step):
    
    acc_per_theshold = []
    threshold = np.arange(start, stop, step)
    for i in threshold:
        acc = np.mean(err < i, axis=0)
        acc_per_theshold.append(acc)
    acc_per_theshold = np.array(acc_per_theshold)
    
    return acc_per_theshold, threshold


def is_nan(tensor:torch.Tensor):
    return torch.isnan(tensor).any()

def augment_part_transform(gt, part_id, gt_npcs2central_rt, batch_size, device, min=0, max=90, level2_info=False):

    
    gt_naocs_rt = gt['naocs2cam_rt'].view(batch_size, 4, 4).transpose(2, 1)
    gt_axis_offset = gt['unitvec_per_point'] * (1 - gt['heatmap_per_point'].reshape(batch_size, -1, 1)) * 0.2
    axis_point = gt['naocs_per_point'] + gt_axis_offset

    axis_mask = torch.where(gt['joint_cls_per_point'] == part_id, 1, torch.nan)
    # axis_mask = torch.where(gt['joint_cls_per_point'] == part_id, 1, 0)
    gt_axis_offset_part = axis_point * axis_mask[..., None]
    
    
    gt_axis_offset_noscale = gt_axis_offset_part * gt['naocs2cam_scale'][:, None, :].repeat(1, 1, 3)
    gt_axis_offset_cam = transform_points(gt_naocs_rt, gt_axis_offset_noscale)
    gt_axis_offset_central_part = transform_points(torch.inverse(gt_npcs2central_rt)[:, part_id, :, :], gt_axis_offset_cam)
    gt_axis_point_central_part = torch.nanmean(gt_axis_offset_central_part, dim=-2, keepdim=True)

    gt_axis_offset_central_base = transform_points(torch.inverse(gt_npcs2central_rt)[:, 0, :, :], gt_axis_offset_cam)
    gt_axis_point_central_base = torch.nanmean(gt_axis_offset_central_base, dim=-2, keepdim=True)

    angle = torch.rand((batch_size, 1), device=device) * (max - min) + min
    angle = angle / 180 * np.pi
    
    axis = gt['axis_per_point'] * axis_mask[..., None]
    axis = torch.nanmean(axis, dim=-2, keepdim=False)
    _t1 = torch.eye(4).to(device)[None, :, :].repeat(batch_size, 1, 1)
    _t1[:, :3, 3] = -gt_axis_point_central_part[:, 0, :]
    _t2 = torch.eye(4).to(device)[None, :, :].repeat(batch_size, 1, 1)
    _t2[:, :3, :3] = axis_angle_to_matrix(axis * angle) # (B, 3, 3)
    _t3 = torch.eye(4).to(device)[None, :, :].repeat(batch_size, 1, 1)
    _t3[:, :3, 3] = gt_axis_point_central_base[:, 0, :]
    __t = torch.bmm(torch.bmm(_t3, _t2), _t1)
        
    if is_nan(__t):
        # print('nan')
        nan_indices = torch.where(torch.isnan(axis).any(-1))[0]
        origin_t = torch.bmm(torch.inverse(gt_npcs2central_rt[:, 0, :, :]), gt_npcs2central_rt[:, part_id, :, :])
        __t[nan_indices] = origin_t[nan_indices]
        
        # if level2_info:
        #     _t2 = torch.bmm(torch.bmm(torch.inverse(_t3), __t), torch.inverse(_t1))
        #     axisangle = matrix_to_axis_angle(_t2[:, :3, :3])
        #     angle = torch.norm(axisangle, dim=-1, keepdim=True)
        #     axis = axisangle / angle
    
    # if is_nan(__t):
    #     print('nan')
    if level2_info:
        return __t, -gt_axis_point_central_part[:, 0, :], gt_axis_point_central_base[:, 0, :], axis, angle
    
    return __t


def go_to_device(data, device):
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
    
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device)
        return data
    
    elif isinstance(data, (list, tuple)):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].to(device)
        return data
    
    else:
        raise ValueError('Unsupported data type')
    
    
    
def is_new_better(old_dict:dict, new_dict:dict, key:str, smaller_better=True):
    if key in old_dict.keys():
        if old_dict[key].mean() > new_dict[key].mean():
            return True if smaller_better else False
        else:
            return False if smaller_better else True
    else:
        print(f"is_new_better: Key {key} not in the old_dict, return True.")
        return True


def save_visual_result(result, dir, filename='', subdir='visual'):
    
    print(f"Save visual result at {dir}/{subdir}")
    os.makedirs(os.path.join(dir, subdir), exist_ok=True)
    with open(os.path.join(dir, subdir, f'{filename}.pkl'), 'wb') as f:
        pickle.dump(result, f)   
        
        
def compose_rt(rotation, translation):
    '''
    rotation: (..., 3, 3)
    '''
    shape_batch = rotation.shape[:-2]
    RT = torch.zeros((*shape_batch, 4, 4), device=rotation.device)
    
    translation = translation.reshape(*shape_batch, 3)
    RT[..., :3, :3] = rotation
    RT[..., :3, 3] = translation
    RT[..., 3, 3] = 1
    return RT



def solve_pose_torch(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    assert A.size() == B.size()
    # get number of dimensions
    m = A.size()[1]
    # translate points to their centroids
    centroid_A = torch.mean(A, dim=0)
    centroid_B = torch.mean(B, dim=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = torch.mm(AA.transpose(1, 0), BB)
    U, S, Vt = torch.linalg.svd(H)
    # R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
    R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
    # special reflection case
    # print('S:', S)
    # print('detR', torch.det(R))
    
    if torch.det(R) < 0:
        Vt[m-1, :] *= -1
        R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
    
    # translation
    t = centroid_B - torch.mm(R, centroid_A.view(3, 1))[:, 0]
    T = torch.eye(4).to(A.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return  T


def batch_solve_pose_torch(A:torch.Tensor, B:torch.Tensor, reflection=False):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    assert A.size() == B.size()
    # get number of dimensions
    m = A.size()[-1]
    # translate points to their centroids
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = torch.einsum('...ji,...jk->...ik', AA, BB)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.einsum('...ji,...kj->...ik', Vt, U) # V @ Ut

    # special reflection case
    rflag = -1 if reflection else 1

    sign_detR = torch.sign(torch.det(R))
    Vt_new = Vt.clone()
    S_new = S.clone()
    Vt_new[..., m-1, :] *= sign_detR.unsqueeze(-1) * rflag
    S_new[..., m-1] *= sign_detR
    
    R = torch.einsum('...ji,...kj->...ik', Vt_new, U)   
    
    # translation
    
    t = centroid_B - torch.einsum('...ij,...kj->...ki', R, centroid_A) # (R @ centroid_A.T).T
    T = torch.zeros((*A.size()[:-2], 4, 4), device=A.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t[..., 0, :]
    T[..., 3, 3] = 1.
    return T

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1
    return  T



def base_dif_func(base_param, base_norm_kp, base_pred_kp):
    # param is np.array([quat, trans])
    # dif is l2 between pred_kp and norm_kp with transform
    base_r_quat, base_t = base_param[:4], base_param[4:]
    base_r_quat /= np.linalg.norm(base_r_quat)
    a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
    base_rot_matrix = np.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                    2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                    2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                    1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
    base_dis = np.linalg.norm((base_rot_matrix @ (base_norm_kp + base_t).T).T - base_pred_kp, axis=-1)
    return base_dis

def least_square_optim(init_rt, norm_kp, pred_kp):
    device = norm_kp.device
    qua = matrix_to_quaternion(init_rt[:3, :3])
    init_base_param = torch.cat([qua, init_rt[:3, 3]])
    init_base_param = init_base_param.detach().cpu().numpy()
    norm_kp = norm_kp.detach().cpu().numpy()
    pred_kp = pred_kp.detach().cpu().numpy()
    res_base = least_squares(base_dif_func, init_base_param, loss='soft_l1', args=(norm_kp, pred_kp))
    new_base_param = res_base.x
    print(res_base.success)
    new_base_param = torch.tensor(new_base_param, device=device)
    T = torch.eye(4, device=device)
    T[:3, :3] = quaternion_to_matrix(new_base_param[:4])
    T[:3, 3] = new_base_param[4:]
    return T

def gen_noise_transform(B, device, maxd, maxt):
    # raise ValueError('Bug function')
    # angle = torch.randn(B, 1).to(device)
    # angle = angle / 180. * np.pi * maxd
    # noise_rot = axis_angle_to_matrix(torch.randn(B, 3).to(device) * angle)
    # noise_trans = torch.randn(B, 3).to(device) * maxt
    # noise_transform = torch.eye(4).to(device)[None, :, :].repeat(B, 1, 1)
    # noise_transform[:, :3, :3] = noise_rot
    # noise_transform[:, :3, 3] = noise_trans
    # return noise_transform
    
    noise_transform = torch.eye(4).to(device)[None, :, :].repeat(B, 1, 1)
    
    if maxd > 0:
        angle = torch.empty((B, 1), device=device).uniform_(-maxd / 180. * np.pi, maxd / 180. * np.pi)
        axis = torch.empty((B, 3), device=device).uniform_(-1.0, 1.0)
        noise_rot = axis_angle_to_matrix(axis * angle)
        noise_transform[:, :3, :3] = noise_rot

    if maxt > 0:
        noise_trans = torch.empty((B, 3), device=device).uniform_(-maxt, maxt)
        noise_transform[:, :3, 3] = noise_trans

    return noise_transform



def gen_noise_transform_per_axis(B, device, maxx, maxy, maxz, maxt):
    # 随机生成绕 x, y, z 轴的旋转角度
    angles_x = torch.rand(B, 1).to(device) * maxx
    angles_y = torch.rand(B, 1).to(device) * maxy
    angles_z = torch.rand(B, 1).to(device) * maxz

    # 生成绕 x 轴的旋转矩阵
    rot_x = torch.eye(3).to(device).repeat(B, 1, 1)
    rot_x[:, 1, 1] = torch.cos(angles_x).squeeze()
    rot_x[:, 1, 2] = -torch.sin(angles_x).squeeze()
    rot_x[:, 2, 1] = torch.sin(angles_x).squeeze()
    rot_x[:, 2, 2] = torch.cos(angles_x).squeeze()

    # 生成绕 y 轴的旋转矩阵
    rot_y = torch.eye(3).to(device).repeat(B, 1, 1)
    rot_y[:, 0, 0] = torch.cos(angles_y).squeeze()
    rot_y[:, 0, 2] = torch.sin(angles_y).squeeze()
    rot_y[:, 2, 0] = -torch.sin(angles_y).squeeze()
    rot_y[:, 2, 2] = torch.cos(angles_y).squeeze()

    # 生成绕 z 轴的旋转矩阵
    rot_z = torch.eye(3).to(device).repeat(B, 1, 1)
    rot_z[:, 0, 0] = torch.cos(angles_z).squeeze()
    rot_z[:, 0, 1] = -torch.sin(angles_z).squeeze()
    rot_z[:, 1, 0] = torch.sin(angles_z).squeeze()
    rot_z[:, 1, 1] = torch.cos(angles_z).squeeze()

    noise_trans = torch.rand(B, 3).to(device) * maxt
    # 合成最终旋转矩阵 R = Rz * Ry * Rx
    rotation_matrix = torch.bmm(torch.bmm(rot_z, rot_y), rot_x)

    # 创建齐次变换矩阵
    transform_matrix = torch.eye(4).to(device).repeat(B, 1, 1)
    transform_matrix[:, :3, :3] = rotation_matrix
    transform_matrix[:, :3, 3] = noise_trans
    return transform_matrix





class Collector:
    def __init__(self, auto_convert=False):
        self.__data = {}
        self.__composed = False
        self.auto_convert = auto_convert
        
    def add(self, key, value):
        if value is not None:
            if key not in self.__data:
                self.__data[key] = []
            if self.auto_convert and isinstance(value, torch.Tensor):
                self.__data[key].append(value.detach().cpu().numpy())
            else:
                self.__data[key].append(value)
                
    def add_batch(self, data:dict):
        for k, v in data.items():
            self.add(k, v)
                
        
    def get(self, key):
        return self.__data[key]
    
    def get_all(self):
        return self.__data
    
    def clear(self):
        self.__data = {}
        
    def __str__(self):
        return str(self.__data)
    
    def __repr__(self):
        return str(self.__data)
    
    def __getitem__(self, key):
        return self.__data[key]
    
    def __setitem__(self, key, value):
        self.__data[key] = value
        
    def __len__(self):
        return len(self.__data)
    
    def __iter__(self):
        return iter(self.__data)
    
    def __contains__(self, key):
        return key in self.__data
    
    def __delitem__(self, key):
        del self.__data[key]
    
    def keys(self):
        return self.__data.keys()
    
    def values(self):
        return self.__data.values()
    
    def items(self):
        return self.__data.items()
    
    def update(self, other):
        self.__data.update(other)
        
    def save(self, path):
        dir = os.path.dirname(path)
        if len(dir) and not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, 'wb') as f:
            pickle.dump(self.__data, f)
            
    def save_slice(self, dir, base_name, step=1):
        assert self.is_composed(), "Please compose the data before saving."
        if len(dir) and not os.path.exists(dir):
            os.makedirs(dir)
        
        lendata = self.__data[list(self.__data.keys())[0]].shape[0]
            
        for idx in range(0, lendata, step):
            slice_name = os.path.join(dir, f"{base_name}_{idx:04d}.pkl")
            sliced = {}
            for k, v in self.__data.items():
                sliced[k] = v[idx]
            with open(slice_name, 'wb') as f:
                pickle.dump(sliced, f)            
            
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.__data = pickle.load(f)
            
    def saveh5(self, path):
        
        
        _tc = copy.deepcopy(self)
        _tc.numpy()
        if not _tc.is_composed():
            _tc.compose()
        # assert self.is_composed(), "Please compose the data before saving."
        
        with h5py.File(path, 'w', track_order=True) as f:
            default_key = list(_tc.__data.keys())[0]
            if isinstance(_tc[default_key], np.ndarray):
                for i in range(_tc[default_key].shape[0]):
                    g = f.require_group(str(i))
                    for key in _tc.keys():
                        try:
                            data = _tc[key][i]
                        except:
                            data = _tc[key]
                        g.create_dataset(key, data=data)
                        
        del _tc
       
    def saveh5_lite(self, path):
        
        
        
        assert self.is_composed(), "Please compose the data before saving."
        
        with h5py.File(path, 'w', track_order=True) as f:
            default_key = list(self.__data.keys())[0]
            if isinstance(self[default_key], np.ndarray):
                for i in range(self[default_key].shape[0]):
                    g = f.require_group(str(i))
                    for key in self.keys():
                        try:
                            data = self[key][i]
                        except:
                            data = self[key]
                        g.create_dataset(key, data=data)
       
                    
    def loadh5(self, h5file:h5py.File, groupid, addBtachDim=False):
        keys = h5file[groupid].keys()
        for key in keys:
            data = np.array(h5file[groupid][key])[None, ...] if addBtachDim else np.array(h5file[groupid][key])
            self.update({key: data})
            
            
    def loadh5_all(self, h5file:h5py.File, ):
        
        file_ids = natsort.natsorted(h5file)
        for id in file_ids:
            self.addh5(h5file, id, addBtachDim=True)
            
        self.compose()
            
    def addh5(self, h5file:h5py.File, groupid, addBtachDim=False):
        keys = h5file[groupid].keys()
        for key in keys:
            data = np.array(h5file[groupid][key])[None, ...] if addBtachDim else np.array(h5file[groupid][key])
            self.add(key, data)
                             
            
    def subframe(self, keys=[]):
        new_data = {}
        for key in keys:
            try:
                new_data[key] = self.__data[key]
            except:
                print(f"Warning: key {key} is missing, skip")
        
        c = Collector()
        c.update(new_data)
        return c
            
    def compose(self, ):
        '''
        if the ndim of data is 0, this function will stack the data along the 0 axis.
        if the ndim of data is greater than 0, this function will concatenate the data along the 0 axis.
        '''
        assert not self.__composed, "The data has been composed."
        try:
            for key in self.__data:
                if isinstance(self.__data[key], list) and len(self.__data[key]) > 0:
                    
                    
                    if isinstance(self.__data[key][0], torch.Tensor):
                        if self.__data[key][0].ndim == 0:
                            func = torch.stack
                        else:
                            func = torch.cat
                    elif isinstance(self.__data[key][0], np.ndarray):
                        if self.__data[key][0].ndim == 0:
                            func = np.stack
                        else:
                            func = np.concatenate
                    else:
                        raise ValueError(f"Unsupported data type: {type(self.__data[key][0])}")
                    
                    self.__data[key] = func(self.__data[key], 0)
                    
                    # if isinstance(self.__data[key][0], torch.Tensor):
                    #     if self.__data[key][0].ndim == 0:
                    #         self.__data[key] = torch.stack(self.__data[key], dim=0)
                    #     else:
                    #         self.__data[key] = torch.cat(self.__data[key], dim=0)
                    # elif isinstance(self.__data[key][0], np.ndarray):
                    #     self.__data[key] = np.concatenate(self.__data[key], axis=0)
                    
            self.__composed = True
            return self
        
        except Exception as e:
            traceback.print_exc()
            print(f"Error when composing {key}")
            raise e
            
    
    def is_composed(self, ):
        return self.__composed
    
    
    def slice(self, start=None, end=None, step=None):
        for key in self.__data:
            if isinstance(self.__data[key], (torch.Tensor, np.ndarray)):
                self.__data[key] = self.__data[key][start:end:step]

            elif isinstance(self.__data[key], dict):
                for k in self.__data[key]:
                    if isinstance(self.__data[key][k], (torch.Tensor, np.ndarray)):
                        self.__data[key][k] = self.__data[key][k][start:end:step]
                        
            elif isinstance(self.__data[key], list):
                for i in range(len(self.__data[key])):
                    if isinstance(self.__data[key][i], (torch.Tensor, np.ndarray)):
                        self.__data[key][i] = self.__data[key][i][start:end:step]

        return self
        
            
    def tensor(self, ):
        for key in self.__data:
            if isinstance(self.__data[key], np.ndarray):
                self.__data[key] = torch.tensor(self.__data[key])
            elif isinstance(self.__data[key], dict):
                for k in self.__data[key]:
                    if isinstance(self.__data[key][k], np.ndarray):
                        self.__data[key][k] = torch.tensor(self.__data[key][k])
            elif isinstance(self.__data[key], list):
                for i in range(len(self.__data[key])):
                    if isinstance(self.__data[key][i], np.ndarray):
                        self.__data[key][i] = torch.tensor(self.__data[key][i])
        return self
            
    def numpy(self, ):
        for key in self.__data:
            if isinstance(self.__data[key], torch.Tensor):
                self.__data[key] = self.__data[key].detach().cpu().numpy()
            elif isinstance(self.__data[key], dict):
                for k in self.__data[key]:
                    if isinstance(self.__data[key][k], torch.Tensor):
                        self.__data[key][k] = self.__data[key][k].detach().cpu().numpy()
            elif isinstance(self.__data[key], list):
                for i in range(len(self.__data[key])):
                    if isinstance(self.__data[key][i], torch.Tensor):
                        self.__data[key][i] = self.__data[key][i].detach().cpu().numpy()
        return self
            
    def to(self, device):
        for key in self.__data:
            if isinstance(self.__data[key], torch.Tensor):
                self.__data[key] = self.__data[key].to(device)
            elif isinstance(self.__data[key], dict):
                for k in self.__data[key]:
                    if isinstance(self.__data[key][k], torch.Tensor):
                        self.__data[key][k] = self.__data[key][k].to(device)
            elif isinstance(self.__data[key], list):
                for i in range(len(self.__data[key])):
                    if isinstance(self.__data[key][i], torch.Tensor):
                        self.__data[key][i] = self.__data[key][i].to(device)
        return self


class MetricCollector(Collector):
    
    def acc_single_th(self, key, th=5):
        return np.mean(self[key] < th, axis=0).flatten()
    
    def acc_double_th(self, key1, key2, th1=5, th2=0.05):
        return np.mean((self[key1] < th1) | (self[key2] < th2), axis=0).flatten()
    
    def acc_single_lambda(self, key, func):
        return np.mean(func(self[key]), axis=0).flatten()
    
    def acc_lambda(self, func, **kwargs):
        for k, v in kwargs.items():
            kwargs[k] = self[v]
        return np.mean(func(**kwargs), axis=0).flatten()
    
class H5Dataset(data.Dataset):
    def __init__(self, h5file):
        if isinstance(h5file, str):
            self.h5file = h5py.File(h5file, 'r')
            self._file_path = h5file
        elif isinstance(h5file, h5py.File):
            self.h5file = h5file
            self._file_path = self.h5file.filename
        else:
            raise ValueError('h5file must be a path or h5py.File')
        
        self.file_ids = natsort.natsorted(self.h5file)
        self.keys= self.h5file[self.file_ids[0]].keys()
        
        self._dataset_len = len(self.file_ids)
        
    def get_file_path(self):
        return self._file_path
        
    def setKeys(self, *keys):
        self.keys = keys
        
        
    def _set_dataset_len(self, length:int):
        self._dataset_len = length
        
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        id = self.file_ids[idx]
        
        data = []
        for key in self.keys:
            ele = torch.from_numpy(np.array(self.h5file[id][key]))
            if ele.dim() == 0:
                ele = ele.unsqueeze(0)
            data.append(
                ele
            )
        return tuple(data)
    
    
    
class H5Dataset_multi(data.Dataset):
    def __init__(self, *h5file, allow_missing_keys=False, return_dict=False):
        
        self.allow_missing_keys = allow_missing_keys
        self.return_dict = return_dict
        self._file_path = []

        if not isinstance(h5file, (list, tuple)):
            h5file = [h5file]
        
        if isinstance(h5file, (list, tuple)):
            self.h5file = []
            for f in h5file:
                if isinstance(f, str):
                    self.h5file.append(h5py.File(f, 'r'))
                    self._file_path.append(f)
                elif isinstance(f, h5py.File):
                    self.h5file.append(f)
                    self._file_path.append(f.filename)
                else:
                    raise ValueError('h5file must be a path or h5py.File')
        else:
            raise ValueError('h5file must be a list')
        
        self.file_ids = []
        
        for h5 in self.h5file:
            self.file_ids.append(natsort.natsorted(h5))
            
        for i in range(len(self.file_ids)-1):
            if self.file_ids[i] != self.file_ids[i+1]:
                raise ValueError('All h5 files must have the same file ids')
            
        self.file_ids = self.file_ids[0]  # use the first file_ids as the reference
        
        
        self.keys = [list(h5[self.file_ids[0]].keys()) for h5 in self.h5file]


        # build a key-index map
        self.key_map = {}
        for i, keys in enumerate(self.keys):
            keys = [str(k) for k in keys]
            for key in keys:
                self.key_map[key] = i
                
        self._dataset_len = len(self.file_ids)

    def setKeys(self, *keys):
        assert isinstance(keys, (list, tuple)), "keys must be a list or tuple"
        for key in keys:
            assert (key in self.key_map) or self.allow_missing_keys, f"Unknown key: {key}"

        self.keys = keys
        
    def get_file_path(self):
        return self._file_path
        
    def _set_dataset_len(self, length:int):
        self._dataset_len = length
        
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        id = self.file_ids[idx]
        
        data = {} if self.return_dict else []
        for key in self.keys:
            
            if key not in self.key_map:
                if not self.allow_missing_keys:
                    raise KeyError(f"Unknown key: {key}, allow_missing_keys={self.allow_missing_keys}")
                
                if self.return_dict:
                    continue
                else:
                    data.append(torch.tensor(np.nan))  # or some default value
                 
            else:
                file_idx = self.key_map[key]
                ele = torch.from_numpy(np.array(self.h5file[file_idx][id][key]))
                if ele.dim() == 0:
                    ele = ele.unsqueeze(0)

                if self.return_dict:
                    data[key] = ele
                else:
                    data.append(ele)
        return data
    
    

def markdown_table(data, header=None):
    if header is not None:
        data.insert(0, header)
    md = tabulate.tabulate(data, tablefmt='github', headers='firstrow', numalign='center', stralign='center', )
    lines = md.split('\n')
    # 手动修改第二行，添加用于居中对齐的冒号
    align_line = lines[1]
    alignments = align_line.strip().split('|')
    new_alignments = []
    for align in alignments:
        if align:
            new_alignments.append(':---:')
    lines[1] = '|' + '|'.join(new_alignments) + '|'
    # 合并修改后的行
    md = '\n'.join(lines)
    
    md = md.strip('[').strip(']')
    
    return md



def part_augmentation(cloud, gt_part_rt, part_id, gt_part_cls, norm_joint_loc, norm_joint_axis, joint_state, min_angle=-90, max_angle=90, releative=False, joint_type='revolute'):
    '''
        cloud (B N 3)
        norm_joint_axis (B P-1 3)
        norm_joint_loc (B P-1 3)
    '''
    B = cloud.size(0)
    assert part_id > 0
    
    aug_matrix = torch.eye(4, device=cloud.device).unsqueeze(0).repeat(B, 1, 1) # B 4 4
    
    if joint_type == 'revolute':
    
        angle = torch.rand((B, 1), device=cloud.device) * (max_angle - min_angle) + min_angle
        # FIXME
        # angle = torch.ones((B, 1), device=cloud.device) * (max_angle - min_angle) + min_angle
        
        # angle = angle * np.pi / 180.
        aug_matrix[:, :3, :3] = axis_angle_to_matrix( - norm_joint_axis[:, part_id-1, :] * angle)
    
    elif joint_type == 'prismatic':
        # NOTE:
        angle = torch.rand((B, 1), device=cloud.device) * (max_angle - min_angle) + min_angle
        t = - norm_joint_axis[:, part_id-1, :] * angle
        aug_matrix[:, :3, 3] = t
        
    else:
        raise ValueError('Unsupported joint type')

    
    tmat1 = torch.eye(4, device=cloud.device).unsqueeze(0).repeat(B, 1, 1)
    tmat1[:, :3, 3] = -norm_joint_loc[:, part_id-1, :]
    
    tmat2 = torch.eye(4, device=cloud.device).unsqueeze(0).repeat(B, 1, 1)
    tmat2[:, :3, 3] = norm_joint_loc[:, part_id-1, :]
    
    # mat = torch.matmul(tmat2, torch.matmul(aug_matrix, tmat1))
    if releative:
        mat = gt_part_rt[:, part_id] @ tmat2 @ aug_matrix @ tmat1 @ torch.linalg.inv(gt_part_rt[:, part_id])
    else:
        mat = gt_part_rt[:, 0] @ tmat2 @ aug_matrix @ tmat1 @ torch.linalg.inv(gt_part_rt[:, part_id])
    aug_part_cloud = transform_points(mat, cloud)
    
    cloud = torch.where(gt_part_cls.unsqueeze(-1).repeat(1, 1, 3) == part_id, aug_part_cloud, cloud)

    gt_part_rt[:, part_id] = mat @ gt_part_rt[:, part_id] #@ torch.linalg.inv(mat)
    
    joint_state[:, part_id:part_id+1] = angle
    
    return cloud, gt_part_rt, joint_state
    


    
def base_augmentation(cloud, gt_part_rt, maxd=5., maxt=0.1):
    
    B, P, _, _ = gt_part_rt.size()
    nosie_r = gen_noise_transform(B, cloud.device, maxd=maxd, maxt=0) # B 4 4
    nosie_t = gen_noise_transform(B, cloud.device, maxd=0, maxt=maxt) # B 4 4
    
    part_releative_pose = torch.einsum('...ij,...jk->...ik', torch.linalg.inv(gt_part_rt[:, 0:1]), gt_part_rt[:, 1:])
    
    # nosie = torch.linalg.inv(nosie)
    # cloud = transform_points(nosie, cloud)
    aug_base = torch.einsum('...ij,...jk->...ik', gt_part_rt[:, 0:1], nosie_r.unsqueeze(1))
    aug_base = torch.einsum('...ij,...jk->...ik', nosie_t.unsqueeze(1), aug_base)
    aug_part = torch.einsum('...ij,...jk->...ik', aug_base, part_releative_pose)
    gt_part_rt_N = torch.cat([aug_base, aug_part], dim=1)
    
    # gt_part_rt_N = torch.einsum('...ij,...jk->...ik', nosie_r, torch.linalg.inv(gt_part_rt[:, 0]))
    # gt_part_rt_N = torch.einsum('...ij,...jk->...ik', gt_part_rt, gt_part_rt_N.unsqueeze(1))
    pt = gt_part_rt_N[:, 0] @ torch.linalg.inv(gt_part_rt[:, 0])
    cloud = transform_points(pt, cloud)
    return cloud, gt_part_rt_N


def base_augmentation_per_axis(cloud, gt_part_rt, maxx=0., maxy=0., maxz=0., maxt=0.1):
    
    B, P, _, _ = gt_part_rt.size()
    nosie_r = gen_noise_transform_per_axis(B, cloud.device, maxx=maxx, maxy=maxy, maxz=maxz, maxt=0) # B 4 4
    nosie_t = gen_noise_transform_per_axis(B, cloud.device, maxx=0., maxy=0., maxz=0., maxt=maxt) # B 4 4
    
    part_releative_pose = torch.einsum('...ij,...jk->...ik', torch.linalg.inv(gt_part_rt[:, 0:1]), gt_part_rt[:, 1:])
    
    # nosie = torch.linalg.inv(nosie)
    # cloud = transform_points(nosie, cloud)
    aug_base = torch.einsum('...ij,...jk->...ik', gt_part_rt[:, 0:1], nosie_r.unsqueeze(1))
    aug_base = torch.einsum('...ij,...jk->...ik', nosie_t.unsqueeze(1), aug_base)
    aug_part = torch.einsum('...ij,...jk->...ik', aug_base, part_releative_pose)
    gt_part_rt_N = torch.cat([aug_base, aug_part], dim=1)
    
    # gt_part_rt_N = torch.einsum('...ij,...jk->...ik', nosie_r, torch.linalg.inv(gt_part_rt[:, 0]))
    # gt_part_rt_N = torch.einsum('...ij,...jk->...ik', gt_part_rt, gt_part_rt_N.unsqueeze(1))
    pt = gt_part_rt_N[:, 0] @ torch.linalg.inv(gt_part_rt[:, 0])
    cloud = transform_points(pt, cloud)
    return cloud, gt_part_rt_N





def get_kp_visibility(gt_rt, gt_norm_part_kp, cloud, gt_part_cls, num_parts, thresd=0.1, thresn=3):
    
    projected_kp = transform_points(gt_rt, gt_norm_part_kp) # B, KP, 3
    
    dis = torch.linalg.norm(projected_kp.unsqueeze(1) - cloud.unsqueeze(2).unsqueeze(2), dim=-1)
    
    mask_1 = dis < thresd
    mask_2 = F.one_hot(gt_part_cls.long(), num_classes=num_parts).unsqueeze(-1)
    mask = mask_1 * mask_2
    mask = mask.sum(dim=1) > thresn

    return mask.float(), projected_kp


def size_argumentation(cloud, gt_rt, gt_part_cls, 
                       gt_norm_joint_loc, gt_norm_joint_axis,
                       gt_norm_part_kp=None,
                       scale_range=0.1,
                       ratio_lock='all',
                       return_transform=False,
                       ):
    
    # transform cloud in to canonical space
    B, P, _, _ = gt_rt.size()
    seg = gt_part_cls.clone().unsqueeze(-1).repeat(1, 1, 3)
        
    if ratio_lock == 'all':
        scale_ratio = 1 + (torch.rand(B, 1, 1, device=cloud.device) * scale_range * 2 - scale_range)
        scale_ratio = scale_ratio.repeat(1, 1, 3)
    elif ratio_lock == 'yz':
        scale_ratio = 1 + (torch.rand(B, 1, 2, device=cloud.device) * scale_range * 2 - scale_range)
        scale_ratio = torch.cat([scale_ratio, scale_ratio[..., 1:2]], dim=-1)
        
    elif ratio_lock == 'none':
        scale_ratio = 1 + (torch.rand(B, 1, 3, device=cloud.device) * scale_range * 2 - scale_range)
        
    else:
        raise ValueError('Unsupported ratio lock')
    
    scale_transform = torch.eye(4, device=cloud.device).unsqueeze(0).unsqueeze(0).expand_as(gt_rt).clone()
    scale_transform[..., :3, :3] = torch.diag_embed(scale_ratio, dim1=-2, dim2=-1)
    
    # compose transform
    composed_rt = gt_rt @ scale_transform @ torch.linalg.inv(gt_rt)
    
    # transform cloud
    for part_id in range(P):
        cloud = torch.where(seg == part_id, transform_points(composed_rt[:, part_id], cloud), cloud)

    # scale data in canonical space
    # scaled_cloud = cloud #* scale_ratio
    scaled_gt_norm_joint_loc = gt_norm_joint_loc * scale_ratio
    scaled_gt_norm_joint_axis = gt_norm_joint_axis * scale_ratio
    
    if gt_norm_part_kp is not None:
        scaled_gt_norm_part_kp = gt_norm_part_kp * scale_ratio.unsqueeze(-2)
    else:
        scaled_gt_norm_part_kp = None
    
    if return_transform:
        return cloud, scaled_gt_norm_joint_loc, scaled_gt_norm_joint_axis, scaled_gt_norm_part_kp, scale_transform
    else:
        return cloud, scaled_gt_norm_joint_loc, scaled_gt_norm_joint_axis, scaled_gt_norm_part_kp
    

def size_argumentation_2(
                         scale_ratio:torch.Tensor | None,
                         parent_rt:torch.Tensor | None,
                         part_rt:torch.Tensor, 
                         part_norm_joint_loc:torch.Tensor | None, 
                         scale_range:float=0.1,
                         ratio_lock='all',
                         return_scale_ratio=False,
                       ):
    '''
    
    Size augmentation for point cloud and ground truth data.
    scale center is the parent transformation matrix.
    
    Args:
        scale_ratio: (B, 3) or None scale transformation matrix. if None, it will be generated
        parent_rt: (B, 4, 4) or None parent part transformation matrix
        part_rt: (B, 4, 4) part transformation matrix
        part_norm_joint_loc: (B, 1, 3) joint pivot locations in canonical space
        scale_range: (float) scale range for augmentation
        ratio_lock: (str) 'all', 'yz', or 'none' to control
            the scaling ratio for each axis
        return_scale_ratio: (bool) whether to return the scaling ratio
    Returns:
        cloud: (B, N, 3) transformed point cloud
        scaled_gt_norm_joint_loc: (B, P-1, 3) scaled joint locations
        scaled_gt_norm_joint_axis: (B, P-1, 3) scaled joint axes
        scaled_gt_norm_part_kp: (B, KP, 3) scaled keypoints
        scale_ratio: (B, 3) transformation matrix if return_transform is True
    
    
    
    '''
    B = part_rt.size(0)
    if parent_rt is None:
        # generate parent transformation matrix
        parent_rt = part_rt.clone()
    
    if scale_ratio is None:
        # generate scale transformation matrix
        
        
            
        if ratio_lock == 'all':
            scale_ratio = 1 + (torch.rand(B, 1, device=parent_rt.device) * scale_range * 2 - scale_range)
            scale_ratio = scale_ratio.repeat(1, 3)
        elif ratio_lock == 'yz':
            scale_ratio = 1 + (torch.rand(B, 2, device=parent_rt.device) * scale_range * 2 - scale_range)
            scale_ratio = torch.cat([scale_ratio, scale_ratio[..., 1:2]], dim=-1)
            
        elif ratio_lock == 'none':
            scale_ratio = 1 + (torch.rand(B, 3, device=parent_rt.device) * scale_range * 2 - scale_range) # (B, 3)
            
        else:
            raise ValueError('Unsupported ratio lock')
        
        # scale_transform = torch.eye(4, device=parent_rt.device).unsqueeze(0).expand_as(parent_rt).clone()
        # scale_transform[..., :3, :3] = torch.diag_embed(scale_ratio, dim1=-2, dim2=-1)
        
        
        
    # scale part transformation matrix
    # scale only apply to the translation part
    
    # scaled_part_rt = parent_rt @ part_rt @ scale_transform
    
    _T = torch.linalg.inv(parent_rt) @ part_rt
    _T[..., :3, 3] *= scale_ratio
    scaled_part_rt = parent_rt @ _T
    
    if part_norm_joint_loc is not None:
        scaled_part_norm_joint_loc = part_norm_joint_loc * scale_ratio
    else:
        scaled_part_norm_joint_loc = torch.zeros(B, 1, 3, device=part_rt.device)  # (B, 1, 3) if no joint loc is provided
    

    # compose transform
    
    if return_scale_ratio:
        return scaled_part_rt, scaled_part_norm_joint_loc, scale_ratio
    else:
        return scaled_part_rt, scaled_part_norm_joint_loc,
    

def transform_points_by_segmentation(transform, points, seg):
    '''
    Args:
        transform: (B, P, 4, 4) transformation matrix
        points: (B, N, 3) point cloud
        seg: (B, N, 1) segmentation mask, which contains P unique part ids for each point
    Returns:
        transformed_points: (B, N, 3) transformed point cloud
    '''

    tc = torch.zeros_like(points, device=points.device)
    for part_id in range(transform.size(1)):
        tc = torch.where(seg == part_id,
                         transform_points(transform[:, part_id], points),
                         tc)
    return tc



def index_points(points, idx):
    """
    from riconv2
    
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points


# def compute_LRA(xyz, weighting=False, nsample=64, need_mask=False, vail_rate=10.):
#     '''
#     from riconv2
#     '''
#     dists = torch.cdist(xyz, xyz)

#     dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
#     dists = dists.unsqueeze(-1)

#     group_xyz = index_points(xyz, idx)
#     group_xyz = group_xyz - xyz.unsqueeze(2)

#     if weighting:
#         dists_max, _ = dists.max(dim=2, keepdim=True)
#         dists = dists_max - dists
#         dists_sum = dists.sum(dim=2, keepdim=True)
#         weights = dists / dists_sum
#         weights[weights != weights] = 1.0
#         M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
#     else:
#         M = torch.matmul(group_xyz.transpose(3,2), group_xyz) # B, N, S, 3

#     # eigen_values, vec = M.symeig(eigenvectors=True)

#     eigen_values, vec = torch.linalg.eigh(M, UPLO='U')


#     LRA = vec[:,:,:,0]
#     LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
#     LRA = LRA / LRA_length
    
#     if need_mask:
#         surface_mask = (vail_rate * eigen_values[..., 0] < eigen_values[..., 1]) & (vail_rate * eigen_values[..., 0] < eigen_values[..., 2])
#         return LRA, surface_mask
    
#     else:
#         return LRA # B N 3


def compute_LRA(xyz, weighting=False, nsample=64, need_mask=False, vail_rate=10., shared_centroid=None):
    raise NotImplementedError("This function has been updated")
    knn_result = knn_points(xyz, xyz, K=nsample)
    idx = knn_result.idx
    
    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)
    

    if weighting:
        dists = torch.norm(group_xyz, dim=-1, keepdim=True)  
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights = torch.nan_to_num(weights, nan=1.0)  
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    
    
    LRA = vec[:,:,:,0]
    LRA = LRA / torch.norm(LRA, dim=-1, keepdim=True)  

    if shared_centroid is None:
        centroid = torch.mean(xyz, dim=1, keepdim=True)
    else:
        centroid = shared_centroid   
    direction = xyz - centroid                          
    dot_product = torch.sum(LRA * direction, dim=-1, keepdim=True)
    LRA = torch.where(dot_product < 0, -LRA, LRA)       

    if need_mask:
        surface_mask = (vail_rate * eigen_values[...,0] < eigen_values[...,1]) & \
                       (vail_rate * eigen_values[...,0] < eigen_values[...,2])
        return LRA, surface_mask.unsqueeze(-1)
    
    return LRA


def compute_LRA_all(xyz:torch.Tensor, weighting=False, shared_centroid=None):

    centroid = xyz.mean(dim=-2, keepdim=True)
    group_xyz = xyz - centroid
    
    if weighting:
        dists = torch.linalg.norm(group_xyz, dim=-1, keepdim=True)  
        dists_max, _ = dists.max(dim=-2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=-2, keepdim=True)
        weights = dists / dists_sum
        weights = torch.nan_to_num(weights, nan=1.0)  
        M = torch.einsum('...ij,...ik->...jk', group_xyz, weights*group_xyz)
    else:
        M = torch.einsum('...ij,...ik->...jk', group_xyz, group_xyz)

    
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    
    
    LRA = vec[..., 0]
    LRA = LRA / torch.linalg.norm(LRA, dim=-1, keepdim=True)  

    if shared_centroid is not None:
        
        direction = (centroid - shared_centroid).squeeze(-2)                    
        dot_product = torch.sum(LRA * direction, dim=-1, keepdim=True)
        LRA = torch.where(dot_product < 0, -LRA, LRA)       

    
    return LRA




def compute_LRA_PP(xyz, weighting=False, nsample=64, vail_rate=10.):
    B, S, P, N, _ = xyz.shape
    pbatch_xyz = xyz.reshape(-1, P, N, 3)
    centroid = torch.mean(xyz.reshape(-1, P*N, 3), dim=-2, keepdim=True)
    all_lra = []
    all_mask = []
    for part_id in range(P):
        lra, mask = compute_LRA(pbatch_xyz[:, part_id], weighting=weighting, nsample=nsample, need_mask=True, vail_rate=vail_rate, shared_centroid=centroid)
        all_lra.append(lra)
        all_mask.append(mask)
        
    all_lra = torch.stack(all_lra, dim=-3)
    all_mask = torch.stack(all_mask, dim=-3)
    
    all_lra = all_lra.reshape(B, S, P, N, 3)
    all_mask = all_mask.reshape(B, S, P, N, 1)
    return all_lra, all_mask


def fps(data:torch.Tensor, number, needidx=False):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() # B number 3
    if needidx:
        return fps_data, fps_idx
    else:
        return fps_data
    
    
def fps_batch(data:torch.Tensor, number:int, needidx=False):
    '''
        data ... N 3
        number int
    '''
    
    _batch = data.shape[:-2]
    pdata = data.reshape(-1, data.shape[-2], data.shape[-1]) # B N 3

    fps_idx = pointnet2_utils.furthest_point_sample(pdata, number) 
    fps_data = pointnet2_utils.gather_operation(pdata.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous() # B number 3
    
    fps_data = fps_data.reshape(*_batch, number, 3)
    fps_idx = fps_idx.reshape(*_batch, number)
    if needidx:
        return fps_data, fps_idx
    else:
        return fps_data
    
    
def knn_batch_upsample(batch_points, up_ratio=4, k=3):
    """
    kNN-based point cloud upsampling.
    Args:
        batch_points (Tensor): batched point cloud [B, N, 3]
        up_ratio (int): upsample ratio
        k (int): number of neighbors
    Returns:
        Tensor: upsampled point cloud [B, N*up_ratio, 3]
    """
    B, N, C = batch_points.shape
    device = batch_points.device

    # 计算点间距离矩阵 [B, N, N]
    dist_matrix = torch.cdist(batch_points, batch_points, p=2)

    # 获取k+1个最近邻（包含自身）
    _, knn_indices = torch.topk(dist_matrix, k=k+1, dim=-1, largest=False)
    
    # 排除自身索引 [B, N, k]
    neighbor_indices = knn_indices[..., 1:]

    # 收集邻居点坐标 [B, N, k, 3]
    neighbors = torch.gather(
        batch_points.unsqueeze(2).expand(-1, -1, k, -1),
        dim=1,
        index=neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, C)
    )

    # 生成插值权重 [B, N, up_ratio, k, 1]
    weights = torch.rand(B, N, up_ratio, k, 1, device=device)
    weights = weights / (weights.sum(dim=-2, keepdim=True) + 1e-8)  # 归一化

    # 生成新点 [B, N, up_ratio, 3]
    new_points = (neighbors.unsqueeze(2) * weights).sum(dim=-2)

    # 调整形状并打乱顺序
    upsampled = new_points.reshape(B, N * up_ratio, 3)
    shuffled_idx = torch.randperm(N * up_ratio, device=device)
    return upsampled[:, shuffled_idx, :]



def fnp(arr:np.ndarray, precision=4):
    if arr.size == 0:
        return ""
    # 展平数组并格式化每个元素
    flat = arr.ravel()
    elements = [f"{x:.{precision}g}" for x in flat]
    # 计算最大宽度以确保对齐
    max_width = max(len(e) for e in elements) if elements else 0
    # 生成固定宽度右对齐的字符串
    formatted = [e.rjust(max_width) for e in elements]
    # 重塑为原数组形状
    shaped = np.array(formatted, dtype=object).reshape(arr.shape)
    # 递归组合子数组的字符串
    def recurse(sub_arr:np.ndarray) -> str:
        if sub_arr.ndim == 0:
            return sub_arr.item()
        elif sub_arr.ndim == 1:
            return ', '.join(sub_arr)
        else:
            return '\n'.join(recurse(sub) for sub in sub_arr)
    return recurse(shaped)


def get_colored_cloud(cloud:torch.Tensor, 
                      part_cls:torch.Tensor, 
                      color:torch.Tensor) -> torch.Tensor:
    color_batched = color.unsqueeze(0).repeat(cloud.shape[0], 1, 1)
    if part_cls.dim() == 2:
        cls = part_cls.unsqueeze(-1)
    else:
        cls = part_cls
    c = torch.gather(color_batched, dim=1, index=cls.expand_as(cloud))
    return torch.cat([cloud, c], dim=-1)


def get_colored_cloud_numpy(cloud:np.ndarray,
                            part_cls:np.ndarray,
                            color:np.ndarray) -> np.ndarray:
    """
    Numpy version of get_colored_cloud
    Args:
        cloud: (B, N, 3)
        part_cls: (B, N) or (B, N, 1)
        color: (C, 3) - Color palette
    Returns:
        (B, N, 6)
    """
    # Handle dimensions: if (B, N, 1), squeeze to (B, N)
    if part_cls.ndim == 3:
        part_cls = part_cls.squeeze(-1)
    
    # Ensure integer type for indexing
    part_cls = part_cls.astype(int)
    
    # Advanced indexing to map class indices to colors
    # color[part_cls] will have shape (B, N, 3)
    c = color[part_cls]
    
    # Concatenate cloud and color
    return np.concatenate([cloud, c], axis=-1)



def select_by_id(id, data, shared_dim=2, idx_dim=1):
    '''
    id: B, N, ...
    data: B, S, ...
    '''
    
    data_identi_shape = data.shape[shared_dim:]
    id_identi_shape = id.shape[shared_dim:]
    det_dim = len(data_identi_shape) - len(id_identi_shape)
    
    # make sure the shape[shared_dim:] of id is all 1
    if len(id_identi_shape) != sum(id_identi_shape):
        raise ValueError('id shape error')
    
    if det_dim < 0:
        raise ValueError('id dim error')
    
    # extern the id shape to data shape
    id_ext = id.reshape(*id.shape, *([1]*det_dim))
    selected_data = torch.gather(data, idx_dim, id_ext.repeat(*([1]*shared_dim), *data_identi_shape))
    return selected_data, id


def normlize(x, dim):
        
    min_vals, _ = torch.min(x, dim=dim, keepdim=True)
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    
    epsilon = 1e-8
    denominator = max_vals - min_vals
    denominator = torch.where(denominator > epsilon, denominator, torch.ones_like(denominator) * epsilon)
    
    x_normalized = (x - min_vals) / denominator
    
    return x_normalized


def shuffle_sample(sample, sdf):
    
    B, P, N = sample.shape[:3]
    device = sample.device
    
    rand_matrix = torch.rand(B, P, N, device=device)
    idx = rand_matrix.argsort(dim=-1)  
    
    sdf_shuffled = torch.gather(sdf, dim=2, index=idx)
    
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    sample_shuffled = torch.gather(sample, dim=2, index=idx_expanded)
    
    return sample_shuffled, sdf_shuffled



def batch_linspace(start, end, steps):
    """
    批处理版本的 torch.linspace
    
    Args:
        start: (...) - 起始值张量
        end: (...) - 结束值张量
        steps: int - 采样点数量
    
    Returns:
        (..., steps) - 线性空间结果
    """
    # 确保 start 和 end 形状相同
    assert start.shape == end.shape, "start 和 end 形状必须相同"
    
    # 为步长维度扩展
    start_expanded = start.unsqueeze(-1)  # (..., 1)
    end_expanded = end.unsqueeze(-1)      # (..., 1)
    
    # 创建系数张量 [0, 1]
    device = start.device
    step_coefs = torch.linspace(0, 1, steps, device=device)  # (steps,)
    
    # 使用广播计算线性插值
    return start_expanded + (end_expanded - start_expanded) * step_coefs  # (..., steps)

def batch_meshgrid_3d(x, y, z):
    """
    batched meshgrid (indexing='ij')
    
    Args:
        x: (..., N) - x coord
        y: (..., M) - y coord
        z: (..., L) - z coord
        
    Returns:
        X: (..., N, M, L) - x grid
        Y: (..., N, M, L) - y grid
        Z: (..., N, M, L) - z grid
    """
    B = x.shape[:-1]
    N, M, L = x.size(-1), y.size(-1), z.size(-1)
    
    
    x_view = x.view(*B, N, 1, 1)
    y_view = y.view(*B, 1, M, 1)
    z_view = z.view(*B, 1, 1, L)
    

    X = x_view.expand(*B, N, M, L)
    Y = y_view.expand(*B, N, M, L)
    Z = z_view.expand(*B, N, M, L)
    
    return X, Y, Z

def pts_inside_box(pts, bbox):
    """
    Args:
        pts: (..., N, 3)
        bbox: (..., 8, 3)
        
        order1: (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
               Z
               |
            3-----0
           /|    /|
          2-----1 |
          | |   | |
          | 7---|-4 
          |/    |/  ---> Y
          6-----5
            /
           X 
        order2: (-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)
               Z
               |
            0-----1
           /|    /|
          3-----2 |
          | |   | |
          | 4---|-5 
          |/    |/  ---> Y
          7-----6
            /
           X 
    Returns:
        (..., ) 
    """
    pts_shape = pts.shape
    bbox_shape = bbox.shape
    
    assert pts_shape[-1] == 3 and bbox_shape[-2:] == (8, 3)
    
    # order 1
    # bbox_origin = bbox[..., 4, :] # (..., 3)
    
    # u1 = bbox[..., 5, :] - bbox_origin  # (..., 3)
    # u2 = bbox[..., 7, :] - bbox_origin  # (..., 3)
    # u3 = bbox[..., 0, :] - bbox_origin  # (..., 3)
    
    # order 2
    bbox_origin = bbox[..., 5, :] # (..., 3)
    
    u1 = bbox[..., 6, :] - bbox_origin  # (..., 3)
    u2 = bbox[..., 4, :] - bbox_origin  # (..., 3)
    u3 = bbox[..., 1, :] - bbox_origin  # (..., 3)
    
    
    up = pts - bbox_origin.unsqueeze(-2)  # (..., N, 3)

    
    u1 = u1.unsqueeze(-1)  # (..., 3, 1)
    u2 = u2.unsqueeze(-1)  # (..., 3, 1)
    u3 = u3.unsqueeze(-1)  # (..., 3, 1)
        
    p1 = torch.matmul(up, u1)  # (..., 1)
    p2 = torch.matmul(up, u2)  # (..., 1)
    p3 = torch.matmul(up, u3)  # (..., 1)
    
    u1_dot = torch.sum(u1 * u1, dim=-2, keepdim=True)  # (..., 1, 1)
    u2_dot = torch.sum(u2 * u2, dim=-2, keepdim=True)  # (..., 1, 1)
    u3_dot = torch.sum(u3 * u3, dim=-2, keepdim=True)  # (..., 1, 1)
    
    p1 = torch.logical_and(p1 > 0, p1 < u1_dot)  # (..., N, 1, 1)
    p2 = torch.logical_and(p2 > 0, p2 < u2_dot)  # (..., N, 1, 1)
    p3 = torch.logical_and(p3 > 0, p3 < u3_dot)  # (..., N, 1, 1)
    
    result = torch.logical_and(torch.logical_and(p1, p2), p3)  # (..., N, 1, 1)
    return result.squeeze(-1).squeeze(-1)  # (..., N)

def batch_iou_3d(bbox1, bbox2, nres=50):
    B = bbox1.shape[:-2]
    
    bmin = torch.amin(torch.cat([bbox1, bbox2], dim=-2), dim=-2)  # (B, 3)
    bmax = torch.amax(torch.cat([bbox1, bbox2], dim=-2), dim=-2)  # (B, 3)
    
    coords = batch_linspace(bmin, bmax, nres)  # (B, 3, nres)
    xcoord, ycoord, zcoord = coords.unbind(dim=-2)  # (B, nres), (B, nres), (B, nres)
    
    xgrid, ygrid, zgrid = batch_meshgrid_3d(xcoord, ycoord, zcoord)
    ptsgrid = torch.stack([xgrid, ygrid, zgrid], dim=-1)  # (B, nres, nres, nres, 3)
    pts = ptsgrid.reshape(*B, -1, 3)  # (B, nres^3, 3)
    
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    
    intersect = torch.sum(torch.logical_and(flag1, flag2), dim=-1)
    union = torch.sum(torch.logical_or(flag1, flag2), dim=-1)
    
    union = torch.where(union == 0, intersect, union)  
    
    result = intersect / union.float()  # (B, nres^3)
    
    return result
    
def nocs_iou_3d(bbox1, bbox2):
    """    
    Args:
        bbox_3d_1: (B, 8, 3)
        bbox_3d_2: (B, 8, 3)
        
    Returns:
        IoU (B,) 
    """
    
    bbox_1_max = torch.amax(bbox1, dim=-2)  # (B, 3)
    bbox_1_min = torch.amin(bbox1, dim=-2)  # (B, 3)
    bbox_2_max = torch.amax(bbox2, dim=-2)  # (B, 3)
    bbox_2_min = torch.amin(bbox2, dim=-2)  # (B, 3)
    
    
    overlap_min = torch.maximum(bbox_1_min, bbox_2_min)  # (B, 3)
    overlap_max = torch.minimum(bbox_1_max, bbox_2_max)  # (B, 3)
    
    
    diff = overlap_max - overlap_min  # (B, 3)
    intersections = torch.prod(diff, dim=-1)  # (B,)
    volume1 = torch.prod(bbox_1_max - bbox_1_min, dim=-1)  # (B,)
    volume2 = torch.prod(bbox_2_max - bbox_2_min, dim=-1)  # (B,)
    union = volume1 + volume2 - intersections  # (B,)
    
    iou = intersections / union  # (B,)
    
    mask = torch.amin(overlap_max - overlap_min, dim=-1) < 0
    
    iou = torch.where(mask, torch.zeros_like(iou), iou) 
    
    return iou

def get_bbox(points:torch.Tensor):
    '''
    Args:
        points: (..., N, 3)
        
    Returns:
        bbox: (..., 8, 3) arranged according to order2
        order1: (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
               Z
               |
            3-----0
           /|    /|
          2-----1 |
          | |   | |
          | 7---|-4 
          |/    |/  ---> Y
          6-----5
            /
           X 
        order2: (-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)
               Z
               |
            0-----1
           /|    /|
          3-----2 |
          | |   | |
          | 4---|-5 
          |/    |/  ---> Y
          7-----6
            /
           X 
    '''
    points_min = torch.amin(points, dim=-2)  # (..., 3)
    points_max = torch.amax(points, dim=-2)  # (..., 3)
    
    
    template = torch.tensor([
        [-1, -1,  1],  
        [-1,  1,  1],  
        [ 1,  1,  1],  
        [ 1, -1,  1],  
        [-1, -1, -1],  
        [-1,  1, -1],  
        [ 1,  1, -1],  
        [ 1, -1, -1],  
    ], dtype=points.dtype, device=points.device)
    
    template_01 = (template + 1) / 2
    
    batch_dims = points_min.shape[:-1]
    template_01 = template_01.view(*([1] * len(batch_dims)), 8, 3)
    
    points_min = points_min.unsqueeze(-2)  # (..., 1, 3)
    points_max = points_max.unsqueeze(-2)  # (..., 1, 3)
    
    bbox = points_min + (points_max - points_min) * template_01
    
    return bbox  

def get_bbox_seg(points:torch.Tensor, cls:torch.Tensor):
    '''
    Args:
        points: (..., N, 3)
        cls: (..., N, 1)
    Returns:
        bbox: (..., 8, 3) arranged according to order2
        order1: (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
               Z
               |
            3-----0
           /|    /|
          2-----1 |
          | |   | |
          | 7---|-4 
          |/    |/  ---> Y
          6-----5
            /
           X 
        order2: (-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)
               Z
               |
            0-----1
           /|    /|
          3-----2 |
          | |   | |
          | 4---|-5 
          |/    |/  ---> Y
          7-----6
            /
           X 
    '''
    bboxs = []
    for ins in torch.unique(cls):
    
        mask = cls == ins
        
        this_ins_points_pinf = torch.where(mask, points, torch.inf)
        this_ins_points_ninf = torch.where(mask, points, -torch.inf)
    
        points_min = torch.amin(this_ins_points_pinf, dim=-2)  # (..., 3)
        points_max = torch.amax(this_ins_points_ninf, dim=-2)  # (..., 3)
        
        points_max = torch.nan_to_num(points_max, nan=0, posinf=0, neginf=0)
        points_min = torch.nan_to_num(points_min, nan=0, posinf=0, neginf=0)
        
        template = torch.tensor([
            [-1, -1,  1],  
            [-1,  1,  1],  
            [ 1,  1,  1],  
            [ 1, -1,  1],  
            [-1, -1, -1],  
            [-1,  1, -1],  
            [ 1,  1, -1],  
            [ 1, -1, -1],  
        ], dtype=points.dtype, device=points.device)
        
        template_01 = (template + 1) / 2
        
        batch_dims = points_min.shape[:-1]
        template_01 = template_01.view(*([1] * len(batch_dims)), 8, 3)
        
        points_min = points_min.unsqueeze(-2)  # (..., 1, 3)
        points_max = points_max.unsqueeze(-2)  # (..., 1, 3)
        
        bbox = points_min + (points_max - points_min) * template_01
        
        bboxs.append(bbox)
        
    return torch.stack(bboxs, dim=-3)  # (B, 8, 3)


def get_bbox_from_offset_and_scale(offset: torch.Tensor, scale_per_part: torch.Tensor):
    '''    
    Args:
        offset: Tensor of shape (..., 3)
        scale_per_part: Tensor of shape (..., 3)
    
    Returns:
        bbox: Tensor of shape (..., 8, 3) with vertices arranged according to order2
               Z
               |
            1-----2
           /|    /|
          0-----3 |
          | |   | |
          | 5---|-6 
          |/    |/  ---> Y
          4-----7
            /
           X 
    '''
    template = torch.tensor([
        [-1, -1,  1],  
        [-1,  1,  1],  
        [ 1,  1,  1],  
        [ 1, -1,  1],  
        [-1, -1, -1],  
        [-1,  1, -1],  
        [ 1,  1, -1],  
        [ 1, -1, -1],  
    ], dtype=offset.dtype, device=offset.device)
    
    batch_dims = offset.shape[:-1]
    template = template.view(*([1] * len(batch_dims)), 8, 3)
    
    scale_per_part = scale_per_part.unsqueeze(-2)  # (..., 1, 3)
    offset = offset.unsqueeze(-2)  # (..., 1, 3)
    
    bbox = template * scale_per_part + offset
    
    return bbox  


def get_bbox_from_corners(corners: torch.Tensor):
    '''
    
    Args:
        corners: Tensor of shape (..., 6) - [xmin, xmax, ymin, ymax, zmin, zmax]
    
    Returns:
        bbox: Tensor of shape (..., 8, 3) with vertices arranged according to order2
               Z
               |
            1-----2
           /|    /|
          0-----3 |
          | |   | |
          | 5---|-6 
          |/    |/  ---> Y
          4-----7
            /
           X 
        
        Vertex order:
        0: (xmin, ymin, zmax) - front bottom left
        1: (xmin, ymax, zmax) - front top left
        2: (xmax, ymax, zmax) - front top right
        3: (xmax, ymin, zmax) - front bottom right
        4: (xmin, ymin, zmin) - back bottom left
        5: (xmin, ymax, zmin) - back top left
        6: (xmax, ymax, zmin) - back top right
        7: (xmax, ymin, zmin) - back bottom right
    '''
    assert corners.shape[-1] == 6, "corners tensor must have shape (..., 6)"
    
    xmin, xmax, ymin, ymax, zmin, zmax = torch.split(corners, 1, dim=-1)
    
    vertices = [
        torch.cat([xmin, ymin, zmax], dim=-1),  # 0: front bottom left
        torch.cat([xmin, ymax, zmax], dim=-1),  # 1: front top left
        torch.cat([xmax, ymax, zmax], dim=-1),  # 2: front top right
        torch.cat([xmax, ymin, zmax], dim=-1),  # 3: front bottom right
        torch.cat([xmin, ymin, zmin], dim=-1),  # 4: back bottom left
        torch.cat([xmin, ymax, zmin], dim=-1),  # 5: back top left
        torch.cat([xmax, ymax, zmin], dim=-1),  # 6: back top right
        torch.cat([xmax, ymin, zmin], dim=-1),  # 7: back bottom right
    ]
    
    bbox = torch.stack(vertices, dim=-2)  # (..., 8, 3)
    
    return bbox


def save_matrix_as_colored_image(matrix: np.ndarray, output_path: str):
    """
    Saves a 2D matrix as a colored image.

    Args:
        matrix (np.ndarray): Input uint16 (e.g., depth map) or bool (e.g., mask) 2D matrix.
        output_path (str): Output image save path (e.g., 'colored_image.png').
    """
    if matrix.ndim != 2:
        raise ValueError(f"Input matrix must be 2D, but got {matrix.ndim}D")

    # Process the matrix based on its type
    if matrix.dtype == np.uint16:
        # 1a. normalize uint16 matrix to 0-255 range.
        normalized_matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif matrix.dtype == bool:
        # 1b. Convert bool matrix to black and white image (False->0, True->255).
        normalized_matrix = (matrix.astype(np.uint8) * 255)
    else:
        raise TypeError(f"Unsupported matrix type: {matrix.dtype}. Only uint16 and bool are supported.")

    # 2. Apply a colormap (e.g., JET) to convert the grayscale image to a color image.
    # JET colormap maps 0 to blue and 255 to red.
    colored_matrix = cv2.applyColorMap(normalized_matrix, cv2.COLORMAP_JET)

    # 3. Save the image.
    cv2.imwrite(output_path, colored_matrix)
    print(f"Colored image saved to: {output_path}")
    
def overlay_mask_on_rgb(
    rgb_image: np.ndarray, 
    mask_image: np.ndarray, 
    alpha: float = 0.5, 
    brightness_factor: float = 0.6
) -> np.ndarray:
    """
    Overlay a color mask on an RGB image.

    - In the valid areas of the mask (non-black), the mask color is blended onto the RGB image with transparency.
    - In the background areas of the mask (black), the brightness of the RGB image is reduced.

    Args:
        rgb_image (np.ndarray): HxWx3 uint8 RGB Image.
        mask_image (np.ndarray): HxWx3 uint8 Mask Image. Black (0,0,0) represents the background.
        alpha (float, optional): Overlay transparency, between 0 and 1. Default is 0.5.
        brightness_factor (float, optional): Brightness factor for background areas, between 0 and 1. Default is 0.6.

    Returns:
        np.ndarray: HxWx3 uint8 format of the final composite image.
    """
    # --- 1. Input validation ---
    if rgb_image.shape != mask_image.shape:
        raise ValueError("RGB image and mask image must have the same dimensions.")
    if rgb_image.dtype != np.uint8 or mask_image.dtype != np.uint8:
        raise TypeError("Both images must be of type uint8.")

    # --- 2. Identify valid mask areas ---
    # Create a boolean mask where True represents non-black pixels
    # np.any(mask_image != [0, 0, 0], axis=-1) checks if at least one color channel is non-zero
    is_valid_mask_area = np.any(mask_image != [0, 0, 0], axis=-1)

    # --- 3. Create a copy of the output image ---
    output_image = rgb_image.copy()

    # --- 4. Process background areas (reduce brightness) ---
    # Select areas where the mask is black
    background_area = ~is_valid_mask_area
    # To avoid data type overflow, convert to float, multiply by the factor, then back to uint8
    output_image[background_area] = (
        output_image[background_area].astype(np.float32) * brightness_factor
    ).astype(np.uint8)

    # --- 5. Process valid mask areas (overlay with transparency) ---
    # Select areas where the mask is colored
    foreground_area = is_valid_mask_area
    # Apply alpha blending formula: final = alpha * foreground + (1 - alpha) * background
    output_image[foreground_area] = cv2.addWeighted(
        src1=mask_image[foreground_area],
        alpha=alpha,
        src2=rgb_image[foreground_area],
        beta=1 - alpha,
        gamma=0
    )

    return output_image


from skimage.measure import marching_cubes

def create_grid_points_from_bounds(minimun, maximun, res):
    x = np.linspace(minimun, maximun, res)
    y = np.linspace(minimun, maximun, res)
    z = np.linspace(minimun, maximun, res)
    grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return grid_points.astype(np.float32)


def prepare_marching_cubes_points(grid_size=64, device='cuda'):
    
    grid_points = create_grid_points_from_bounds(-1.0, 1.0, grid_size)
    grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float32)
    grid_points = grid_points.view(1, -1, 3)
    return grid_points

def marching_cubes_from_sdf(sdf_pred, grid_size=64, level=0.0):
    verts = []
    faces = []
    for b in range(sdf_pred.shape[0]):
        sdf_values = sdf_pred[b].cpu().numpy().reshape(grid_size, grid_size, grid_size)
        verts_b, faces_b, _, _ = marching_cubes(sdf_values, level=level)
        verts_b = verts_b / (grid_size - 1) * 2 - 1.0
        verts.append(verts_b)
        faces.append(faces_b)
    return verts, faces


def average_quaternion_torch(Q: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the weighted average of a set of unit quaternions using PyTorch.
    Args:
        Q (torch.Tensor): Tensor of shape (N, 4) containing quaternions in (w, x, y, z) format.
        weights (torch.Tensor, optional): Tensor of shape (N,) containing non-negative weights.

    Returns:
        torch.Tensor: Tensor of shape (4,) representing the averaged quaternion (w, x, y, z).
    """
    assert Q.ndim == 2 and Q.shape[1] == 4, "Q must be a Nx4 tensor"
    N = Q.shape[0]
    
    if weights is None:
        weights = torch.ones(N, dtype=Q.dtype, device=Q.device)
    else:
        assert weights.shape[0] == N, "Weights must have shape (N,)"
        weights = weights.to(Q.dtype)

    # Normalize weights to sum to 1
    weights = weights / torch.sum(weights)

    # Ensure all quaternions are on the same hemisphere
    signs = torch.where(Q[:, 0] < 0, -1.0, 1.0).unsqueeze(1)  # Shape: (N, 1)
    Q = Q * signs  # Flip quaternions where w < 0

    # Vectorized computation of accumulator matrix A (4x4)
    # Use einsum for efficient outer product computation
    A = torch.einsum('n,nk,nl->kl', weights, Q, Q)

    # Compute the eigenvector corresponding to the largest eigenvalue
    eigenvalues, eigenvectors = torch.linalg.eigh(A)  # Returns ascending order
    q_avg = eigenvectors[:, -1]  # Eigenvector with largest eigenvalue

    # Ensure final quaternion has positive scalar part (w > 0)
    sign = torch.where(q_avg[0] < 0, -1.0, 1.0)
    q_avg = q_avg * sign

    return q_avg


def average_quaternion_torch_batch(Q: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the weighted average of a set of unit quaternions using PyTorch.
    Args:
        Q (torch.Tensor): Tensor of shape (B, N, 4) containing quaternions in (w, x, y, z) format.
        weights (torch.Tensor, optional): Tensor of shape (B, N) containing non-negative weights.
    Returns:
        torch.Tensor: Tensor of shape (B, 4) representing the averaged quaternion (w, x, y, z).
    """
    assert Q.ndim == 3 and Q.shape[2] == 4, "Q must be a BxNx4 tensor"
    B, N, _ = Q.shape

    if weights is None:
        weights = torch.ones(B, N, dtype=Q.dtype, device=Q.device)
    else:
        assert weights.shape[0] == B and weights.shape[1] == N, "Weights must have shape (B, N)"
        weights = weights.to(Q.dtype)

    # Normalize weights for each batch
    weights = weights / torch.sum(weights, dim=1, keepdim=True)

    # Ensure all quaternions are on the same hemisphere
    signs = torch.where(Q[:, :, 0] < 0, -1.0, 1.0).unsqueeze(2)  # Shape: (B, N, 1)
    Q = Q * signs  # Flip quaternions where w < 0

    # Vectorized computation for each batch
    # Use einsum for efficient batch outer product computation
    A = torch.einsum('bn,bnk,bnl->bkl', weights, Q, Q)  # Shape: (B, 4, 4)

    # Compute eigenvectors for each batch
    eigenvalues, eigenvectors = torch.linalg.eigh(A)  # Returns ascending order
    q_avg = eigenvectors[:, :, -1]  # Eigenvector with largest eigenvalue for each batch

    # Ensure final quaternions have positive scalar part (w > 0)
    sign = torch.where(q_avg[:, 0] < 0, -1.0, 1.0).unsqueeze(1)
    q_avg = q_avg * sign

    return q_avg


def get_comet_exp(project_name, prefix=''):
    print('Comet ML is enabled, initializing...')
    from comet_ml import Experiment
    exp = Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name=project_name,
            workspace=os.environ["COMET_WORKSPACE"]
        )
    exp.set_name(exp.get_name() + '_' + prefix)
    return exp, exp.get_name()



def readable_dict(data, markdown=False) -> str:
    """
    Generates a formatted tree-like string representation of a dictionary.
    Keys are displayed in bold blue.
    
    Args:
        data (dict): The dictionary-like object to visualize.
        
    Returns:
        str: The formatted string representation.
    """
    # ANSI escape codes for styling
    BLUE_BOLD = "\033[1;34m" if not markdown else ""
    RESET = "\033[0m" if not markdown else ""
    LINE = '|' if not markdown else ' '
    LAST_LINE = '`' if not markdown else ' '
    
    output = []

    def _build_tree(current_data, prefix=""):
        # Handle non-dict inputs gracefully if passed recursively, 
        # though the main logic handles dict checks before recursion.
        # if not isinstance(current_data, dict):
        #     return
        if not hasattr(current_data, 'keys'):
            return 'data is not a dict-like object'

        keys = list(current_data.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = f"{LAST_LINE}- " if is_last else f"{LINE}- "
            
            value = current_data[key]
            formatted_key = f"{BLUE_BOLD}{key}{RESET}"
            
            if hasattr(value, 'keys'):
                output.append(f"{prefix}{connector}{formatted_key}")
                # Prepare prefix for children: 
                # if this is the last item, children don't need the vertical bar
                new_prefix = prefix + ("    " if is_last else f"{LINE}   ")
                _build_tree(value, new_prefix)
            else:
                output.append(f"{prefix}{connector}{formatted_key}: {value}")

    _build_tree(data)
    return "\n".join(output)


def empirical_kl_loss(features:torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    Computes the empirical KL divergence between the distribution of features and a standard normal distribution.
    
    Args:
        features: Tensor of shape (B, D)
    Returns:
        kl_loss: scalar tensor
    """
    batch_mean = features.mean(dim=0)  # (D,)
    batch_var = features.var(dim=0)    # (D,)
    
    # 2. calculate KL(N(mu, var) || N(0, 1))
    # Formula: 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    kl_loss = 0.5 * (batch_var + batch_mean**2 - 1 - torch.log(batch_var + eps))
    
    return kl_loss.mean() # or .sum() depending on loss scale


def get_trend_arrow(current, previous, lower_is_better=False):

    GREEN = '\033[32m'
    RED = '\033[31m'
    RESET = '\033[0m'
    
    diff = current - previous
    
    if abs(diff) < 1e-9:
        return "="

    if diff > 0:
        symbol = "↑"
        color = RED if lower_is_better else GREEN
    else:
        symbol = "↓"
        color = GREEN if lower_is_better else RED

    return f"{color}{symbol}{RESET}"